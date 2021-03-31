
import pymbar

import multiprocessing
from collections import namedtuple

import functools
import copy
import jax
import numpy as np

from timemachine.lib import potentials, custom_ops

from typing import Tuple, List, Any

import dataclasses
import jax.numpy as jnp

@dataclasses.dataclass
class SimulationResult:
   xs: np.array
   du_dls: np.array
   du_dps: np.array

def flatten(v):
    return tuple(), (v.xs, v.du_dls, v.du_dps)

def unflatten(aux_data, children):
    xs, du_dls, du_dps = aux_data
    return SimulationResult(xs, du_dls, du_dps)

jax.tree_util.register_pytree_node(SimulationResult, flatten, unflatten)

def simulate(lamb, box, x0, v0, final_potentials, integrator, equil_steps, prod_steps,
    x_interval=50, du_dl_interval=5):
    """
    Run a simulation and collect relevant statistics for this simulation.

    Parameters
    ----------
    lamb: float
        lambda parameter

    box: np.array
        3x3 numpy array of the box, dtype should be np.float64

    x0: np.array
        Nx3 numpy array of the coordinates

    v0: np.array
        Nx3 numpy array of the velocities

    final_potentials: list
        list of unbound potentials

    integrator: timemachine.Integrator
        integrator to be used for dynamics

    equil_steps: int
        number of equilibration steps

    prod_steps: int
        number of production steps

    x_interval: int
        how often we store coordinates. if x_interval == 0 then
        no frames are returned.

    du_dl_interval: int
        how often we store du_dls. if du_dl_interval == 0 then
        no du_dls are returned

    Returns
    -------
    SimulationResult
        Results of the simulation.

    """
    all_impls = []
    bonded_impls = []
    nonbonded_impls = []

    # set up observables for du_dps here as well.
    du_dp_obs = []

    for bp in final_potentials:
        impl = bp.bound_impl(np.float32)
        if isinstance(bp, potentials.Nonbonded):
            nonbonded_impls.append(impl)
        else:
            bonded_impls.append(impl)
        all_impls.append(impl)
        du_dp_obs.append(custom_ops.AvgPartialUPartialParam(impl, 5))

    if integrator.seed == 0:
        integrator = copy.deepcopy(integrator)
        integrator.seed = np.random.randint(np.iinfo(np.int32).max)

    intg_impl = integrator.impl()
    # context components: positions, velocities, box, integrator, energy fxns
    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg_impl,
        all_impls
    )

    # equilibration

    equil_schedule = np.ones(equil_steps)*lamb
    ctxt.multiple_steps(equil_schedule)

    for obs in du_dp_obs:
        ctxt.add_observable(obs)

    prod_schedule = np.ones(prod_steps)*lamb

    full_du_dls, xs = ctxt.multiple_steps(prod_schedule, du_dl_interval, x_interval)

    # keep the structure of grads the same as that of final_potentials so we can properly
    # form their vjps.
    grads = []
    for obs in du_dp_obs:
        grads.append(obs.avg_du_dp())

    # print("xs.shape", xs.shape)

    result = SimulationResult(xs=xs, du_dls=full_du_dls, du_dps=grads)
    return result


FreeEnergyModel = namedtuple(
    "FreeEnergyModel",
    [
     "unbound_potentials_ti",
     "unbound_potentials_end_state", # list of unbound potentials
     "client",
     "box",
     "x0",
     "v0",
     "integrator",
     "lambda_schedule",
     "equil_steps",
     "prod_steps",
     # "num_a_atoms",
     # "num_b_atoms",
     "core",
     "core_params",
     "topology",
     "stage",
     "num_host_atoms"
    ]
)

gradient = List[Any] # TODO: make this more descriptive of dG_grad structure

def _deltaG(model, sys_params) -> Tuple[Tuple[float, List], np.array]:

    print("model.lambda_schedule", model.lambda_schedule)

    if model.client is None:
        results = []
        for lamb_idx, lamb in enumerate(model.lambda_schedule):

            if lamb_idx == len(model.lambda_schedule) - 1:
                ups = model.unbound_potentials_end_state
            else:
                ups = model.unbound_potentials_ti

            bound_potentials = []
            assert len(sys_params) == len(ups)
            for params, unbound_pot in zip(sys_params, ups):
                bp = unbound_pot.bind(np.asarray(params))
                bound_potentials.append(bp)

            results.append(simulate(
                lamb,
                model.box,
                model.x0,
                model.v0,
                bound_potentials,
                model.integrator,
                model.equil_steps,
                model.prod_steps
            ))

            print("lambda_idx", lamb_idx, lamb, "done")
    else:
        assert 0

    # do post-correction here next
    #     futures = []
    #     for lamb, bp in lamb_bp_pairs:
    #         args = (lamb, model.box, model.x0, model.v0, bp, model.integrator, model.equil_steps, model.prod_steps)
    #         futures.append(model.client.submit(simulate, *args))

    #     results = [x.result() for x in futures]

    mean_du_dls = []
    all_grads = []

    for lamb_idx, result in enumerate(results):
        import mdtraj

        md_topology = mdtraj.Topology.from_openmm(model.topology)
        traj = mdtraj.Trajectory(result.xs, md_topology)
        traj.save_xtc(model.stage+"_lambda_"+str(lamb_idx)+".xtc")
        # (ytz): figure out what to do with stddev(du_dl) later
        mean_du_dls.append(np.mean(result.du_dls))
        all_grads.append(result.du_dps)

    dG_ti = np.trapz(mean_du_dls[:-1], model.lambda_schedule[:-1])

    endpoint_core = np.array(model.core)
    endpoint_core[:, 1] += model.num_host_atoms

    # [[1339 8808]
    #  [ 615 8812]
    #  [ 615 8813]
    #  ...
    #  [1034 8837]
    #  [ 864 8838]]

    # lay out linearly
    core_atoms = endpoint_core.T.reshape(-1)

    end_state_xs = results[-1].xs[:, core_atoms]
    restrained_state_xs = results[-2].xs[:, core_atoms]

    import endpoint_correction

    from timemachine.potentials import bonded

    num_core_pairs = endpoint_core.shape[0]

    restr_group_idxs_a = np.arange(num_core_pairs)
    restr_group_idxs_b = np.arange(num_core_pairs) + num_core_pairs

    translation_restr_kb = 200.0
    translation_restr_b0 = 0.0

    translation_restr = functools.partial(
        bonded.centroid_restraint,
        group_a_idxs=restr_group_idxs_a,
        group_b_idxs=restr_group_idxs_b,
        params=None,
        masses=None,
        kb=translation_restr_kb,
        b0=translation_restr_b0
    )

    rotation_restr_kb = 25.0
    rotation_restr = functools.partial(
        bonded.rmsd_restraint,
        params=None,
        group_a_idxs=restr_group_idxs_a,
        group_b_idxs=restr_group_idxs_b,
        k=rotation_restr_kb
    )

    # indexed from 0 to K
    core_idxs = np.stack([
        restr_group_idxs_a,
        restr_group_idxs_b
    ], axis=1)

    # core_idxs = np.array(model.core)
    # core_idxs[:, 1] += model.num_a_atoms

    core_params = model.core_params

    # (ytz): MAKE SURE THIS MATCHES
    # core_params = np.array([
        # [10.0, 0.0],
    # ]*core_idxs.shape[0])

    # print("endpt core_idxs", core_idxs)

    core_restr = functools.partial(
        bonded.harmonic_bond,
        bond_idxs=core_idxs,
        params=core_params
    )

    delta_u_fn = functools.partial(endpoint_correction.delta_u,
        translation_restr=translation_restr,
        rotation_restr=rotation_restr,
        core_restr=core_restr
    )

    lhs_dus = []

    for idx, x in enumerate(restrained_state_xs):
        du = delta_u_fn(x, box=model.box, lamb=lamb)
        lhs_dus.append(du)

    lhs_dus = np.array(lhs_dus)

    from timemachine import constants

    # (Ytz): un-hardcode temperature later
    beta = 1/(constants.BOLTZ*300.0)

    lamb = 0.0
    process_rhs_fn = functools.partial(
        endpoint_correction.process_sample,
        translation_restr_kb=translation_restr_kb,
        NA=len(core_idxs), # (K)
        box=model.box,
        lamb=lamb,
        restr_group_idxs_a=restr_group_idxs_a,
        restr_group_idxs_b=restr_group_idxs_b,
        beta=beta,
        exp_U=endpoint_correction.exp_U,
        rotation_restr=rotation_restr,
        delta_u_partial=delta_u_fn
    )

    pool = multiprocessing.Pool(12)
    processed_results = pool.map(process_rhs_fn, end_state_xs)
    pool.close()

    rhs_dus = np.array([x[1] for x in processed_results])
    rhs_xs = np.array([x[0] for x in processed_results])

    # [K, N, 3]
    cp = np.array(results[-1].xs)

    cp[:, core_atoms, :] = rhs_xs


    md_topology = mdtraj.Topology.from_openmm(model.topology)
    traj = mdtraj.Trajectory(cp, md_topology)
    traj.save_xtc(model.stage+"_lambda_corrected.xtc")

    dG_endpoint = pymbar.BAR(beta*lhs_dus, -beta*np.array(rhs_dus))[0]/beta

    dG_ti = np.trapz(mean_du_dls[:-1], model.lambda_schedule[:-1])

    np.savez(model.stage+"_dus", lhs_dus=lhs_dus, rhs_dus=rhs_dus)

    print("stage", model.stage, "dG_ti", dG_ti, "dG_endpoint", dG_endpoint)

    dG_grad = []
    for rhs, lhs in zip(all_grads[-1], all_grads[0]):
        dG_grad.append(rhs - lhs)

    return (dG_ti + dG_endpoint, results), dG_grad

@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def deltaG(model, sys_params) -> Tuple[float, List]:
    return _deltaG(model=model, sys_params=sys_params)[0]

def deltaG_fwd(model, sys_params) -> Tuple[Tuple[float, List], np.array]:
    """same signature as DeltaG, but returns the full tuple"""
    return _deltaG(model=model, sys_params=sys_params)

def deltaG_bwd(model, residual, grad) -> Tuple[np.array]:
    """Note: nondiff args must appear first here, even though one of them appears last in the original function's signature!
    """
    # residual are the partial dG / partial dparams for each term
    # grad[0] is the adjoint of dG w.r.t. loss: partial L/partial dG
    # grad[1] is the adjoint of dG w.r.t. simulation result, which we don't use
    return ([grad[0]*r for r in residual],)

deltaG.defvjp(deltaG_fwd, deltaG_bwd)