from collections import namedtuple
import pymbar

from scipy.stats import special_ortho_group
import functools
import copy
import jax
import numpy as np
from timemachine import constants

from timemachine.lib import potentials, custom_ops
from timemachine.potentials import bonded

from typing import Tuple, List, Any

import dataclasses
import jax.numpy as jnp

import pickle

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

def simulate(debug_file_info, lamb, box, x0, v0, final_potentials, integrator, equil_steps, prod_steps,
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
    # bonded_impls = []
    # nonbonded_impls = []

    # set up observables for du_dps here as well.
    du_dp_obs = []

    for bp_idx, bp in enumerate(final_potentials):
        impl = bp.bound_impl(np.float32)
        if isinstance(bp, potentials.Nonbonded):
            nb_bp = bp
            # nonbonded_impls.append(impl)
        # else
            # bonded_impls.append(impl)
        all_impls.append(impl)
        du_dp_obs.append(custom_ops.AvgPartialUPartialParam(impl, 5))

    if integrator.seed == 0:
        integrator = copy.deepcopy(integrator)
        integrator.seed = np.random.randint(np.iinfo(np.int32).max)

    intg_impl = integrator.impl()
    # context components: positions, velocities, box, integrator, energy fxns

    binary = {
        "integrator": integrator,
        "x0": x0,
        "v0": v0,
        "final_potentials": final_potentials,
        "lamb": lamb,
        "box": box,
        "equil_steps": equil_steps,
        "prod_steps": prod_steps,
        "x_interval": x_interval,
        "du_dl_interval": du_dl_interval
    }

    pickle.dump(binary,
        open("initial_state_"+debug_file_info+".pkl", "wb")
    )

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

    result = SimulationResult(xs=xs, du_dls=full_du_dls, du_dps=grads)
    return result


FreeEnergyModel = namedtuple(
    "FreeEnergyModel",
    ["unbound_potentials", "client", "box", "x0", "v0", "integrator", "lambda_schedule", "equil_steps", "prod_steps", "topology", "stage", "debug_info"]
)

gradient = List[Any] # TODO: make this more descriptive of dG_grad structure

# FIX
TEMPERATURE = 300.0
BETA = 1/(constants.BOLTZ*TEMPERATURE)

def exp_u(rotation, k):
    return jnp.exp(-BETA*bonded.psi(rotation, k))

exp_batch = jax.jit(jax.vmap(exp_u, (0, None)))

def sample_multiple_rotations(k, size):
    num_batches = 500
    batch_size = 10000

    samples = []

    for batch_attempt in range(num_batches):
        Rs = special_ortho_group.rvs(3, size=batch_size)
        tests = np.random.rand(batch_size)
        M = np.pi**2 # volume of SO(3)
        # acceptance_prob = exp_batch(Rs, k)/M
        acceptance_prob = exp_batch(Rs, k)
        locations = np.argwhere(tests < acceptance_prob).reshape(-1)

        samples.append(Rs[locations])
        if sum([len(x) for x in samples]) > size:
            break

    return np.concatenate(samples)[:size]

def endpoint_correction(
    k_translation,
    k_rotation,
    core_idxs,
    core_params,
    lhs_xs,
    rhs_xs):
    """
    Compute the BAR re-weighted end-point correction of converting an intractable core
    restraint into a tractable RMSD-based orientational restraint.
    """

    k_translation = 200.0
    k_rotation = 100.0

    # core_restraint = bound_potentials[-1]
    # core_idxs = core_restraint.get_idxs()

    box = np.eye(3) * 100.0
    core_restr = functools.partial(
        bonded.harmonic_bond,
        bond_idxs=core_idxs,
        params=core_params,
        # core_restraint.params.reshape((-1, 2)),
        box=box,
        lamb=None
    )

    # center of mass translational restraints
    restr_group_idxs_a = core_idxs[:, 0]
    restr_group_idxs_b = core_idxs[:, 1]

    # disjoint sets
    assert len(set(restr_group_idxs_a.tolist()).intersection(set(restr_group_idxs_b.tolist()))) == 0

    translation_restr = functools.partial(
        bonded.centroid_restraint,
        group_a_idxs=restr_group_idxs_a,
        group_b_idxs=restr_group_idxs_b,
        params=None,
        kb=k_translation,
        b0=0.0,
        box=box,
        lamb=None
    )

    # never generate a force with this
    rotation_restr = functools.partial(
        bonded.rmsd_restraint,
        params=None,
        group_a_idxs=restr_group_idxs_a,
        group_b_idxs=restr_group_idxs_b,
        k=k_rotation,
        box=box,
        lamb=None
    )

    def delta_u_fn(x_t):
        return translation_restr(x_t) + rotation_restr(x_t) - core_restr(x_t)

    delta_u_batch = jax.jit(jax.vmap(delta_u_fn))

    # import mdtraj
    # md_topology = mdtraj.Topology.from_openmm(model.topology)
    # traj = mdtraj.Trajectory(lhs_xs, md_topology)
    # traj.save_xtc(debug_prefix+"lhs.xtc")

    # md_topology = mdtraj.Topology.from_openmm(model.topology)
    # traj = mdtraj.Trajectory(rhs_xs, md_topology)
    # traj.save_xtc(debug_prefix+"rhs.xtc")

    lhs_du = delta_u_batch(lhs_xs)

    sample_size = rhs_xs.shape[0]
    rotation_samples = sample_multiple_rotations(k_rotation, sample_size)
    covariance = np.eye(3)/(2*BETA*k_translation)
    translation_samples = np.random.multivariate_normal((0,0,0), covariance, sample_size)

    rhs_xs_aligned = []
    rhs_du = []
    for x, r, t in zip(rhs_xs, rotation_samples, translation_samples):
        x_a, x_b = rmsd_align(x[restr_group_idxs_a], x[restr_group_idxs_b])
        x_b = x_b@r.T + t
        x_new = np.copy(x)
        # group_idxs_a and group_idxs_b must be disjoint
        x_new[restr_group_idxs_a] = x_a
        x_new[restr_group_idxs_b] = x_b
        rhs_xs_aligned.append(x_new)

    rhs_xs_aligned = np.array(rhs_xs_aligned)
    rhs_du = delta_u_batch(rhs_xs_aligned)

    dG_endpoint = pymbar.BAR(BETA*lhs_du, -BETA*np.array(rhs_du))[0]/BETA
    return dG_endpoint, lhs_du, rhs_du

def rmsd_align(x1, x2):
    com1 = np.mean(x1, axis=0)
    com2 = np.mean(x2, axis=0)

    x1 = x1 - com1
    x2 = x2 - com2

    # x1 and x2 must be already mean aligned.
    correlation_matrix = np.dot(x2.T, x1)
    U, S, V_tr = np.linalg.svd(correlation_matrix, full_matrices=False)
    is_reflection = (np.linalg.det(U) * np.linalg.det(V_tr)) < 0.0
    U = jax.ops.index_update(U,
        jax.ops.index[:, -1],
        np.where(is_reflection, -U[:, -1], U[:, -1])
    )
    rotation = np.dot(U, V_tr)

    # xa = x_t[:NA] - com1
    # xb = x_t[NA:] - com2
    xa = x1
    xb = x2@rotation

    return xa, xb

def _deltaG(model, sys_params) -> Tuple[Tuple[float, List], np.array]:

    assert len(sys_params) == len(model.unbound_potentials)

    # last bound_potential is a restraining potential, this should be turned off at the last lambda window
    bound_potentials = []
    for params, unbound_pot in zip(sys_params, model.unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    debug_prefix = model.debug_info + "_stage_" + model.stage + "_"

    x_interval = 50

    if model.client is None:
        assert 0
        results = []
        for lamb_idx, lamb in enumerate(model.lambda_schedule):
            results.append(simulate(debug_prefix+"lambda_idx_"+str(lamb_idx), lamb, model.box, model.x0, model.v0, bound_potentials, model.integrator, model.equil_steps, model.prod_steps, x_interval))
    else:
        futures = []
        for lamb_idx, lamb in enumerate(model.lambda_schedule):
            args = (debug_prefix+"lambda_idx_"+str(lamb_idx), lamb, model.box, model.x0, model.v0, bound_potentials, model.integrator, model.equil_steps, model.prod_steps, x_interval)
            futures.append(model.client.submit(simulate, *args))

        # add an unrestrained simulation for the end-point correction
        # endpoint-correction has the restraint turned off
        args = (debug_prefix+"lambda_idx_independent_restraint", 1.0, model.box, model.x0, model.v0, bound_potentials[:-1], model.integrator, model.equil_steps, model.prod_steps, x_interval)
        futures.append(model.client.submit(simulate, *args))

        results = [x.result() for x in futures]

    mean_du_dls = []
    std_du_dls = []
    all_grads = []

    for lamb_idx, result in enumerate(results[:-1]):
        # (ytz): figure out what to do with stddev(du_dl) later
        # print(debug_prefix+"lambda_idx_"+str(lamb_idx), "lambda", model.lambda_schedule[lamb_idx], "avg du_dl", np.mean(result.du_dls), "std du_dl", np.std(result.du_dls))
        mean_du_dls.append(np.mean(result.du_dls))
        std_du_dls.append(np.std(result.du_dls))
        all_grads.append(result.du_dps)

    core_restr = bound_potentials[-1]
    for x, y, z in zip(model.lambda_schedule, mean_du_dls, std_du_dls):
        print(f'{debug_prefix}du_dl_ti lambda {x:5.3f} <du/dl> {y:5.3f} o(du/dl) {z:5.3f}')
    dG_ti = np.trapz(mean_du_dls, model.lambda_schedule)
    dG_endpoint, lhs_du, rhs_du = endpoint_correction(
        k_translation=200.0,
        k_rotation=100.0,
        core_idxs=core_restr.get_idxs(),
        core_params=core_restr.params.reshape((-1,2)),
        lhs_xs=results[-2].xs,
        rhs_xs=results[-1].xs
    )

    import matplotlib.pyplot as plt

    plt.clf()
    plt.hist(lhs_du, alpha=0.5, density=True, label='lhs')
    plt.hist(rhs_du, alpha=0.5, density=True, label='rhs')
    plt.xlim(-100, 10)
    plt.savefig(debug_prefix+"overlap")

    print(debug_prefix, "dG_ti", dG_ti, "dG_endpoint", dG_endpoint)
    dG = dG_ti + dG_endpoint

    dG_grad = []
    for rhs, lhs in zip(all_grads[-1], all_grads[0]):
        dG_grad.append(rhs - lhs)

    return (dG, results), dG_grad

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