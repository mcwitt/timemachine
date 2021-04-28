
import pymbar
from fe import endpoint_correction
from collections import namedtuple

import functools
import copy
import jax
import numpy as np

from timemachine.lib import potentials, custom_ops

from typing import Tuple, List, Any

import dataclasses
import jax.numpy as jnp

import os

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

    # print("SIMULATING WITH DEVICE", os.environ['CUDA_VISIBLE_DEVICES'])
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

    result = SimulationResult(xs=xs, du_dls=full_du_dls, du_dps=grads)
    return result


FreeEnergyModel = namedtuple(
    "FreeEnergyModel",
    [
     "unbound_potentials",
     "endpoint_correct",
     "client",
     "box",
     "x0",
     "v0",
     "integrator",
     "lambda_schedule",
     "equil_steps",
     "prod_steps",
     "beta",
     "prefix"
    ]
)

gradient = List[Any] # TODO: make this more descriptive of dG_grad structure

def _deltaG(model, sys_params) -> Tuple[Tuple[float, List], np.array]:

    assert len(sys_params) == len(model.unbound_potentials)

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, model.unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    # if endpoint-correction is turned on, it is assumed that the last unbound_potential corresponds to the restraining potential

    if model.client is None:
        assert 0
        results = []
        for lamb in model.lambda_schedule:
            results.append(simulate(lamb, model.box, model.x0, model.v0, bound_potentials, model.integrator, model.equil_steps, model.prod_steps))

        if model.endpoint_correct:
            results.append(simulate(1.0, model.box, model.x0, model.v0, bound_potentials[:-1], model.integrator, model.equil_steps, model.prod_steps))

    else:
        futures = []
        for lamb in model.lambda_schedule:
            args = (lamb, model.box, model.x0, model.v0, bound_potentials, model.integrator, model.equil_steps, model.prod_steps)
            futures.append(model.client.submit(simulate, *args))

        if model.endpoint_correct:
            args = (1.0, model.box, model.x0, model.v0, bound_potentials, model.integrator, model.equil_steps, model.prod_steps)
            futures.append(model.client.submit(simulate, *args))

        results = [x.result() for x in futures]

    mean_du_dls = []
    all_grads = []

    if model.endpoint_correct:
        endpoint_results = results[-1]
        ti_results = results[:-1]
    else:
        ti_results = results

    for lambda_window, result in zip(model.lambda_schedule, ti_results):
        # (ytz): figure out what to do with stddev(du_dl) later
        print(f"{model.prefix} lambda {lambda_window:.3f} <du/dl> {np.mean(result.du_dls):.3f} o(du/dl) {np.std(result.du_dls):.3f}")
        mean_du_dls.append(np.mean(result.du_dls))
        all_grads.append(result.du_dps)

    dG = np.trapz(mean_du_dls, model.lambda_schedule)
    dG_grad = []
    for rhs, lhs in zip(all_grads[-1], all_grads[0]):
        dG_grad.append(rhs - lhs)

    if model.endpoint_correct:
        core_restr = bound_potentials[-1]
        lhs_du, rhs_du, _, _ = endpoint_correction.estimate_delta_us(
            k_translation=200.0,
            k_rotation=100.0,
            core_idxs=core_restr.get_idxs(),
            core_params=core_restr.params.reshape((-1,2)),
            beta=model.beta,
            lhs_xs=results[-2].xs,
            rhs_xs=results[-1].xs
        )
        dG_endpoint = pymbar.BAR(model.beta*lhs_du, model.beta*np.array(rhs_du))[0]/model.beta
        overlap = endpoint_correction.overlap_from_cdf(lhs_du, rhs_du)
        print(f"{model.prefix} dG_endpoint {dG_endpoint:.3f} overlap {overlap:.3f}")
        dG += dG_endpoint

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