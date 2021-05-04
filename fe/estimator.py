from collections import namedtuple

import functools
import copy
import jax
import numpy as np

from typing import Tuple, List, Any

from fe.estimator_common import SimulationResult, simulate


FreeEnergyModel = namedtuple(
    "FreeEnergyModel",
    ["unbound_potentials",
     "client",
     "box",
     "x0",
     "v0",
     "integrator",
     "lambda_schedule",
     "equil_steps",
     "prod_steps"
    ]
)

gradient = List[Any] # TODO: make this more descriptive of dG_grad structure

def _deltaG(model, sys_params) -> Tuple[Tuple[float, List], np.array]:

    assert len(sys_params) == len(model.unbound_potentials)

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, model.unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    if model.client is None:
        results = []
        for lamb in model.lambda_schedule:
            results.append(simulate(lamb, model.box, model.x0, model.v0, bound_potentials, model.integrator, model.equil_steps, model.prod_steps))
    else:
        futures = []
        for lamb in model.lambda_schedule:
            args = (lamb, model.box, model.x0, model.v0, bound_potentials, model.integrator, model.equil_steps, model.prod_steps)
            futures.append(model.client.submit(simulate, *args))

        results = [x.result() for x in futures]

    mean_du_dls = []
    all_grads = []

    for result in results:
        # (ytz): figure out what to do with stddev(du_dl) later
        mean_du_dls.append(np.mean(result.du_dls))
        all_grads.append(result.du_dps)

    dG = np.trapz(mean_du_dls, model.lambda_schedule)
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