import pymbar
from fe import endpoint_correction
from collections import namedtuple

import time
import functools
import copy
import jax

import numpy as np

from typing import Tuple, List, Any
import os

from fe import standard_state
from fe.estimator_common import SimulationResult, simulate

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
     "barostat",
     "lambda_schedule",
     "equil_steps",
     "prod_steps",
     "beta",
     "prefix",
     # "cache_results",
     # "cache_lambda" # if lambda < cache_lambda, then the simulation is re-ran
    ]
)

gradient = List[Any] # TODO: make this more descriptive of dG_grad structure

def _deltaG(model, sys_params) -> Tuple[Tuple[float, List], np.array]:

    assert len(sys_params) == len(model.unbound_potentials)

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, model.unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    all_args = []
    for lamb_idx, lamb in enumerate(model.lambda_schedule):

        x_interval = 500

        all_args.append((
            lamb,
            model.box,
            model.x0,
            model.v0,
            bound_potentials,
            model.integrator,
            model.barostat,
            model.equil_steps,
            model.prod_steps,
            x_interval
        ))

    if model.endpoint_correct:
        all_args.append((
            1.0,
            model.box,
            model.x0,
            model.v0,
            bound_potentials[:-1],
            model.integrator,
            model.barostat,
            model.equil_steps,
            model.prod_steps,
            x_interval
        ))

    # assert len(all_args) == len(model.cache_results)

    # if model.client is None:
    #     results = []
    #     for args, cache in zip(all_args, model.cache_results):
    #         if cache is None or args[0] <= model.cache_lambda:
    #             results.append(simulate(*args))
    #         else:
    #             results.append(cache)
    # else:
    #     futures = []
    #     for args, cache in zip(all_args, model.cache_results):
    #         if cache is None or args[0] <= model.cache_lambda:
    #             futures.append(model.client.submit(simulate, *args))
    #         else:
    #             futures.append(None)

    #     results = []
    #     for future, cache in zip(futures, model.cache_results):
    #         if future is None:
    #             results.append(cache)
    #         else:
    #             results.append(future.result())


    if model.client is None:
        results = []
        for args in all_args:
            results.append(simulate(*args))
    else:
        futures = []
        for args in all_args:
            futures.append(model.client.submit(simulate, *args))

        results = []
        for future in futures:
            results.append(future.result())

    mean_du_dls = []
    all_grads = []

    if model.endpoint_correct:
        ti_results = results[:-1]
    else:
        ti_results = results

    for lambda_idx, (lambda_window, result) in enumerate(zip(model.lambda_schedule, ti_results)):
        # (ytz): figure out what to do with stddev(du_dl) later
        print(f"{model.prefix} index {lambda_idx} lambda {lambda_window:.5f} <du/dl> {np.mean(result.du_dls):.5f} med(du/dl) {np.median(result.du_dls):.5f}  o(du/dl) {np.std(result.du_dls):.5f}")
        mean_du_dls.append(np.mean(result.du_dls))
        all_grads.append(result.du_dps)

    dG = np.trapz(mean_du_dls, model.lambda_schedule)
    dG_grad = []
    for rhs, lhs in zip(all_grads[-1], all_grads[0]):
        dG_grad.append(rhs - lhs)

    if model.endpoint_correct:
        core_restr = bound_potentials[-1]
        # (ytz): tbd, automatically find optimal k_translation/k_rotation such that
        # standard deviation and/or overlap is maximized
        k_translation = 200.0
        k_rotation = 100.0
        start = time.time()
        lhs_du, rhs_du, _, _ = endpoint_correction.estimate_delta_us(
            k_translation=k_translation,
            k_rotation=k_rotation,
            core_idxs=core_restr.get_idxs(),
            core_params=core_restr.params.reshape((-1,2)),
            beta=model.beta,
            lhs_xs=results[-2].xs,
            rhs_xs=results[-1].xs
        )
        dG_endpoint = pymbar.BAR(model.beta*lhs_du, model.beta*np.array(rhs_du))[0]/model.beta
        # compute standard state corrections for translation and rotation
        dG_ssc_translation, dG_ssc_rotation = standard_state.release_orientational_restraints(
            k_translation,
            k_rotation,
            model.beta
        )
        overlap = endpoint_correction.overlap_from_cdf(lhs_du, rhs_du)
        print(f"{model.prefix} dG_ti {dG:.3f} dG_endpoint {dG_endpoint:.3f} dG_ssc_translation {dG_ssc_translation:.3f} dG_ssc_rotation {dG_ssc_rotation:.3f} overlap {overlap:.3f} time: {time.time()-start:.3f}s")
        dG += dG_endpoint + dG_ssc_translation + dG_ssc_rotation
    else:
        print(f"{model.prefix} dG_ti {dG:.3f}")

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