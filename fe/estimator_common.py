# common functions used in the estimator code.

import copy
import jax
import numpy as np
import dataclasses
from md import minimizer

from timemachine.lib import potentials, custom_ops

@dataclasses.dataclass
class SimulationResult:
   xs: np.array
   boxes: np.array
   du_dls: np.array
   du_dps: np.array

def flatten(v):
    return tuple(), (v.xs, v.boxes, v.du_dls, v.du_dps)

def unflatten(aux_data, children):
    xs, boxes, du_dls, du_dps = aux_data
    return SimulationResult(xs, boxes, du_dls, du_dps)

jax.tree_util.register_pytree_node(SimulationResult, flatten, unflatten)

def simulate(lamb, box, x0, v0, final_potentials, integrator, barostat, equil_steps, prod_steps,
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

    barostat: timemachine.Barostat
        barostat to be used for dynamics

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

    # set up observables for du_dps here as well.
    du_dp_obs = []

    for bp in final_potentials:
        impl = bp.bound_impl(np.float32)
        all_impls.append(impl)
        du_dp_obs.append(custom_ops.AvgPartialUPartialParam(impl, 5))

    # fire minimize once again, needed for parameter interpolation
    x0 = minimizer.fire_minimize(x0, all_impls, box, np.ones(100, dtype=np.float64)*lamb)

    # sanity check that forces are well behaved
    for bp in all_impls:
        du_dx, du_dl, u = bp.execute(x0, box, lamb)
        norm_forces = np.linalg.norm(du_dx, axis=1)
        assert np.all(norm_forces < 25000)

    if integrator.seed == 0:
        # this deepcopy is needed if we're running if client == None
        integrator = copy.deepcopy(integrator)
        integrator.seed = np.random.randint(np.iinfo(np.int32).max)

    if barostat.seed == 0:
        barostat = copy.deepcopy(barostat)
        barostat.seed = np.random.randint(np.iinfo(np.int32).max)

    intg_impl = integrator.impl()
     # technically we need to only pass in the nonbonded impl
    barostat_impl = barostat.impl(all_impls)
    # context components: positions, velocities, box, integrator, energy fxns
    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg_impl,
        all_impls,
        barostat_impl
    )

    # equilibration
    equil_schedule = np.ones(equil_steps)*lamb
    ctxt.multiple_steps(equil_schedule)

    for obs in du_dp_obs:
        ctxt.add_observable(obs)

    prod_schedule = np.ones(prod_steps)*lamb

    full_du_dls, xs, boxes, = ctxt.multiple_steps(prod_schedule, du_dl_interval, x_interval)

    # keep the structure of grads the same as that of final_potentials so we can properly
    # form their vjps.
    grads = []
    for obs in du_dp_obs:
        grads.append(obs.avg_du_dp())

    result = SimulationResult(xs=xs, boxes=boxes, du_dls=full_du_dls, du_dps=grads)
    return result
