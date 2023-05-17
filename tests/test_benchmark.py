"""Run vanilla "apo" MD on DHFR and HIF2A test systems,
and running an intermediate lambda window "rbfe" MD for a
relative binding free energy edge from the HIF2A test system"""

import time
from importlib import resources

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from timemachine import constants
from timemachine.fe import rbfe
from timemachine.fe.model_utils import apply_hmr
from timemachine.fe.single_topology import SingleTopology
from timemachine.ff import Forcefield
from timemachine.ff.handlers import openmm_deserializer
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.potentials import BoundPotential, Nonbonded, NonbondedInteractionGroup, Potential
from timemachine.testsystems.dhfr import setup_dhfr
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology

SECONDS_PER_DAY = 24 * 60 * 60


@pytest.fixture(scope="module")
def hi2fa_test_frames():
    return generate_hif2a_frames(100, 1, seed=2022)


def generate_hif2a_frames(n_frames: int, frame_interval: int, seed=None, barostat_interval: int = 5, hmr: bool = True):

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, forcefield)

    # build the protein system.
    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        host_system, host_coords, host_box, _ = builders.build_protein_system(
            str(path_to_pdb), forcefield.protein_ff, forcefield.water_ff
        )

    initial_state = prepare_hif2a_initial_state(st, host_system, host_coords, host_box)

    ligand_idxs = np.arange(len(host_coords), len(initial_state.x0), dtype=np.int32)

    temperature = constants.DEFAULT_TEMP
    pressure = constants.DEFAULT_PRESSURE

    harmonic_bond_potential = initial_state.potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential.potential)
    masses = initial_state.integrator.masses
    if hmr:
        dt = 2.5e-3
        masses = apply_hmr(masses, bond_list)
    if seed is None:
        seed = np.random.randint(np.iinfo(np.int32).max)
    intg = LangevinIntegrator(temperature, dt, 1.0, np.array(masses), seed).impl()

    bps = []

    for potential in initial_state.potentials:
        bps.append(potential.to_gpu(precision=np.float32).bound_impl)  # get the bound implementation

    baro_impl = None

    if barostat_interval > 0:
        group_idxs = get_group_indices(bond_list, len(masses))
        baro = MonteCarloBarostat(
            initial_state.x0.shape[0],
            pressure,
            temperature,
            group_idxs,
            barostat_interval,
            seed,
        )
        baro_impl = baro.impl(bps)

    ctxt = custom_ops.Context(
        initial_state.x0,
        initial_state.v0,
        host_box,
        intg,
        bps,
        barostat=baro_impl,
    )
    steps = n_frames * frame_interval
    coords, boxes = ctxt.multiple_steps(steps, frame_interval)
    assert coords.shape[0] == n_frames, f"Got {coords.shape[0]} frames, expected {n_frames}"
    return initial_state.potentials, coords, boxes, ligand_idxs


def benchmark_potential(
    label,
    potential: Potential,
    precision,
    params,
    coords,
    boxes,
    lambdas,
    verbose=True,
    num_batches=5,
    compute_du_dx=True,
    compute_du_dp=True,
    compute_u=True,
):
    if precision == np.float32:
        label = label + "_f32"
    else:
        label = label + "_f64"
    unbound = potential.to_gpu(precision=precision).unbound_impl
    start = time.time()
    batch_times = []
    frames = coords.shape[0]
    param_batches = params.shape[0]
    num_lambs = len(lambdas)
    runs_per_batch = frames * param_batches * num_lambs
    for _ in range(num_batches):
        batch_start = time.time()
        _, _, _ = unbound.execute_selective_batch(
            coords,
            params,
            boxes,
            compute_du_dx,
            compute_du_dp,
            compute_u,
        )
        batch_end = time.time()
        delta = batch_end - batch_start

        batch_times.append(delta)
        runs_per_second = runs_per_batch / np.mean(batch_times)

        if verbose:
            print(f"executions per second: {runs_per_second:.3f}")
    print(
        f"{label}: N={coords.shape[1]} Frames={frames} Params={param_batches} Lambdas={num_lambs} speed: {runs_per_second:.2f} executions/seconds (ran {runs_per_batch * num_batches} potentials in {(time.time() - start):.2f}s)"
    )


def benchmark(
    label,
    masses,
    x0,
    v0,
    box,
    bound_potentials,
    hmr=False,
    verbose=True,
    num_batches=100,
    steps_per_batch=1000,
    barostat_interval=0,
):
    """
    TODO: configuration blob containing num_batches, steps_per_batch, and any other options
    """

    seed = 1234
    dt = 1.5e-3
    temperature = constants.DEFAULT_TEMP
    pressure = constants.DEFAULT_PRESSURE

    harmonic_bond_potential = bound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential.potential)
    if hmr:
        dt = 2.5e-3
        masses = apply_hmr(masses, bond_list)
    intg = LangevinIntegrator(temperature, dt, 1.0, np.array(masses), seed).impl()

    bps = []

    for potential in bound_potentials:
        bps.append(potential.to_gpu(precision=np.float32).bound_impl)  # get the bound implementation

    baro_impl = None
    if barostat_interval > 0:
        group_idxs = get_group_indices(bond_list, len(masses))
        baro = MonteCarloBarostat(
            x0.shape[0],
            pressure,
            temperature,
            group_idxs,
            barostat_interval,
            seed,
        )
        baro_impl = baro.impl(bps)

    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg,
        bps,
        barostat=baro_impl,
    )

    batch_times = []

    # run once before timer starts
    ctxt.multiple_steps(steps_per_batch)

    start = time.time()

    for batch in range(num_batches):

        # time the current batch
        batch_start = time.time()
        _, _ = ctxt.multiple_steps(steps_per_batch)
        batch_end = time.time()

        delta = batch_end - batch_start

        batch_times.append(delta)

        steps_per_second = steps_per_batch / np.mean(batch_times)
        steps_per_day = steps_per_second * SECONDS_PER_DAY

        ps_per_day = dt * steps_per_day
        ns_per_day = ps_per_day * 1e-3

        if verbose:
            print(f"steps per second: {steps_per_second:.3f}")
            print(f"ns per day: {ns_per_day:.3f}")

    assert np.all(np.abs(ctxt.get_x_t()) < 1000)

    print(
        f"{label}: N={x0.shape[0]} speed: {ns_per_day:.2f}ns/day dt: {dt*1e3}fs (ran {steps_per_batch * num_batches} steps in {(time.time() - start):.2f}s)"
    )


def benchmark_local(
    label,
    masses,
    x0,
    v0,
    box,
    bound_potentials,
    ligand_idxs,
    hmr=False,
    verbose=True,
    num_batches=100,
    steps_per_batch=1000,
):
    """
    TODO: configuration blob containing num_batches, steps_per_batch, and any other options
    """

    seed = 1234
    dt = 1.5e-3
    temperature = constants.DEFAULT_TEMP
    friction = 1.0

    rng = np.random.default_rng(seed)

    harmonic_bond_potential = bound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential.potential)
    if hmr:
        dt = 2.5e-3
        masses = apply_hmr(masses, bond_list)
    intg = LangevinIntegrator(temperature, dt, friction, np.array(masses), seed).impl()

    bps = []

    for potential in bound_potentials:
        bps.append(potential.to_gpu(precision=np.float32).bound_impl)  # get the bound implementation

    ctxt = custom_ops.Context(
        x0,
        v0,
        box,
        intg,
        bps,
    )

    batch_times = []

    ligand_idxs = ligand_idxs.astype(np.int32)

    local_seed = rng.integers(np.iinfo(np.int32).max)
    # run once before timer starts
    ctxt.multiple_steps_local(steps_per_batch, ligand_idxs, seed=local_seed, burn_in=0)

    start = time.time()

    for batch in range(num_batches):

        local_seed = rng.integers(np.iinfo(np.int32).max)
        # time the current batch
        batch_start = time.time()
        _, _ = ctxt.multiple_steps_local(steps_per_batch, ligand_idxs, seed=local_seed, burn_in=0)
        batch_end = time.time()

        delta = batch_end - batch_start

        batch_times.append(delta)

        steps_per_second = steps_per_batch / np.mean(batch_times)
        steps_per_day = steps_per_second * SECONDS_PER_DAY

        ps_per_day = dt * steps_per_day
        ns_per_day = ps_per_day * 1e-3

        if verbose:
            print(f"steps per second: {steps_per_second:.3f}")
            print(f"ns per day: {ns_per_day:.3f}")

    assert np.all(np.abs(ctxt.get_x_t()) < 1000)

    print(
        f"{label}: N={x0.shape[0]} speed: {ns_per_day:.2f}ns/day dt: {dt*1e3}fs (ran {steps_per_batch * num_batches} steps in {(time.time() - start):.2f}s)"
    )


def benchmark_dhfr(verbose=False, num_batches=100, steps_per_batch=1000):

    host_fns, host_masses, host_conf, box = setup_dhfr()

    x0 = host_conf
    v0 = np.zeros_like(host_conf)

    benchmark(
        "dhfr-apo",
        host_masses,
        x0,
        v0,
        box,
        host_fns,
        verbose=verbose,
        num_batches=num_batches,
        steps_per_batch=steps_per_batch,
    )
    benchmark(
        "dhfr-apo-barostat-interval-25",
        host_masses,
        x0,
        v0,
        box,
        host_fns,
        verbose=verbose,
        num_batches=num_batches,
        steps_per_batch=steps_per_batch,
        barostat_interval=25,
    )
    benchmark(
        "dhfr-apo-hmr-barostat-interval-25",
        host_masses,
        x0,
        v0,
        box,
        host_fns,
        verbose=verbose,
        hmr=True,
        num_batches=num_batches,
        steps_per_batch=steps_per_batch,
        barostat_interval=25,
    )


def prepare_hif2a_initial_state(st, host_system, host_coords, host_box):
    st = rbfe.SingleTopology(st.mol_a, st.mol_b, st.core, st.ff)
    host_config = rbfe.HostConfig(host_system, host_coords, host_box)
    temperature = constants.DEFAULT_TEMP
    lamb = 0.1
    host = rbfe.setup_optimized_host(st, host_config)
    initial_state = rbfe.setup_initial_states(st, host, temperature, [lamb], seed=2022)[0]
    bound_impls = [p.to_gpu(np.float32).bound_impl for p in initial_state.potentials]
    val_and_grad_fn = minimizer.get_val_and_grad_fn(bound_impls, initial_state.box0)
    assert np.all(np.isfinite(initial_state.x0)), "Initial coordinates contain nan or inf"
    ligand_coords = initial_state.x0[initial_state.ligand_idxs]
    d_ij = cdist(ligand_coords, initial_state.x0)
    # if any atom is within any of the ligand's atom's ixn radius, flag it for minimization
    cutoff = 0.5  # in nanometers
    free_idxs = np.where(np.any(d_ij < cutoff, axis=0))[0].tolist()
    x0_min = minimizer.local_minimize(initial_state.x0, val_and_grad_fn, free_idxs)
    initial_state.x0 = x0_min
    return initial_state


def benchmark_hif2a(verbose=False, num_batches=100, steps_per_batch=1000):

    # we use simple charge "sc" to be able to run on machines that don't have openeye licenses.
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, forcefield)

    # build the protein system.
    with resources.path("timemachine.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        complex_system, complex_coords, complex_box, _ = builders.build_protein_system(
            str(path_to_pdb), forcefield.protein_ff, forcefield.water_ff
        )

    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0, forcefield.water_ff)

    for stage, host_system, host_coords, host_box in [
        ("hif2a", complex_system, complex_coords, complex_box),
        ("solvent", solvent_system, solvent_coords, solvent_box),
    ]:

        host_fns, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)

        # resolve host clashes
        min_host_coords = minimizer.minimize_host_4d([st.mol_a, st.mol_b], host_system, host_coords, st.ff, host_box)

        x0 = min_host_coords
        v0 = np.zeros_like(x0)

        benchmark(
            stage + "-apo",
            host_masses,
            x0,
            v0,
            host_box,
            host_fns,
            verbose=verbose,
            num_batches=num_batches,
            steps_per_batch=steps_per_batch,
        )
        benchmark(
            stage + "-apo-barostat-interval-25",
            host_masses,
            x0,
            v0,
            host_box,
            host_fns,
            verbose=verbose,
            num_batches=num_batches,
            steps_per_batch=steps_per_batch,
            barostat_interval=25,
        )

        # RBFE
        initial_state = prepare_hif2a_initial_state(st, host_system, host_coords, host_box)

        benchmark(
            stage + "-rbfe",
            initial_state.integrator.masses,
            initial_state.x0,
            initial_state.v0,
            host_box,
            initial_state.potentials,
            verbose=verbose,
            num_batches=num_batches,
            steps_per_batch=steps_per_batch,
        )

        benchmark_local(
            stage + "-rbfe-local",
            initial_state.integrator.masses,
            initial_state.x0,
            initial_state.v0,
            host_box,
            initial_state.potentials,
            initial_state.ligand_idxs,
            verbose=verbose,
            num_batches=num_batches,
            steps_per_batch=steps_per_batch,
        )


def test_dhfr():
    benchmark_dhfr(verbose=True, num_batches=2, steps_per_batch=100)


def test_hif2a():
    benchmark_hif2a(verbose=True, num_batches=2, steps_per_batch=100)


def test_nonbonded_interaction_group_potential(hi2fa_test_frames):
    bps, frames, boxes, ligand_idxs = hi2fa_test_frames
    lambdas = np.array([0.0, 1.0])
    nonbonded_potential: BoundPotential[Nonbonded] = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))

    num_param_batches = 5
    beta = 1 / (constants.BOLTZ * 300)
    cutoff = 1.2

    precisions = [np.float32, np.float64]
    nonbonded_params = np.stack([nonbonded_potential.params] * num_param_batches)

    potential = NonbondedInteractionGroup(
        nonbonded_potential.potential.num_atoms,
        ligand_idxs,
        beta,
        cutoff,
    )
    class_name = potential.__class__.__name__
    for precision in precisions:
        benchmark_potential(
            class_name,
            potential,
            precision,
            nonbonded_params,
            frames,
            boxes,
            lambdas,
            verbose=False,
        )


def test_nonbonded_potential(hi2fa_test_frames):
    bps, frames, boxes, _ = hi2fa_test_frames

    nonbonded_pot: BoundPotential[Nonbonded]
    nonbonded_pot = next(bp for bp in bps if isinstance(bp.potential, Nonbonded))
    assert nonbonded_pot is not None
    lambdas = np.array([0.0, 1.0])

    num_param_batches = 5

    nonbonded_params = np.stack([nonbonded_pot.params] * num_param_batches)

    precisions = [np.float32, np.float64]

    potential = Nonbonded(
        nonbonded_pot.potential.num_atoms,
        nonbonded_pot.potential.exclusion_idxs,
        nonbonded_pot.potential.scale_factors,
        nonbonded_pot.potential.beta,
        nonbonded_pot.potential.cutoff,
    )

    class_name = potential.__class__.__name__
    for precision in precisions:
        benchmark_potential(
            class_name,
            potential,
            precision,
            nonbonded_params,
            frames,
            boxes,
            lambdas,
            verbose=False,
        )


def test_bonded_potentials(hi2fa_test_frames):
    bps, frames, boxes, _ = hi2fa_test_frames

    lambdas = np.array([0.0, 1.0])
    for bp in bps[:-1]:
        potential = bp.potential
        class_name = potential.__class__.__name__
        params = np.expand_dims(bp.params, axis=0)
        for precision in [np.float32, np.float64]:
            benchmark_potential(
                class_name,
                bp.potential,
                precision,
                params,
                frames,
                boxes,
                lambdas,
                verbose=False,
            )


if __name__ == "__main__":

    benchmark_dhfr(verbose=False, num_batches=100)
    benchmark_hif2a(verbose=False, num_batches=100)

    hif2a_frames = generate_hif2a_frames(1000, 5, seed=2022)
    test_nonbonded_interaction_group_potential(hif2a_frames)
    test_nonbonded_potential(hif2a_frames)
    test_bonded_potentials(hif2a_frames)
