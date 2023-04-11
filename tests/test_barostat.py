import numpy as np
import pytest

from timemachine.constants import AVOGADRO, BAR_TO_KJ_PER_NM3, BOLTZ
from timemachine.fe.free_energy import AbsoluteFreeEnergy
from timemachine.fe.topology import BaseTopology
from timemachine.ff import Forcefield
from timemachine.lib import LangevinIntegrator, custom_ops
from timemachine.md.barostat.moves import CentroidRescaler
from timemachine.md.barostat.utils import compute_box_center, compute_box_volume, get_bond_list, get_group_indices
from timemachine.md.builders import build_water_system
from timemachine.md.enhanced import get_solvent_phase_system
from timemachine.md.thermostat.utils import sample_velocities
from timemachine.testsystems.relative import get_hif2a_ligand_pair_single_topology


def test_barostat_zero_interval():
    pressure = 1.013  # bar
    temperature = 300.0  # kelvin
    seed = 2021
    np.random.seed(seed)

    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    unbound_potentials, sys_params, _, coords, _ = get_solvent_phase_system(mol_a, ff, lamb=0.0)

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list)

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    u_impls = []
    for bp in bound_potentials:
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    with pytest.raises(RuntimeError):
        custom_ops.MonteCarloBarostat(
            coords.shape[0],
            pressure,
            temperature,
            group_indices,
            0,
            u_impls,
            seed,
        )
    # Setting it to 1 should be valid.
    baro = custom_ops.MonteCarloBarostat(
        coords.shape[0],
        pressure,
        temperature,
        group_indices,
        1,
        u_impls,
        seed,
    )
    # Setting back to 0 should raise another error
    with pytest.raises(RuntimeError):
        baro.set_interval(0)


def test_barostat_partial_group_idxs():
    """Verify that the barostat can handle a subset of the molecules
    rather than all of them. This test only verify that it runs, not the behavior"""
    lam = 1.0
    temperature = 300.0  # kelvin
    timestep = 1.5e-3  # picosecond
    barostat_interval = 3  # step count
    collision_rate = 1.0  # 1 / picosecond

    seed = 2021
    np.random.seed(seed)

    pressure = 1.013  # bar
    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    unbound_potentials, sys_params, masses, coords, complex_box = get_solvent_phase_system(mol_a, ff, lam)

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list)

    # Cut the number of groups in half
    group_indices = group_indices[len(group_indices) // 2 :]

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    u_impls = []
    for bp in bound_potentials:
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        masses,
        seed,
    )
    integrator_impl = integrator.impl()

    v_0 = sample_velocities(masses, temperature)

    baro = custom_ops.MonteCarloBarostat(
        coords.shape[0],
        pressure,
        temperature,
        group_indices,
        barostat_interval,
        u_impls,
        seed,
    )

    ctxt = custom_ops.Context(coords, v_0, complex_box, integrator_impl, u_impls, barostat=baro)
    ctxt.multiple_steps(1000)


@pytest.mark.memcheck
def test_barostat_is_deterministic():
    """Verify that the barostat results in the same box size shift after 1000
    steps. This is important to debugging as well as providing the ability to replicate
    simulations
    """
    lam = 1.0
    temperature = 300.0
    timestep = 1.5e-3
    barostat_interval = 3
    collision_rate = 1.0
    seed = 2021
    np.random.seed(seed)

    box_vol = 26.89966

    pressure = 1.013

    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    host_system, host_coords, host_box, host_top = build_water_system(3.0, ff.water_ff)
    bt = BaseTopology(mol_a, ff)
    afe = AbsoluteFreeEnergy(mol_a, bt)

    unbound_potentials, sys_params, masses = afe.prepare_host_edge(ff.get_params(), host_system, lam)
    coords = afe.prepare_combined_coords(host_coords=host_coords)

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list)

    u_impls = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        masses,
        seed,
    )
    integrator_impl = integrator.impl()

    v_0 = sample_velocities(masses, temperature)

    baro = custom_ops.MonteCarloBarostat(
        coords.shape[0],
        pressure,
        temperature,
        group_indices,
        barostat_interval,
        u_impls,
        seed,
    )

    ctxt = custom_ops.Context(coords, v_0, host_box, integrator_impl, u_impls, barostat=baro)
    ctxt.multiple_steps(15)
    atm_box = ctxt.get_box()
    # Verify that the volume of the box has changed
    assert compute_box_volume(atm_box) != compute_box_volume(host_box)
    np.testing.assert_almost_equal(compute_box_volume(atm_box), box_vol, decimal=5)


def test_barostat_varying_pressure():
    lam = 1.0
    temperature = 300.0
    timestep = 1.5e-3
    barostat_interval = 3
    collision_rate = 1.0
    seed = 2021
    np.random.seed(seed)

    # Start out with a very large pressure
    pressure = 1013.0
    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    unbound_potentials, sys_params, masses, coords, complex_box = get_solvent_phase_system(mol_a, ff, lam, margin=0.0)

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list)

    u_impls = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        masses,
        seed,
    )
    integrator_impl = integrator.impl()

    v_0 = sample_velocities(masses, temperature)

    baro = custom_ops.MonteCarloBarostat(
        coords.shape[0],
        pressure,
        temperature,
        group_indices,
        barostat_interval,
        u_impls,
        seed,
    )

    ctxt = custom_ops.Context(coords, v_0, complex_box, integrator_impl, u_impls, barostat=baro)
    ctxt.multiple_steps(1000)
    ten_atm_box = ctxt.get_box()
    ten_atm_box_vol = compute_box_volume(ten_atm_box)
    # Expect the box to shrink thanks to the barostat
    assert compute_box_volume(complex_box) - ten_atm_box_vol > 0.4

    # Set the pressure to 1 atm
    baro.set_pressure(1.013)
    # Changing the barostat interval resets the barostat step.
    baro.set_interval(2)

    ctxt.multiple_steps(2000)
    atm_box = ctxt.get_box()
    # Box will grow thanks to the lower pressure
    assert compute_box_volume(atm_box) > ten_atm_box_vol


def test_molecular_ideal_gas():
    """


    References
    ----------
    OpenMM testIdealGas
    https://github.com/openmm/openmm/blob/d8ef57fed6554ec95684e53768188e1f666405c9/tests/TestMonteCarloBarostat.h#L86-L140
    """

    # simulation parameters
    timestep = 1.5e-3
    collision_rate = 1.0
    n_moves = 10000
    barostat_interval = 5
    seed = 2021

    # thermodynamic parameters
    temperatures = np.array([300, 600, 1000])
    pressure = 100.0  # very high pressure, to keep the expected volume small

    # generate an alchemical system of a waterbox + alchemical ligand:
    # effectively discard ligands by running in AbsoluteFreeEnergy mode at lambda = 1.0
    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    _unbound_potentials, _sys_params, masses, coords, complex_box = get_solvent_phase_system(
        mol_a, ff, lamb=1.0, margin=0.0
    )

    # drop the nonbonded potential
    unbound_potentials = _unbound_potentials[:-1]
    sys_params = _sys_params[:-1]

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = unbound_potentials[0]
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list)

    volume_trajs = []

    relative_tolerance = 1e-2
    initial_relative_box_perturbation = 2 * relative_tolerance

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    u_impls = []
    for bp in bound_potentials:
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    # expected volume
    n_water_mols = len(group_indices) - 1  # 1 for the ligand
    expected_volume_in_md = (n_water_mols + 1) * BOLTZ * temperatures / (pressure * AVOGADRO * BAR_TO_KJ_PER_NM3)

    for i, temperature in enumerate(temperatures):

        # define a thermostat
        integrator = LangevinIntegrator(
            temperature,
            timestep,
            collision_rate,
            masses,
            seed,
        )
        integrator_impl = integrator.impl()

        v_0 = sample_velocities(masses, temperature)

        # rescale the box to be approximately the desired box volume already
        rescaler = CentroidRescaler(group_indices)
        initial_volume = compute_box_volume(complex_box)
        initial_center = compute_box_center(complex_box)
        length_scale = ((1 + initial_relative_box_perturbation) * expected_volume_in_md[i] / initial_volume) ** (
            1.0 / 3
        )
        new_coords = rescaler.scale_centroids(coords, initial_center, length_scale)
        new_box = complex_box * length_scale

        baro = custom_ops.MonteCarloBarostat(
            new_coords.shape[0],
            pressure,
            temperature,
            group_indices,
            barostat_interval,
            u_impls,
            seed,
        )

        ctxt = custom_ops.Context(new_coords, v_0, new_box, integrator_impl, u_impls, barostat=baro)
        vols = []
        for move in range(n_moves // barostat_interval):
            ctxt.multiple_steps(barostat_interval)
            new_box = ctxt.get_box()
            volume = np.linalg.det(new_box)
            vols.append(volume)
        volume_trajs.append(vols)

    equil_time = len(volume_trajs[0]) // 2  # TODO: don't hard-code this?
    actual_volume_in_md = np.array([np.mean(volume_traj[equil_time:]) for volume_traj in volume_trajs])

    np.testing.assert_allclose(actual=actual_volume_in_md, desired=expected_volume_in_md, rtol=relative_tolerance)
