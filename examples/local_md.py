"""
This example demonstrates the possibility of running Molecular Dynamics over a small 'local' section of a system.

The example compound is a benzene in a water box
"""
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem
from simtk import unit
from simtk.openmm import app

from timemachine import constants
from timemachine.fe import pdb_writer, topology
from timemachine.fe.free_energy import AbsoluteFreeEnergy
from timemachine.fe.model_utils import apply_hmr
from timemachine.ff import Forcefield
from timemachine.integrator import LangevinIntegrator as ReferenceLangevinIntegrator
from timemachine.lib import LangevinIntegrator, custom_ops
from timemachine.lib.potentials import NonbondedInteractionGroup
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import get_bond_list
from timemachine.md.states import CoordsVelBox
from timemachine.md.thermostat.utils import sample_velocities


@dataclass
class MDState:
    seed: int
    mol: Any
    temperature: float
    dt: float
    masses: NDArray
    unbound_potentials: Any
    sys_params: NDArray
    ff: Any
    friction: float = 1.0
    _bound_pots: Optional[Any] = None

    def get_integrator(self):
        return LangevinIntegrator(self.temperature, self.dt, self.friction, self.masses, self.seed).impl()

    def get_bound_potentials(self, idxs: Optional[NDArray] = None):
        if self._bound_pots is None:
            bound_potentials = []
            for i, (params, unbound_pot) in enumerate(zip(self.sys_params, self.unbound_potentials)):
                params = np.asarray(params)
                bp = unbound_pot.bind(params)
                bound_potentials.append(bp.bound_impl(precision=np.float32))
            self._bound_pots = bound_potentials
        return self._bound_pots


def run_global_steps(state: MDState, sys_state: CoordsVelBox, steps: int) -> List[CoordsVelBox]:
    """Using the Timemachine Custom Ops, perform Global MD for a certain number of steps"""
    integrator = state.get_integrator()
    bound_impls = state.get_bound_potentials()
    ctxt = custom_ops.Context(sys_state.coords, sys_state.velocities, sys_state.box, integrator, bound_impls)

    _, xs, boxes = ctxt.multiple_steps(np.zeros(steps), 0, 1)
    states = [CoordsVelBox(coords, None, box) for coords, box in zip(xs, boxes)]
    # Multiple steps doesn't return the last xs
    states.append(CoordsVelBox(ctxt.get_x_t(), ctxt.get_v_t(), ctxt.get_box()))
    return states


def reduce_nblist_ixns(ixns) -> NDArray:
    flattened = np.concatenate(ixns).ravel()
    return np.unique(flattened)


def get_frozen_idxs(coords, mol, box, cutoff=1.2):
    num_lig_atoms = mol.GetNumAtoms()
    num_host_atoms = len(coords) - num_lig_atoms

    # Construct list of atoms in the inner shell
    inner_nblist = custom_ops.Neighborlist_f64(len(coords))
    inner_nblist.set_row_idxs(np.arange(len(coords) - num_lig_atoms, len(coords), dtype=np.uint32))
    first_shell_ixns = inner_nblist.get_nblist(coords, box, cutoff)
    inner_shell_idxs = reduce_nblist_ixns(first_shell_ixns)

    # Construct list of atoms in the outer shell, currently not necessary
    # outer_nblist = custom_ops.Neighborlist_f32(num_host_atoms - len(inner_shell_idxs), len(inner_shell_idxs))
    # second_shell_ixns = outer_nblist.get_nblist_host_ligand(np.delete(coords[:num_host_atoms], inner_shell_idxs, axis=0), coords[inner_shell_idxs], sys_state.box, outer_shell_size)
    # outer_shell_indices = reduce_nblist_ixns(second_shell_ixns)

    # Combine all of the indices that are involved in the inner shell
    subsystem_idxs = np.unique(np.concatenate([inner_shell_idxs, np.arange(num_host_atoms, len(coords))]))
    # Construct the idxs of atoms that shouldn't be moved
    frozen_idxs = np.delete(np.arange(len(coords)), subsystem_idxs)
    assert len(frozen_idxs) + len(subsystem_idxs) == len(coords), "Doesn't Match!"
    return frozen_idxs


def write_frames(pdb_path: str, output_path: str, frames: List[CoordsVelBox]):
    xs = []
    boxes = []
    for frame in frames:
        xs.append(frame.coords)
        boxes.append(frame.box)
    xs = np.array(xs)
    boxes = np.array(boxes)
    if os.path.splitext(output_path)[1] != ".npz":
        import mdtraj  # noqa

        pdb = app.PDBFile(pdb_path)

        traj = mdtraj.Trajectory(xs, mdtraj.Topology.from_openmm(pdb.topology))
        traj.unitcell_vectors = boxes
        traj.image_molecules()
        traj.save(output_path)
    else:
        np.savez(output_path, xs=xs, boxes=boxes)


def run_local_steps(state: MDState, sys_state: CoordsVelBox, steps: int, reference: bool = False) -> List[CoordsVelBox]:
    """Perform Local MD for a certain number of steps.

    Currently just resets the coords/velocities for coordinates not within the inner shell.
    """

    box = sys_state.box

    # Compute forces on the complete system, revert changes to coords/velocities
    bound_impls = state.get_bound_potentials()

    # shift_tolerance = 0.25 * (1.2 ** 2)
    coords = sys_state.coords
    velocities = sys_state.velocities.copy()
    # velocities = sample_velocities(state.masses * unit.amu, state.temperature * unit.kelvin),
    states = []
    if reference:
        frozen_idxs = get_frozen_idxs(sys_state.coords, state.mol, box)

        def force_func(coords: NDArray):
            du_dxs = np.array([bp.execute(coords, box, 0.0)[0] for bp in bound_impls])
            forces = -np.sum(du_dxs, 0)
            return forces

        langevin = ReferenceLangevinIntegrator(force_func, state.masses, state.temperature, state.dt, state.friction)
        x = coords.copy()
        v = velocities
        for i in range(steps):
            x, v = langevin.step(x, v)
            x[frozen_idxs] = coords[frozen_idxs]
            v[frozen_idxs] = sys_state.velocities[frozen_idxs]
            updated_state = CoordsVelBox(x, v, box)
            states.append(updated_state)
    else:
        integrator = state.get_integrator()
        ctxt = custom_ops.Context(coords, sys_state.velocities.copy(), box, integrator, bound_impls)
        frames, boxes = ctxt.local_md(
            np.zeros(1),
            steps,
            0,
            1,  # Run n iterations of a single local step
            1,
            # Use ligand as the coordinates around the molecule
            np.arange(len(coords) - state.mol.GetNumAtoms(), len(coords), dtype=np.uint32),
        )
        states.extend([CoordsVelBox(x, None, box) for x, box in zip(frames, boxes)])
        states.append(CoordsVelBox(ctxt.get_x_t(), ctxt.get_v_t(), ctxt.get_box()))
        # np.testing.assert_array_equal(v[frozen_idxs], velocities[frozen_idxs])
    for i in range(len(states) - 1):
        # Clear the velocities for older states, else memory can become an issue
        states[i].velocities = None
    return states


def write_energy_distribution(
    iterations: int, global_steps: int, local_steps: int, potentials: Any, frames: List[CoordsVelBox]
):
    def energy_func(coords: NDArray, box: NDArray):
        energies = np.array([bp.execute(coords, box, 0.0)[2] for bp in potentials])
        return np.sum(energies)

    energies = []
    for frame in frames:
        energies.append(energy_func(frame.coords, frame.box))
    plt.hist(energies, bins=100, density=True)
    plt.xlabel("Energies")
    plt.ylabel("Density")
    plt.savefig(f"iterations_{iterations}_global_{global_steps}_local_{local_steps}_energies.png", dpi=150)


def main():
    parser = ArgumentParser(description="Local MD Prototype")
    parser.add_argument("iterations", default=10, type=int, help="Number of iterations of global MD + local MD")
    parser.add_argument("--global_steps", default=1, type=int, help="Number of global MD steps to take per iteration")
    parser.add_argument("--local_steps", default=500, type=int, help="Number of local MD steps to take per iteration")
    parser.add_argument("--host_output", default="host_system.pdb", help="Path to write out pdb file of the system")
    parser.add_argument(
        "--traj_output",
        default="host_system.npz",
        help="Path to write out trajectory, default format is npz but supports mdtraj if it is installed",
    )
    parser.add_argument(
        "--forcefield",
        default="smirnoff_1_1_0_ccc.py",
        choices=["smirnoff_1_1_0_sc.py", "smirnoff_1_1_0_ccc.py"],
        type=str,
        help="Forcefield to use for ligand",
    )
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument(
        "--preequil_steps", default=50000, type=int, help="Number of steps to pre-equilibrate water box"
    )
    args = parser.parse_args()

    temperature = 300.0
    pressure = 1.0
    friction = 1.0
    cutoff = 1.2
    dt = 2.5e-3

    ff = Forcefield.load_from_file(args.forcefield)
    # Generate a Benzene mol
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=args.seed)

    system, coords, box, solv_topo = builders.build_water_system(4.0)

    coords, box = minimizer.equilibrate_host(
        mol,
        system,
        coords,
        temperature,
        pressure,
        ff,
        box,
        args.preequil_steps,
        seed=args.seed,
    )
    guest_topology = topology.BaseTopology(mol, ff)
    afe = AbsoluteFreeEnergy(mol, guest_topology)

    # Hackery where you drop the coords as the ligand coords already combined in equilibrate_host
    unbound_potentials, sys_params, masses = afe.prepare_host_edge(ff.get_ordered_params(), system)

    writer = pdb_writer.PDBWriter([solv_topo, mol], args.host_output)
    writer.write_frame(coords * 10)
    writer.close()

    if dt > 1.5e-3:
        bond_list = get_bond_list(unbound_potentials[0])
        masses = apply_hmr(masses, bond_list)
    ctx = MDState(args.seed, mol, temperature, dt, masses, unbound_potentials, sys_params, ff, friction)

    state = CoordsVelBox(
        coords,
        # Requires units
        sample_velocities(masses * unit.amu, temperature * unit.kelvin),
        box,
    )
    iteration_frames = [state]

    for i in range(args.iterations):
        if args.global_steps <= 0 and args.local_steps <= 0:
            raise RuntimeError("No MD steps to be run")
        if args.global_steps > 0:
            states = run_global_steps(ctx, iteration_frames[-1], args.global_steps)
            iteration_frames.extend(states)
        if args.local_steps > 0:
            states = run_local_steps(ctx, iteration_frames[-1], args.local_steps)
            iteration_frames.extend(states)

    beta = 1 / (constants.BOLTZ * temperature)
    potential = (
        NonbondedInteractionGroup(
            np.arange(len(coords) - mol.GetNumAtoms(), len(coords), dtype=np.int32),
            np.zeros(len(coords), dtype=np.int32),
            np.zeros(len(coords), dtype=np.int32),
            beta,
            cutoff,
        )
        .bind(sys_params[-1])
        .bound_impl(np.float32)
    )
    write_energy_distribution(args.iterations, args.global_steps, args.local_steps, [potential], iteration_frames)
    write_frames(args.host_output, args.traj_output, iteration_frames)


if __name__ == "__main__":
    main()
