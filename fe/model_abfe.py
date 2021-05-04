import functools
import numpy as np
import jax.numpy as jnp
import tempfile
import mdtraj

from simtk import openmm
from simtk.openmm import app
from rdkit import Chem

from timemachine import constants

from md import minimizer
from timemachine.lib import LangevinIntegrator
from timemachine.lib import potentials
from fe import free_energy, topology, estimator_abfe
from ff import Forcefield

from parallel.client import AbstractClient
from typing import Optional
from functools import partial

from ff.handlers import openmm_deserializer
from scipy.optimize import linear_sum_assignment
import scipy.spatial
from simtk import unit

import matplotlib.pyplot as plt

def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm

def generate_topology(objs, host_coords, out_filename):
    rd_mols = []
    # super jank
    for obj in objs:
        if isinstance(obj, app.Topology):
            with tempfile.NamedTemporaryFile(mode='w') as fp:
                # write
                app.PDBFile.writeHeader(obj, fp)
                app.PDBFile.writeModel(obj, host_coords, fp, 0)
                app.PDBFile.writeFooter(obj, fp)
                fp.flush()
                romol = Chem.MolFromPDBFile(fp.name, removeHs=False)
                rd_mols.append(romol)

        if isinstance(obj, Chem.Mol):
            rd_mols.append(obj)

    combined_mol = rd_mols[0]
    for mol in rd_mols[1:]:
        print(mol)
        combined_mol = Chem.CombineMols(combined_mol, mol)

    # with tempfile.NamedTemporaryFile(mode='w') as fp:
    fp = open(out_filename, "w")
    # write
    Chem.MolToPDBFile(combined_mol, out_filename)
    fp.flush()
    # read
    combined_pdb = app.PDBFile(out_filename)
    return combined_pdb.topology


def setup_restraints(
    mol,
    core,
    host_topology,
    host_coords,
    k_core):
    """
    Setup rigid restraint between protein c-alpha and the core atoms in the ligand.
    It is assumed that the ligand is properly positioned within the protein.

    Parameters
    ----------
    mol: Chem.Mol
        molecule with conformer information specified

    core: np.array of shape [C]
        core atoms in the molecule

    host_topology: openmm.Topology
        Topology of the host

    host_coords: Nx3
        Coordinates of the host.

    k_core: float
        force constant of the harmonic bond.

    Returns
    -------
    tuple
        core_idxs (Nx2), core_params (Nx2)

    """
    ligand_coords = get_romol_conf(mol)
    core_coords = ligand_coords[core]

    # restrict core to the convex hull
    hull_idxs = scipy.spatial.ConvexHull(core_coords).vertices
    core = [core[x] for x in hull_idxs]

    ri = np.expand_dims(ligand_coords, 1)
    rj = np.expand_dims(host_coords, 0)

    # (ytz): should we use PBCs here?
    # d2ij = np.sum(np.power(delta_r(ri, rj, host_box), 2), axis=-1)
    dij = np.sqrt(np.sum(np.power(ri-rj, 2), axis=-1))

    atom_names = [a.name for a in host_topology.atoms()]

    def is_alpha_carbon(idx):
        return atom_names[idx] == 'CA'

    # what if one of these c-alphas is a highly motile loop?
    pocket_atoms = set()

    # 5 angstrom radius
    pocket_cutoff = 1.0

    for i_idx in range(mol.GetNumAtoms()):
        if i_idx in core:
            dists = dij[i_idx]
            for j_idx, dist in enumerate(dists):
                if is_alpha_carbon(j_idx):
                    if dist < pocket_cutoff:
                        pocket_atoms.add(j_idx)

    pocket_atoms = np.array(list(pocket_atoms))

    # print("c-alphas to be used as restraints:", pocket_atoms)

    ri = np.expand_dims(ligand_coords[core], 1)
    rj = np.expand_dims(host_coords[pocket_atoms], 0)
    dij_pocket = np.sqrt(np.sum(np.power(ri-rj, 2), axis=-1))

    row_idxs, col_idxs = linear_sum_assignment(dij_pocket)
    num_host_atoms = host_coords.shape[0]

    core_idxs = []
    core_params = []
    for core_i, protein_j in zip(row_idxs, col_idxs):
        core_idxs.append((core[core_i] + num_host_atoms, pocket_atoms[protein_j]))
        core_params.append((k_core, 0.0))

    print(core_idxs)
        # (ytz): intentionally left commented out in-case we want to try a non-0 value
        # later on. But this will cause poor overlap with the RMSD restraint when we
        # do the endpoint correction.
        # core_params.append((self.k_core, dij_pocket[core_i, protein_j]))

    core_idxs = np.array(core_idxs, dtype=np.int32)
    core_params = np.array(core_params, dtype=np.float64)

    return core_idxs, core_params


class AbsoluteModel():

    def __init__(
        self,
        client: AbstractClient or None,
        ff: Forcefield,
        host_system: openmm.System,
        host_coords: np.ndarray,
        host_box: np.ndarray,
        host_schedule: np.ndarray,
        host_topology: np.ndarray,
        equil_steps: int,
        prod_steps: int):

        self.host_system = host_system
        self.host_coords = host_coords
        self.host_box = host_box
        self.host_schedule = host_schedule
        self.host_topology = host_topology

        self.client = client
        self.ff = ff
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps

    def predict(self,
        ff_params,
        mol,
        restraints,
        prefix,
        cache_results=None):

        print(f"Minimizing the host structure to remove clashes.")
        # min_host_coords = self.host_coords
        min_host_coords = minimizer.minimize_host_4d([mol], self.host_system, self.host_coords, self.ff, self.host_box)


        # dummy_atoms = np.array([29, 30, 31, 32, 33, 34]) - 1
        dummy_atoms = None

        afe = free_energy.AbsoluteFreeEnergy(mol, self.ff, dummy=dummy_atoms)


        unbound_potentials, sys_params, masses, coords = afe.prepare_host_edge(
            ff_params,
            self.host_system,
            min_host_coords
        )


        atom_names = [a.name for a in self.host_topology.atoms()]

        def is_alpha_carbon(idx):
            return atom_names[idx] == 'CA'

        print("freezing C-alphas...")
        num_host_atoms = len(min_host_coords)
        # freeze c-alphas
        for idx in range(num_host_atoms):
            if atom_names[idx] == 'CA':
                # print(idx)
                masses[idx] = 999999.9

        # setup lambda transformations
        transform_qlj = "lambda < 0.5 ? sin(lambda*PI)*sin(lambda*PI) : 1"
        transform_w = "lambda < 0.5 ? 0.0 : sin((lambda+0.5)*PI)*sin((lambda+0.5)*PI)"
        # cache_lambda = 0.5 # if lambda <= cache_lambda then we re-run the simulation.
        # transform_qlj = "lambda"
        # transform_w = "lambda"
        cache_lambda = 1.0

        nonbonded = unbound_potentials[-1]
        nonbonded.args.extend([
            transform_qlj,
            transform_qlj,
            transform_qlj,
            transform_w]
        )

        core = []
        for a in mol.GetAtoms():
            if a.IsInRing():
                core.append(a.GetIdx())

        if restraints:
            k_core = 50.0

            core_idxs, core_params = setup_restraints(mol, core, self.host_topology, self.host_coords, k_core)
            combined_topology = generate_topology([self.host_topology, mol], min_host_coords*10, "complex.pdb")
            B = len(core_idxs)

            # leave core-restraints turned on for now
            # core_lambda_mult = np.ones(B)
            # core_lambda_offset = np.zeros(B)
            core_lambda_mult = np.zeros(B)
            core_lambda_offset = np.ones(B)

            restraint_potential = potentials.HarmonicBond(
                core_idxs,
                core_lambda_mult.astype(np.int32),
                core_lambda_offset.astype(np.int32)
            )

            unbound_potentials.append(restraint_potential)
            sys_params.append(core_params)

            endpoint_correct = True

        else:
            endpoint_correct = False

        seed = 0

        temperature = 300.0
        beta = 1/(constants.BOLTZ*temperature)

        integrator = LangevinIntegrator(
            temperature,
            1.5e-3,
            1.0,
            masses,
            seed
        )

        x0 = coords
        v0 = np.zeros_like(coords)

        # minimize pressure
        print("Start pressure minimization")
        xs_and_box, volume_traj = minimizer.minimize_pressure(
            x0,
            self.host_box,
            masses,
            unbound_potentials,
            sys_params,
            self.host_schedule[0],
            pressure=1.013*unit.bar,
            n_moves=4000
        )

        # plt.plot(volume_traj)
        # plt.show()

        xs = []
        vs = []
        boxes = []

        for t in xs_and_box:
            xs.append(t.coords)
            vs.append(t.velocities)
            boxes.append(t.box)

        xs = np.array(xs)
        vs = np.array(vs)
        boxes = np.array(boxes)

        print("Equilibration")
        # traj = mdtraj.Trajectory(xs, mdtraj.Topology.from_openmm(combined_topology))
        # unit_cell = boxes
        # traj.unitcell_vectors = unit_cell
        # traj.image_molecules()
        # traj.save_xtc("complex_lambda_npt_"+str(idx)+".xtc")

        # assert 0
        x0 = xs[-1]
        v0 = vs[-1]
        box0 = boxes[-1]

        if cache_results is None:
            cache_results = [None]*len(self.host_schedule)
            if endpoint_correct:
                cache_results.append(None)

        model = estimator_abfe.FreeEnergyModel(
            unbound_potentials,
            endpoint_correct,
            self.client,
            box0,
            x0,
            v0,
            integrator,
            self.host_schedule,
            self.equil_steps,
            self.prod_steps,
            beta,
            prefix,
            cache_results,
            cache_lambda
        )

        dG, results = estimator_abfe.deltaG(model, sys_params)

        for idx, result in enumerate(results):
            # print(result.xs.shape)
            traj = mdtraj.Trajectory(result.xs, mdtraj.Topology.from_openmm(combined_topology))
            unit_cell = np.repeat(self.host_box[None, :], len(result.xs), axis=0)
            traj.unitcell_vectors = unit_cell
            traj.image_molecules()
            traj.save_xtc("complex_lambda_"+str(idx)+".xtc")

        np.savez("results.npz", results=results)

        assert 0

        return dG, results
