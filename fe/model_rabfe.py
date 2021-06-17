import os
import pickle

import numpy as np
import jax.numpy as jnp
import functools
import tempfile
import mdtraj

from simtk import openmm
from simtk.openmm import app
from rdkit import Chem

from md import minimizer
from timemachine.lib import potentials
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine import constants
from timemachine.potentials import rmsd
from fe import free_energy, topology, estimator_abfe, model_utils
from ff import Forcefield

from parallel.client import AbstractClient
from typing import Optional
from functools import partial
from scipy.optimize import linear_sum_assignment

from md.barostat.utils import get_group_indices

def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm

def setup_relative_restraints(mol_a, mol_b):
    """
    Setup restraints between ring atoms in two molecules.

    """
    # setup relative orientational restraints
    # rough sketch of algorithm:
    # find core atoms in mol_a
    # find core atoms in mol_b
    # use the hungarian algorithm to assign matching

    ligand_coords_a = get_romol_conf(mol_a)
    ligand_coords_b = get_romol_conf(mol_b)

    core_idxs_a = []
    for idx, a in enumerate(mol_a.GetAtoms()):
        if a.IsInRing():
            core_idxs_a.append(idx)

    core_idxs_b = []
    for idx, b in enumerate(mol_b.GetAtoms()):
        if b.IsInRing():
            core_idxs_b.append(idx)

    ri = np.expand_dims(ligand_coords_a[core_idxs_a], 1)
    rj = np.expand_dims(ligand_coords_b[core_idxs_b], 0)
    rij = np.sqrt(np.sum(np.power(ri-rj, 2), axis=-1))

    row_idxs, col_idxs = linear_sum_assignment(rij)

    core_idxs = []

    for core_a, core_b in zip(row_idxs, col_idxs):
        core_idxs.append((
            core_idxs_a[core_a],
            core_idxs_b[core_b]
        ))

    core_idxs = np.array(core_idxs, dtype=np.int32)

    return core_idxs

class ReferenceAbsoluteModel():
    """
    Absolute free energy using a reference molecule to block the binding pocket.
    """

    def __init__(
        self,
        client: Optional[AbstractClient],
        ff: Forcefield,
        host_system: openmm.System,
        host_coords: np.ndarray,
        host_box: np.ndarray,
        host_schedule: np.ndarray,
        host_topology,
        ref_mol,
        equil_steps: int,
        prod_steps: int):

        self.host_system = host_system
        self.host_coords = host_coords
        self.host_box = host_box
        self.host_schedule = host_schedule
        self.host_topology = host_topology
        self.ref_mol = ref_mol # attached mol has old coordinates
        self.client = client
        self.ff = ff
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps
        # self.standardize = standardize

        # equilibriate the combined structure.
        save_state = "equil.pkl"
        if os.path.exists(save_state):
            print("restoring pickle...")
            self.x0, self.v0, self.box0 = pickle.load(open(save_state, "rb"))
        else:
            print("generating new equilibrium structure from scratch...")
            self.x0, self.v0, self.box0 = minimizer.minimize_host_4d(
                [self.ref_mol],
                self.host_system,
                self.host_coords,
                self.ff,
                self.host_box
            )
            pickle.dump([self.x0, self.v0, self.box0], open(save_state, "wb"))

    def _predict_a_to_b(
        self,
        ff_params,
        mol_a,
        mol_b,
        core_idxs,
        combined_coords,
        prefix):

        dual_topology = topology.DualTopologyStandardDecoupling(mol_a, mol_b, self.ff)
        rfe = free_energy.RelativeFreeEnergy(dual_topology)

        unbound_potentials, sys_params, masses = rfe.prepare_host_edge(
            ff_params,
            self.host_system
        )

        # setup restraints and align to the blocker
        num_host_atoms = len(self.host_coords)
        combined_topology = model_utils.generate_topology(
            [self.host_topology, mol_a, mol_b],
            self.host_coords,
            "complex"+prefix+".pdb"
        )

        # generate initial structure
        coords = combined_coords

        traj = mdtraj.Trajectory([coords], mdtraj.Topology.from_openmm(combined_topology))
        traj.save_xtc("initial_coords_aligned.xtc")

        k_core = 100.0
        core_params = np.zeros_like(core_idxs).astype(np.float64)
        core_params[:, 0] = k_core

        B = len(core_idxs)

        restraint_potential = potentials.HarmonicBond(
            core_idxs,
        )

        unbound_potentials.append(restraint_potential)
        sys_params.append(core_params)

        endpoint_correct = True

        # tbd sample from boltzmann distribution later
        x0 = coords
        v0 = np.zeros_like(coords)

        seed = 0
        temperature = 300.0
        beta = 1/(constants.BOLTZ*temperature)

        bond_list = np.concatenate([unbound_potentials[0].get_idxs(), core_idxs])
        masses = model_utils.apply_hmr(masses, bond_list)

        integrator = LangevinIntegrator(
            temperature,
            2.5e-3,
            1.0,
            masses,
            seed
        )
        bond_list = list(map(tuple, bond_list))
        group_indices = get_group_indices(bond_list)
        barostat_interval = 25

        barostat = MonteCarloBarostat(
            coords.shape[0],
            group_indices,
            1.0,
            temperature,
            barostat_interval,
            seed
        )

        endpoint_correct = True
        model = estimator_abfe.FreeEnergyModel(
            unbound_potentials,
            endpoint_correct,
            self.client,
            self.box0, # important, use equilibrated box.
            x0,
            v0,
            integrator,
            barostat,
            self.host_schedule,
            self.equil_steps,
            self.prod_steps,
            beta,
            prefix
        )

        dG, results = estimator_abfe.deltaG(model, sys_params)

        for idx, result in enumerate(results):
            traj = mdtraj.Trajectory(result.xs, mdtraj.Topology.from_openmm(combined_topology))
            traj.unitcell_vectors = result.boxes
            traj.image_molecules()
            traj.save_xtc(prefix+"_complex_lambda_"+str(idx)+".xtc")

        return dG, results

    def predict(self, ff_params: list, mol: Chem.Mol, prefix: str):
        """
        Predict the ddG of decoupling mol_a and coupling a reference molecule.

        This function is differentiable w.r.t. ff_params.

        Parameters
        ----------

        ff_params: list of np.ndarray
            This should match the ordered params returned by the forcefield

        mol: Chem.Mol
            Molecule we want to decouple

        Returns
        -------
        float
            delta delta G in kJ/mol
        aux
            list of TI results

        """

        host_system = self.host_system
        host_lambda_schedule = self.host_schedule

        # generate indices
        core_idxs = setup_relative_restraints(self.ref_mol, mol)
        mol_coords = get_romol_conf(mol) # original coords
        num_host_atoms = len(self.host_coords)
        R, t = rmsd.get_optimal_rotation_and_translation(
            self.x0[num_host_atoms:][core_idxs[:, 0]], # reference
            mol_coords[core_idxs[:, 1]],
        )

        mol_com = np.mean(mol_coords, axis=0)
        aligned_coords = (mol_coords-mol_com)@R - t + mol_com
        ref_coords = self.x0[num_host_atoms:]
        equil_host_coords = self.x0[:num_host_atoms]

        combined_core_idxs = np.copy(core_idxs)
        combined_core_idxs[:, 0] += num_host_atoms
        combined_core_idxs[:, 1] += num_host_atoms + self.ref_mol.GetNumAtoms()
        combined_coords = np.concatenate([equil_host_coords, ref_coords, aligned_coords])
        dG_0, results_0 = self._predict_a_to_b(
            ff_params,
            self.ref_mol,
            mol,
            combined_core_idxs,
            combined_coords,
            prefix+"_ref_to_mol")

        # do only one pass
        combined_core_idxs = np.copy(core_idxs)
        # swap
        combined_core_idxs[:, 0] = core_idxs[:, 1]
        combined_core_idxs[:, 1] = core_idxs[:, 0]
        combined_core_idxs[:, 0] += num_host_atoms
        combined_core_idxs[:, 1] += num_host_atoms + mol.GetNumAtoms()
        combined_coords = np.concatenate([equil_host_coords, aligned_coords, ref_coords])
        dG_1, results_1 = self._predict_a_to_b(
            ff_params,
            mol,
            self.ref_mol,
            combined_core_idxs,
            combined_coords,
            prefix+"_mol_to_ref")

        # dG_0 is the free energy of moving X-B-A into X-B+A
        # dG_1 is the free energy of moving X-A-B into X-A+B
        # -dG_1 + dG_0 is the free energy of moving X-A+B -> X-B+A
        # i.e. the free energy of "unbinding" A

        return -dG_1 + dG_0

