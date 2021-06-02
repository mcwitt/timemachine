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
from fe import free_energy, topology, estimator_abfe
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
        combined_mol = Chem.CombineMols(combined_mol, mol)

    # with tempfile.NamedTemporaryFile(mode='w') as fp:
    fp = open(out_filename, "w")
    # write
    Chem.MolToPDBFile(combined_mol, out_filename)
    fp.flush()
    # read
    combined_pdb = app.PDBFile(out_filename)
    return combined_pdb.topology

def setup_relative_restraints(
    mol_a,
    mol_b,
    k_core):

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
    core_params = []

    for core_a, core_b in zip(row_idxs, col_idxs):
        core_idxs.append((
            core_idxs_a[core_a],
            core_idxs_b[core_b]
        ))
        core_params.append((k_core, 0.0))

    core_idxs = np.array(core_idxs, dtype=np.int32)
    core_params = np.array(core_params, dtype=np.float64)

    # print(core_idxs)
    # print(core_params)

    return core_idxs, core_params

def apply_hmr(masses, bond_list, multiplier=2):

    def is_hydrogen(i):
        return np.abs(masses[i] - 1.00794) < 1e-3

    for i, j in bond_list:
        i, j = np.array([i, j])[np.argsort([masses[i], masses[j]])]
        if is_hydrogen(i):
            if is_hydrogen(j):
                # H-H, skip
                continue
            else:
                # H-X
                # order of operations is important!
                masses[j] -= multiplier*masses[i]
                masses[i] += multiplier*masses[i]
        else:
            # do nothing
            continue

    return masses


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

        # equilibriate the combined structure.
        save_state = "equil.pkl"
        if os.path.exists(save_state):
            print("restoring pickle")
            self.x0, self.v0, self.box0 = pickle.load(open(save_state, "rb"))
        else:
            self.x0, self.v0, self.box0 = minimizer.minimize_host_4d(
                [self.ref_mol],
                self.host_system,
                self.host_coords,
                self.ff,
                self.host_box
            )
            pickle.dump([self.x0, self.v0, self.box0], open(save_state, "wb"))

    def predict(self, ff_params: list, mol: Chem.Mol, prefix: str):
        """
        Predict the ddG of morphing mol_a into mol_b. This function is differentiable w.r.t. ff_params.

        Parameters
        ----------

        ff_params: list of np.ndarray
            This should match the ordered params returned by the forcefield

        mol_a: Chem.Mol
            Starting molecule corresponding to lambda = 0

        mol_b: Chem.Mol
            Starting molecule corresponding to lambda = 1

        core: np.ndarray
            N x 2 list of ints corresponding to the atom mapping of the core.

        Returns
        -------
        float
            delta delta G in kJ/mol
        aux
            list of TI results
        """

        # tbd rmsd align core using old mapping.

        stage_dGs = []
        stage_results = []

        host_system = self.host_system
        # host_coords = self.host_coords
        # host_box = self.host_box
        host_lambda_schedule = self.host_schedule

        print(f"Minimizing the host structure to remove clashes.")
        # min_host_coords = minimizer.minimize_host_4d(
        #     [self.ref_mol, mol],
        #     host_system,
        #     host_coords,
        #     self.ff,
        #     host_box
        # )

        # tbd swap this
        dual_topology = topology.DualTopology(self.ref_mol, mol, self.ff)
        dual_topology.parameterize_nonbonded = functools.partial(
            dual_topology.parameterize_nonbonded,
            minimize=False)
        rfe = free_energy.RelativeFreeEnergy(dual_topology)

        unbound_potentials, sys_params, masses = rfe.prepare_host_edge(
            ff_params,
            host_system
        )

        # (ytz): MODIFY ME WHEN WE SWAP
        mol_coords = get_romol_conf(mol) # original coords
        # mol_coords =  # original coords
        # x0 includes "ref_mol" coords
        # atom_names = [a.name for a in self.host_topology.atoms()]
        # def is_alpha_carbon(idx):
        #     return atom_names[idx] == 'CA'
        # print("freezing C-alphas...")
        # num_host_atoms = len(min_host_coords)
        # for idx in range(num_host_atoms):
        #     if atom_names[idx] == 'CA':
        #         masses[idx] = 100000.0

        # setup restraints and align to the blocker
        k_core = 100.0

        num_host_atoms = len(self.host_coords)
        core_idxs, core_params = setup_relative_restraints(self.ref_mol, mol, k_core)

        # rmsd align target mol onto reference
        R, t = rmsd.get_optimal_rotation_and_translation(
            self.x0[num_host_atoms:][core_idxs[:, 0]], # reference
            mol_coords[core_idxs[:, 1]]
        )

        mol_com = np.mean(mol_coords, axis=0)
        aligned_mol = (mol_coords-mol_com)@R - t + mol_com
        combined_topology = generate_topology([self.host_topology, self.ref_mol, mol], self.host_coords, "complex.pdb")

        # generate initial structure
        coords = np.concatenate([self.x0, aligned_mol])

        traj = mdtraj.Trajectory([coords], mdtraj.Topology.from_openmm(combined_topology))
        traj.save_xtc("initial_coords_aligned.xtc")

        # offset core_idxs appropriately
        core_idxs[:, 0] += num_host_atoms
        core_idxs[:, 1] += num_host_atoms + self.ref_mol.GetNumAtoms()

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
        masses = apply_hmr(masses, bond_list)

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
            # cache_results,
            # cache_lambda
        )

        dG, results = estimator_abfe.deltaG(model, sys_params)

        for idx, result in enumerate(results):
            # print(result.xs.shape)
            traj = mdtraj.Trajectory(result.xs, mdtraj.Topology.from_openmm(combined_topology))
            traj.unitcell_vectors = result.boxes
            traj.image_molecules()
            traj.save_xtc("complex_lambda_"+str(idx)+".xtc")

        assert 0

        return dG, results

        # stage_dGs.append(dG)
        # stage_results.append((stage, results))


        # pred = stage_dGs[0] - stage_dGs[1]

        # return pred, stage_results

    def loss(self, ff_params, mol_a, mol_b, core, label_ddG):
        """
        Computes the L1 loss relative to some label. See predict() for the type signature.

        This function is differentiable w.r.t. ff_params.

        Parameters
        ----------
        label_ddG: float
            Label ddG in kJ/mol of the alchemical transformation.

        Returns
        -------
        float
            loss

        TODO: make this configurable, using loss functions from in fe/loss.py

        """
        pred_ddG, results = self.predict(ff_params, mol_a, mol_b, core)
        loss = jnp.abs(pred_ddG - label_ddG)
        return loss, results
