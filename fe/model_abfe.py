import functools
import numpy as np
import jax.numpy as jnp

from simtk import openmm
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
from simtk import unit

from ff.handlers import openmm_deserializer
from scipy.optimize import linear_sum_assignment

def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm

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
    pocket_cutoff = 0.7

    for i_idx in range(mol.GetNumAtoms()):
        if i_idx in core:
            dists = dij[i_idx]
            for j_idx, dist in enumerate(dists):
                if is_alpha_carbon(j_idx):
                    if dist < pocket_cutoff:
                        pocket_atoms.add(j_idx)

    pocket_atoms = np.array(list(pocket_atoms))

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
        # (ytz): intentionally left commented out in-case we want to try a non-0 value
        # later on. But this will cause poor overlap with the RMSD restraint when we
        # do the endpoint correction.
        # core_params.append((self.k_core, dij_pocket[core_i, protein_j]))

    core_idxs = np.array(core_idxs, dtype=np.int32)
    core_params = np.array(core_params, dtype=np.float64)

    return core_idxs, core_params


def setup_centroid_restraints(
    mol,
    host_topology,
    host_coords):
    """
    Setup rigid restraint between protein c-alpha and the core atoms in the ligand.
    It is assumed that the ligand is properly positioned within the protein.

    Parameters
    ----------
    mol: Chem.Mol
        molecule with conformer information specified

    host_topology: openmm.Topology
        Topology of the host

    host_coords: Nx3
        Coordinates of the host.

    Returns
    -------
    potentials.CentroidRestraint

    """
    ligand_coords = get_romol_conf(mol)

    # print(ligand_coords)
    host_coords = host_coords.value_in_unit(unit.nanometers)
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
        # if i_idx in core:
        dists = dij[i_idx]
        for j_idx, dist in enumerate(dists):
            if is_alpha_carbon(j_idx):
                if dist < pocket_cutoff:
                    pocket_atoms.add(j_idx)

    pocket_atoms = np.array(list(pocket_atoms))

    # ri = np.expand_dims(ligand_coords[core], 1)
    # rj = np.expand_dims(host_coords[pocket_atoms], 0)
    # dij_pocket = np.sqrt(np.sum(np.power(ri-rj, 2), axis=-1))

    # row_idxs, col_idxs = linear_sum_assignment(dij_pocket)
    num_host_atoms = host_coords.shape[0]

    # core_idxs = []
    # core_params = []
    # for core_i, protein_j in zip(row_idxs, col_idxs):
    #     core_idxs.append((core[core_i] + num_host_atoms, pocket_atoms[protein_j]))
    #     core_params.append((k_core, 0.0))
    #     # (ytz): intentionally left commented out in-case we want to try a non-0 value
    #     # later on. But this will cause poor overlap with the RMSD restraint when we
    #     # do the endpoint correction.
    #     # core_params.append((self.k_core, dij_pocket[core_i, protein_j]))

    # core_idxs = np.array(core_idxs, dtype=np.int32)
    # core_params = np.array(core_params, dtype=np.float64)

    ligand_atoms = np.arange(mol.GetNumAtoms()) + num_host_atoms

    b0 = np.linalg.norm(np.mean(host_coords[pocket_atoms], axis=0) - np.mean(ligand_coords, axis=0)) + 0.01
    # print(np.mean(pocket_atoms, axis=0))
    # print(np.mean(ligand_atoms, axis=0))
    # print(b0)

    # assert 0
    # b0 = 0
    k = 200.0

    # print("b0", b0)

    return potentials.CentroidRestraint(
        pocket_atoms.astype(np.int32),
        ligand_atoms.astype(np.int32),
        k,
        b0
    )


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
        min_host_coords = minimizer.minimize_host_4d([mol], self.host_system, self.host_coords, self.ff, self.host_box)

        afe = free_energy.AbsoluteFreeEnergy(mol, self.ff)

        unbound_potentials, sys_params, masses, coords = afe.prepare_host_edge(
            ff_params,
            self.host_system,
            min_host_coords
        )

        # setup lambda transformations
        # transform_qlj = "lambda < 0.5 ? sin(lambda*PI)*sin(lambda*PI) : 1"
        # transform_w = "lambda < 0.5 ? 0.0 : sin((lambda+0.5)*PI)*sin((lambda+0.5)*PI)"
        num_host_atoms = self.host_coords.shape[0]
        transform_qlj = "return lambda;"
        transform_w = """
            if(atom_idx < """ + str(num_host_atoms) + """) { return 0; }
            double offset = (atom_idx - """ + str(num_host_atoms) + """)/ (2.0*(N-"""+str(num_host_atoms)+"""));
            if(lambda < offset) {
                return 0;
            } else if (lambda > (0.5 + offset)) {
                return 1;
            } else {
                NumericType term = sin((lambda - offset)*PI);
                return term*term;
            }
        """
        print(transform_w)

        # assert 0
        # transform_w = "return lambda;"
        cache_lambda = 1.0 # if lambda <= cache_lambda then we re-run the simulation.
        # transform_qlj = "lambda"
        # transform_w = "lambda"

        # guest_idxs = np.arange(mol.GetNumAtoms()) + len(min_host_coords)
        # print(guest_idxs.tolist())
        # assert 0

        nonbonded = unbound_potentials[-1]
        nonbonded.args.extend([
            # guest_idxs.tolist(),
            transform_qlj,
            transform_qlj,
            transform_qlj,
            transform_w]
        )


        if restraints:


            # core = []
            # for a in mol.GetAtoms():
            #     if a.IsInRing():
            #         core.append(a.GetIdx())
            # k_core = 50.0
            # core_idxs, core_params = setup_restraints(mol, core, self.host_topology, self.host_coords, k_core)
            # B = len(core_idxs)
            # core_lambda_mult = np.ones(B)
            # core_lambda_offset = np.zeros(B)

            # restraint_potential = potentials.HarmonicBond(
            #     core_idxs,
            #     core_lambda_mult.astype(np.int32),
            #     core_lambda_offset.astype(np.int32)
            # )
            # unbound_potentials.append(restraint_potential)
            # sys_params.append(core_params)

            restraint_potential = setup_centroid_restraints(
                mol,
                self.host_topology,
                self.host_coords
            )
            unbound_potentials.append(restraint_potential)
            sys_params.append(np.array([], dtype=np.float64))

            endpoint_correct = False
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

        if cache_results is None:
            cache_results = [None]*len(self.host_schedule)
            if endpoint_correct:
                cache_results.append(None)

        model = estimator_abfe.FreeEnergyModel(
            unbound_potentials,
            endpoint_correct,
            self.client,
            self.host_box,
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

        return dG, results
