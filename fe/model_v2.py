import functools
import numpy as np
import jax.numpy as jnp

from simtk import openmm
from rdkit import Chem

from md import minimizer
from timemachine.lib import LangevinIntegrator
from timemachine.lib import potentials
from fe import free_energy, topology, estimator
from ff import Forcefield

from parallel.client import AbstractClient
from typing import Optional
from functools import partial

from ff.handlers import openmm_deserializer


def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm



class ABFEModel():

    def __init__(
        self,
        client: AbstractClient or None,
        ff: Forcefield,
        complex_system: openmm.System,
        complex_coords: np.ndarray,
        complex_box: np.ndarray,
        complex0_schedule: np.ndarray,
        complex1_schedule: np.ndarray,
        complex_topology: np.ndarray,
        solvent_system: openmm.System,
        solvent_coords: np.ndarray,
        solvent_box: np.ndarray,
        solvent_schedule: np.ndarray,
        solvent_topology: np.ndarray,
        equil_steps: int,
        prod_steps: int,
        k_core: float):

        self.complex_system = complex_system
        self.complex_coords = complex_coords
        self.complex_box = complex_box
        self.complex0_schedule = complex0_schedule
        self.complex1_schedule = complex1_schedule
        self.complex_topology = complex_topology
        self.solvent_system = solvent_system
        self.solvent_coords = solvent_coords
        self.solvent_box = solvent_box
        self.solvent_schedule = solvent_schedule
        self.solvent_topology = solvent_topology
        self.client = client
        self.ff = ff
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps
        self.k_core = k_core
        # self.k_translation = k_translation
        # self.k_rotation = k_rotation

    def predict(self,
        ff_params: list,
        mol: Chem.Mol,
        core: np.ndarray,
        epoch: int):
        """
        Predict the ddG of morphing mol_a into mol_b. This function is differentiable w.r.t. ff_params.
        Parameters
        ----------
        ff_params: list of np.ndarray
            This should match the ordered params returned by the forcefield
        mol: Chem.Mol
            Starting molecule corresponding to lambda = 0
        core: np.ndarray
            N list of ints corresponding to the atom mapping of the core.
        Returns
        -------
        float
            delta delta G in kJ/mol
        aux
            list of TI results
        """

        assert len(core.shape) == 1

        stage_dGs = []
        stage_results = []

        for stage, host_system, host_coords, host_box, lambda_schedule, leg_topology in [
            ("complex0", self.complex_system, self.complex_coords, self.complex_box, self.complex0_schedule, self.complex_topology),
            ("complex1", self.complex_system, self.complex_coords, self.complex_box, self.complex1_schedule, self.complex_topology),
            ("solvent", self.solvent_system, self.solvent_coords, self.solvent_box, self.solvent_schedule, self.solvent_topology)]:

            print(f"Minimizing the {stage} host structure to remove clashes.")
            # (ytz): this isn't strictly symmetric, and we should modify minimize later on remove
            # the hysteresis by jointly minimizing against a and b at the same time. We may also want
            # to remove the randomness completely from the minimization.
            min_host_coords = minimizer.minimize_host_4d([mol], host_system, host_coords, self.ff, host_box)

            top = topology.BaseTopology(mol, self.ff)


            top.parameterize_nonbonded = functools.partial(top.parameterize_nonbonded, stage=stage)
            # NA = mol.GetNumAtoms()

            # combined_lambda_plane_idxs = np.zeros(NA, dtype=np.int32)
            # combined_lambda_offset_idxs = np.concatenate([
            #     np.zeros(NA, dtype=np.int32),
            #     np.ones(NB, dtype=np.int32)
            # ])

            # top.parameterize_nonbonded = functools.partial(top.parameterize_nonbonded,
                # minimize=False,
            # )

            rfe = free_energy.AbsoluteFreeEnergy(mol, top)
            unbound_potentials, sys_params, masses, coords = rfe.prepare_host_edge(ff_params, host_system, min_host_coords)

            # setup restraints
            if stage != "solvent":
                ligand_coords = get_romol_conf(mol)
                ri = np.expand_dims(ligand_coords, 1)
                rj = np.expand_dims(host_coords, 0)

                # tbd: use PBCs when doing distance calculations
                # d2ij = np.sum(np.power(delta_r(ri, rj, host_box), 2), axis=-1)
                dij = np.sqrt(np.sum(np.power(ri-rj, 2), axis=-1))

                _, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)

                atom_names = [a.name for a in leg_topology.atoms()]


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

                # print(pocket_atoms)
                # print(dij_pocket.shape)

                from scipy.optimize import linear_sum_assignment

                row_idxs, col_idxs = linear_sum_assignment(dij_pocket)
                num_host_atoms = host_coords.shape[0]

                core_idxs = []
                core_params = []
                for core_i, protein_j in zip(row_idxs, col_idxs):
                    core_idxs.append((core[core_i] + num_host_atoms, pocket_atoms[protein_j]))
                    core_params.append((self.k_core, 0.0))
                    # core_params.append((self.k_core, dij_pocket[core_i, protein_j]))

                core_idxs = np.array(core_idxs, dtype=np.int32)
                core_params = np.array(core_params, dtype=np.float64)

                B = core_idxs.shape[0]

                if stage == "complex0":
                    core_lambda_mult = np.ones(B)
                    core_lambda_offset = np.zeros(B)
                elif stage == "complex1":
                    core_lambda_mult = np.zeros(B)
                    core_lambda_offset = np.ones(B)
                else:
                    assert 0

                unbound_potentials.append(potentials.HarmonicBond(
                    core_idxs,
                    core_lambda_mult.astype(np.int32),
                    core_lambda_offset.astype(np.int32)
                ))
                sys_params.append(core_params)

            x0 = coords
            v0 = np.zeros_like(coords)
            box = np.eye(3, dtype=np.float64)*100 # note: box unused

            seed = 0

            integrator = LangevinIntegrator(
                300.0,
                1.5e-3,
                1.0,
                masses,
                seed
            )

            debug_info = "epoch_"+str(epoch)

            model = estimator.FreeEnergyModel(
                unbound_potentials,
                self.client,
                host_box,
                x0,
                v0,
                integrator,
                lambda_schedule,
                self.equil_steps,
                self.prod_steps,
                leg_topology,
                stage,
                debug_info
            )

            dG, results = estimator.deltaG(model, sys_params)

            stage_dGs.append(dG)
            stage_results.append((stage, results))

        # complex - solvent
        pred = (stage_dGs[0] + stage_dGs[1]) - stage_dGs[2]


        return pred, stage_results

    # def loss(self, ff_params, mol_a, mol_b, core, label_ddG):
    #     """
    #     Computes the L1 loss relative to some label. See predict() for the type signature.
    #     This function is differentiable w.r.t. ff_params.
    #     Parameters
    #     ----------
    #     label_ddG: float
    #         Label ddG in kJ/mol of the alchemical transformation.
    #     Returns
    #     -------
    #     float
    #         loss
    #     TODO: make this configurable, using loss functions from in fe/loss.py
    #     """
    #     pred_ddG, results = self.predict(ff_params, mol_a, mol_b, core)
    #     loss = jnp.abs(pred_ddG - label_ddG)
    #     return loss, results