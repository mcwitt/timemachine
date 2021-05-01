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
    Setup rigid restraint between the protein and the core atoms in the ligand.

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
        # later on.
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
        min_host_coords = minimizer.minimize_host_4d([mol], self.host_system, self.host_coords, self.ff, self.host_box)

        afe = free_energy.AbsoluteFreeEnergy(mol, self.ff)

        unbound_potentials, sys_params, masses, coords = afe.prepare_host_edge(
            ff_params,
            self.host_system,
            min_host_coords
        )

        # setup lambda transformations
        transform_qlj = "lambda < 0.5 ? sin(lambda*PI)*sin(lambda*PI) : 1"
        transform_w = "lambda < 0.5 ? 0.0 : sin((lambda+0.5)*PI)*sin((lambda+0.5)*PI)"
        # transform_qlj = "lambda"
        # transform_w = "lambda"

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
            B = len(core_idxs)
            core_lambda_mult = np.ones(B)
            core_lambda_offset = np.zeros(B)

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
            cache_results
        )

        dG, results = estimator_abfe.deltaG(model, sys_params)

        return dG, results



    # def stage_0(
    #     self,
    #     sys_params,
    #     mol,
    #     restraint):
    #     """
    #     Alchemically scales the nonbonded parameters of the decoupling ligand to a neutral state.

    #     Charges are interpolated to zero and vdw/eps are interpolated to 0.2/0.2. If restraints are present,
    #     they will be alchemically turned on during this stage.
    #     """

    #     print(f"Minimizing the {stage} host structure to remove clashes.")
    #     min_host_coords = minimizer.minimize_host_4d([mol], host_system, host_coords, self.ff, host_box)

    #     top = topology.BaseTopology(mol, self.ff)
    #     top.parameterize_nonbonded = functools.partial(top.parameterize_nonbonded, stage=stage)

    #     afe = free_energy.AbsoluteFreeEnergy(mol, top)
    #     unbound_potentials, sys_params, masses, coords = afe.prepare_host_edge(ff_params, host_system, min_host_coords)

    #     # setup restraints
    #     core_idxs, core_params = setup_restraints(mol, core, self.complex_topology, self.solvent_topology, self.k_core)

    #     core_lambda_mult = np.ones(B)
    #     core_lambda_offset = np.zeros(B)

    #     bound_restraint_potential = potentials.HarmonicBond(
    #         core_idxs,
    #         core_lambda_mult.astype(np.int32),
    #         core_lambda_offset.astype(np.int32)
    #     ).bind(core_params)

    #     # sys_params.append(core_params)

    #     x0 = coords
    #     v0 = np.zeros_like(coords)

    #     seed = 0

    #     integrator = LangevinIntegrator(
    #         300.0,
    #         1.5e-3,
    #         1.0,
    #         masses,
    #         seed
    #     )

    #     debug_info = "epoch_"+str(epoch)

    #     model = estimator.FreeEnergyModel(
    #         unbound_potentials,
    #         self.client,
    #         host_box,
    #         x0,
    #         v0,
    #         integrator,
    #         lambda_schedule,
    #         self.equil_steps,
    #         self.prod_steps,
    #         leg_topology,
    #         stage,
    #         debug_info
    #     )

    #     dG, results = estimator.deltaG(model, sys_params)

    #     return dG, results
    #     # stage_dGs.append(dG)
    #     # stage_results.append((stage, results))



    # def predict(self,
    #     ff_params: list,
    #     reorg_dg: float,
    #     mol: Chem.Mol,
    #     core: np.ndarray,
    #     # complex1_dg: float,
    #     # solvent1_dg: float,
    #     epoch: int):
    #     """
    #     Predict the ddG of morphing mol_a into mol_b. This function is differentiable w.r.t. ff_params.

    #     Parameters
    #     ----------
    #     ff_params: list of np.ndarray
    #         This should match the ordered params returned by the forcefield
    #     reorg_dg: float
    #         delta g of protein reorganization
    #     mol: Chem.Mol
    #         Starting molecule corresponding to lambda = 0
    #     core: np.ndarray
    #         N list of ints corresponding to the atom mapping of the core.
    #     complex1_dg: float
    #         delta g of the complex1 stage.
    #     solvent1_dg: float
    #         delta g of the solvent1 stage.

    #     Returns
    #     -------
    #     float
    #         delta delta G in kJ/mol
    #     aux
    #         list of TI results

    #     """

    #     complex_dg = self.predict_complex0() + self.predict_complex1()
    #     solvent_dg = self.predict_solvent0() + self.predict_complex1()

    #     return complex_dg - solvent_dg + reorg_dg

    #     assert len(core.shape) == 1

    #     stage_dGs = []
    #     stage_results = []

    #     for stage, host_system, host_coords, host_box, lambda_schedule, leg_topology in [
    #         ("complex0", self.complex_system, self.complex_coords, self.complex_box, self.complex0_schedule, self.complex_topology),
    #         ("complex1", self.complex_system, self.complex_coords, self.complex_box, self.complex1_schedule, self.complex_topology),
    #         ("solvent", self.solvent_system, self.solvent_coords, self.solvent_box, self.solvent_schedule, self.solvent_topology)]:

    #         print(f"Minimizing the {stage} host structure to remove clashes.")
    #         # (ytz): this isn't strictly symmetric, and we should modify minimize later on remove
    #         # the hysteresis by jointly minimizing against a and b at the same time. We may also want
    #         # to remove the randomness completely from the minimization.
    #         min_host_coords = minimizer.minimize_host_4d([mol], host_system, host_coords, self.ff, host_box)

    #         top = topology.BaseTopology(mol, self.ff)
    #         top.parameterize_nonbonded = functools.partial(top.parameterize_nonbonded, stage=stage)

    #         afe = free_energy.AbsoluteFreeEnergy(mol, top)
    #         unbound_potentials, sys_params, masses, coords = afe.prepare_host_edge(ff_params, host_system, min_host_coords)

    #         # setup restraints
    #         if stage != "solvent":

    #             B = core_idxs.shape[0]

    #             if stage == "complex0":
    #                 core_lambda_mult = np.ones(B)
    #                 core_lambda_offset = np.zeros(B)
    #             elif stage == "complex1":
    #                 core_lambda_mult = np.zeros(B)
    #                 core_lambda_offset = np.ones(B)
    #             else:
    #                 assert 0

    #             unbound_potentials.append(potentials.HarmonicBond(
    #                 core_idxs,
    #                 core_lambda_mult.astype(np.int32),
    #                 core_lambda_offset.astype(np.int32)
    #             ))
    #             sys_params.append(core_params)

    #         x0 = coords
    #         v0 = np.zeros_like(coords)
    #         box = np.eye(3, dtype=np.float64)*100 # note: box unused

    #         seed = 0

    #         integrator = LangevinIntegrator(
    #             300.0,
    #             1.5e-3,
    #             1.0,
    #             masses,
    #             seed
    #         )

    #         debug_info = "epoch_"+str(epoch)

    #         model = estimator.FreeEnergyModel(
    #             unbound_potentials,
    #             self.client,
    #             host_box,
    #             x0,
    #             v0,
    #             integrator,
    #             lambda_schedule,
    #             self.equil_steps,
    #             self.prod_steps,
    #             leg_topology,
    #             stage,
    #             debug_info
    #         )

    #         dG, results = estimator.deltaG(model, sys_params)

    #         stage_dGs.append(dG)
    #         stage_results.append((stage, results))

    #     # complex - solvent
    #     pred = (stage_dGs[0] + stage_dGs[1]) - stage_dGs[2]


    #     return pred, stage_results

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