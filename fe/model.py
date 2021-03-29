import numpy as np
import jax.numpy as jnp

from simtk import openmm
from rdkit import Chem

from md import minimizer
from timemachine.lib import LangevinIntegrator
from fe import free_energy, topology, estimator
from ff import Forcefield

from parallel.client import AbstractClient
from typing import Optional
from functools import partial

from ff.handlers import openmm_deserializer


def delta_r(ri, rj, box=None):
    diff = ri - rj # this can be either N,N,3 or B,3
    dims = ri.shape[-1]

    # box is None for harmonic bonds, not None for nonbonded terms
    if box is not None:
        for d in range(dims):
            diff -= box[d]*np.floor(np.expand_dims(diff[...,d], axis=-1)/box[d][d]+0.5)

    return diff

def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm

def prepare_dual_topology(mol_a, mol_b, core, ff):
    """
    Generates dual topologies for mol_a and mol_b. This generates
    the topologies needed for the TI calculation as well as that
    of the end-state. mol_a is designed to be fully interacting
    at all times, and mol_b is slowly decoupled in TI, and fully
    non-interacting in the end-state. The end-state must be ran
    with lambda=1.0.

    The generated sys_params are identical for both topologies.

    Parameters
    ----------
    mol_a: ROMol
        Fully interacting molecule

    mol_b: ROMol
        Decoupled molecule

    core: np.array
        Atom-mapping

    ff: Forcefield
        forcefield to be used

    Returns
    -----
    2-tuple of Topology
        Returns (end_state_topology, ti_topology)

    """

    topo_ti = topology.DualTopology(mol_a, mol_b, ff)

    core_k = 35
    core_b = None

    topo_ti.parameterize_harmonic_bond = partial(
        topo_ti.parameterize_harmonic_bond,
        core=core,
        core_k=core_k,
        core_b=core_b,
        core_lambda_mult=0.0,
        core_lambda_offset=1.0
    )

    topo_ti.parameterize_nonbonded = partial(
        topo_ti.parameterize_nonbonded,
        mol_a_lambda_offset=0,
        mol_a_lambda_plane=0,
        mol_b_lambda_offset=1,
        mol_b_lambda_plane=0,
    )

    topo_end_state = topology.DualTopology(mol_a, mol_b, ff)

    # when lambda = 1 the core restraints are fully turned off:
    # prefactor = offset + mult*lambda
    topo_end_state.parameterize_harmonic_bond = partial(
        topo_end_state.parameterize_harmonic_bond,
        core=core,
        core_k=core_k,
        core_b=core_b,
        core_lambda_mult=-1.0,
        core_lambda_offset=1.0
    )

    topo_end_state.parameterize_nonbonded = partial(
        topo_end_state.parameterize_nonbonded,
        mol_a_lambda_offset=0,
        mol_a_lambda_plane=0,
        mol_b_lambda_offset=0,
        mol_b_lambda_plane=1,
    )

    return topo_ti, topo_end_state


class BFEModel():

    def __init__(
        self,
        client: AbstractClient or None,
        ff: Forcefield,
        complex_system: openmm.System,
        complex_coords: np.ndarray,
        complex_box: np.ndarray,
        complex_schedule: np.ndarray,
        solvent_system: openmm.System,
        solvent_coords: np.ndarray,
        solvent_box: np.ndarray,
        solvent_schedule: np.ndarray,
        equil_steps: int,
        prod_steps: int,
        complex_topology,
        solvent_topology):

        self.complex_system = complex_system
        self.complex_coords = complex_coords
        self.complex_box = complex_box
        self.complex_schedule = complex_schedule

        self.solvent_system = solvent_system
        self.solvent_coords = solvent_coords
        self.solvent_box = solvent_box
        self.solvent_schedule = solvent_schedule
        self.client = client
        self.ff = ff
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps

        self.complex_topology = complex_topology
        self.solvent_topology = solvent_topology

    def predict_abfe(self, ff_params: list, mol: Chem.Mol, ligand_core: np.ndarray, epoch: int):
        """
        Predict the dG of mol. This function is differentiable w.r.t. ff_params.

        Parameters
        ----------

        ff_params: list of np.ndarray
            This should match the ordered params returned by the forcefield

        mol: Chem.Mol
            Starting molecule corresponding to lambda = 0

        core: np.ndarray
            N x 1 list of ints corresponding to the core that we should restrain
            to the protein's alpha carbon. Needed only for the complex leg.

        Returns
        -------
        float
            delta G in kJ/mol

        aux
            list of TI results
        """

        stage_dGs = []
        stage_results = []

        for stage, host_system, host_coords, host_box, lambda_schedule, omm_topology in [
            ("complex", self.complex_system, self.complex_coords, self.complex_box, self.complex_schedule, self.complex_topology),
            ]:
            # ("solvent", self.solvent_system, self.solvent_coords, self.solvent_box, self.solvent_schedule, self.solvent_topology)]:

            print("Minimizing the host structure to remove clashes.")
            min_host_coords = minimizer.minimize_host_4d([mol], host_system, host_coords, self.ff, host_box)
            topo = topology.BaseTopology(mol, self.ff)
            afe = free_energy.AbsoluteFreeEnergy(topo)


            # core_k = 30.0

            print("LIGAND_CORE", ligand_core)


            ligand_coords = get_romol_conf(mol)
            ri = np.expand_dims(ligand_coords, 1)
            rj = np.expand_dims(host_coords, 0)

            # use PBCs when doing distance calculations
            # d2ij = np.sum(np.power(delta_r(ri, rj, host_box), 2), axis=-1)
            dij = np.sqrt(np.sum(np.power(ri-rj, 2), axis=-1))

            _, host_masses = openmm_deserializer.deserialize_system(host_system, cutoff=1.2)

            def is_carbon(idx):
                return host_masses[idx] > 12.0 and host_masses[idx] < 12.1

            atom_names = []

            for a in omm_topology.atoms():
                atom_names.append(a.name)

            def is_alpha_carbon(idx):
                return atom_names[idx] == 'CA'
                # print(a.name)
            # print(omm_topology._chains[0]._chain)

            # assert 0


            core_idxs = []
            core_params = []

            # what if one of these c-alphas is a highly motile loop?
            pocket_atoms = set()

            # 5 angstrom radius
            pocket_cutoff = 0.5

            for i_idx in range(mol.GetNumAtoms()):
                if i_idx in ligand_core:
                    dists = dij[i_idx]
                    for j_idx, dist in enumerate(dists):
                        if is_alpha_carbon(j_idx):
                            if dist < pocket_cutoff:
                                pocket_atoms.add(j_idx)

            pocket_atoms = np.array(list(pocket_atoms))


            ri = np.expand_dims(ligand_coords, 1)
            rj = np.expand_dims(host_coords[pocket_atoms], 0)
            dij_pocket = np.sqrt(np.sum(np.power(ri-rj, 2), axis=-1))

            for i_idx in range(mol.GetNumAtoms()):
                if i_idx in ligand_core:
                    dists = dij_pocket[i_idx]
                    j_idx = np.argsort(dists)[0]
                    core_idxs.append((pocket_atoms[j_idx], i_idx))
                    core_params.append((10.0, 0.0))

            # for i_idx in range(mol.GetNumAtoms()):
            #     if i_idx in ligand_core:
            #         dists = dij[i_idx]
            #         closest_jdxs = []
            #         closests = []
            #         for j_idx, d2 in enumerate(dists):
            #             # if is_alpha_carbon(j_idx) and (closest is None or d2 < closest):
            #             if is_alpha_carbon(j_idx):
            #                 closests.append(d2)
            #                 closest_jdxs.append(j_idx)

            #         # assert closest_jdx is not None
            #         closests = np.array(closests)
            #         closest_jdxs = np.array(closest_jdxs)

            #         perm = np.argsort(closests)[:]
            #         for d2, jj in zip(closests[perm], closest_jdxs[perm]):
            #             core_idxs.append((jj, i_idx))
            #             # core_params.append((30.0, np.sqrt(d2)))
            #             # placate RMSD
            #             core_params.append((10.0, 0.0))

            core_idxs = np.array(core_idxs)

            print("core_idxs host", core_idxs[:, 0])
            print("core_idxs ligand", core_idxs[:, 1])
            # assert 0

            unbound_potentials_end_state, sys_params_end_state, masses_end_state, coords_end_state = afe.prepare_host_edge(
                ff_params,
                host_system,
                min_host_coords,
                core_idxs,
                core_params,
                core_lambda_mult=-1.0,
                core_lambda_offset=1.0
            )

            afe = free_energy.AbsoluteFreeEnergy(topo)
            unbound_potentials_ti, sys_params_ti, masses_ti, coords_ti = afe.prepare_host_edge(
                ff_params,
                host_system,
                min_host_coords,
                core_idxs,
                core_params,
                core_lambda_mult=0.0,
                core_lambda_offset=1.0
            )

            # rfe.prepare_host_edge(ff_params, host_system, min_host_coords)

            np.testing.assert_array_equal(masses_end_state, masses_ti)
            np.testing.assert_array_equal(coords_end_state, coords_ti)

            assert len(sys_params_end_state) == len(sys_params_ti)

            for idx, (a, b) in enumerate(zip(sys_params_end_state, sys_params_ti)):
                np.testing.assert_array_equal(a, b)

            x0 = coords_ti
            v0 = np.zeros_like(coords_ti)

            seed = 0

            combined_lambda_schedule = np.concatenate([
                lambda_schedule,
                [1.0], # endpoint
            ])

            integrator = LangevinIntegrator(300.0, 1.5e-3, 1.0, masses_ti, seed)

            model = estimator.FreeEnergyModel(
                unbound_potentials_ti,
                unbound_potentials_end_state,
                None, # self.client,
                host_box,
                x0, # combined_coords,
                v0, # combined_velocities,
                integrator, # combined_integrator,
                combined_lambda_schedule, # combined_lambda_schedule,
                self.equil_steps,
                self.prod_steps,
                core_idxs,
                omm_topology,
                stage+"_"+str(epoch),
                host_coords.shape[0],
            )

            # combined_sys_params = [
            #     sys_params_AB,
            #     *([sys_params_B]*len(lambda_schedule)),
            #     *([sys_params_A]*len(lambda_schedule)),
            #     sys_params_BA
            # ]

            dG, results = estimator.deltaG(model, sys_params_ti)

            stage_dGs.append(dG)
            stage_results.append((stage, results))

        # (ytz): don't for get to undo me
        # pred = stage_dGs[0] - stage_dGs[1]
        pred = stage_dGs[0]

        return pred, stage_results


    def predict_rbfe(self, ff_params: list, mol_a: Chem.Mol, mol_b: Chem.Mol, core: np.ndarray, epoch: int):
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

        stage_dGs = []
        stage_results = []

        for stage, host_system, host_coords, host_box, lambda_schedule, topology in [
            ("complex", self.complex_system, self.complex_coords, self.complex_box, self.complex_schedule, self.complex_topology),
            ("solvent", self.solvent_system, self.solvent_coords, self.solvent_box, self.solvent_schedule, self.solvent_topology)]:

            print("Minimizing the host structure to remove clashes.")
            # (ytz): this isn't strictly symmetric, and we should modify minimize later on remove
            # the hysteresis by jointly minimizing against a and b at the same time. We may also want
            # to remove the randomness completely from the minimization.
            min_host_coords = minimizer.minimize_host_4d([mol_a, mol_b], host_system, host_coords, self.ff, host_box)

            if False:
                assert 0
                topo = topology.SingleTopology(mol_a, mol_b, core, self.ff)

            else:

                assert 0

                topo_ti, topo_end_state = prepare_dual_topology(mol_a, mol_b, core, self.ff)

                rfe = free_energy.RelativeFreeEnergy(topo_end_state)
                unbound_potentials_end_state, sys_params_end_state, masses_end_state, coords_end_state = rfe.prepare_host_edge(ff_params, host_system, min_host_coords)
                rfe = free_energy.RelativeFreeEnergy(topo_ti)
                unbound_potentials_ti, sys_params_ti, masses_ti, coords_ti = rfe.prepare_host_edge(ff_params, host_system, min_host_coords)

                np.testing.assert_array_equal(masses_end_state, masses_ti)
                np.testing.assert_array_equal(coords_end_state, coords_ti)

                assert len(sys_params_end_state) == len(sys_params_ti)

                for idx, (a, b) in enumerate(zip(sys_params_end_state, sys_params_ti)):
                    np.testing.assert_array_equal(a, b)

                x0 = coords_ti
                v0 = np.zeros_like(coords_ti)

                seed = 0

                combined_lambda_schedule = np.concatenate([
                    lambda_schedule,
                    [1.0], # endpoint
                ])

                integrator = LangevinIntegrator(300.0, 1.5e-3, 1.0, masses_ti, seed)

                model = estimator.FreeEnergyModel(
                    unbound_potentials_ti,
                    unbound_potentials_end_state,
                    # self.client,
                    None,
                    host_box,
                    x0, # combined_coords,
                    v0, # combined_velocities,
                    integrator, # combined_integrator,
                    combined_lambda_schedule, # combined_lambda_schedule,
                    self.equil_steps,
                    self.prod_steps,
                    mol_a.GetNumAtoms(),
                    mol_b.GetNumAtoms(),
                    core,
                    topology,
                    stage+"_"+str(epoch)
                )

                # combined_sys_params = [
                #     sys_params_AB,
                #     *([sys_params_B]*len(lambda_schedule)),
                #     *([sys_params_A]*len(lambda_schedule)),
                #     sys_params_BA
                # ]

                dG, results = estimator.deltaG(model, sys_params_ti)

                stage_dGs.append(dG)
                stage_results.append((stage, results))

        pred = stage_dGs[0] - stage_dGs[1]

        return pred, stage_results

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
