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
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine.lib import potentials
from fe import free_energy, topology, estimator_abfe, model_utils
from ff import Forcefield

from parallel.client import AbstractClient
from typing import Optional
from functools import partial

from ff.handlers import openmm_deserializer
from scipy.optimize import linear_sum_assignment
import scipy.spatial
from simtk import unit

import matplotlib.pyplot as plt

from md.barostat.utils import get_group_indices, get_bond_list


def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm

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
        prefix,
        standardize=False):

        print(f"Minimizing the host structure to remove clashes.")
        minimized_coords, _, _ = minimizer.minimize_host_4d(
            [mol],
            self.host_system,
            self.host_coords,
            self.ff,
            self.host_box
        )

        if standardize:
            top = topology.BaseTopologyStandardDecoupling(mol, self.ff)
        else:
            top = topology.BaseTopology(mol, self.ff)

        afe = free_energy.AbsoluteFreeEnergy(mol, self.ff)

        unbound_potentials, sys_params, masses = afe.prepare_host_edge(
            ff_params,
            self.host_system,
            top
        )

        endpoint_correct = False

        seed = 0

        temperature = 300.0
        beta = 1/(constants.BOLTZ*temperature)

        bond_list = get_bond_list(unbound_potentials[0])
        masses = model_utils.apply_hmr(masses, bond_list)

        integrator = LangevinIntegrator(
            temperature,
            2.5e-3,
            1.0,
            masses,
            seed
        )

        group_indices = get_group_indices(bond_list)
        barostat_interval = 5
        barostat = MonteCarloBarostat(
            minimized_coords.shape[0],
            group_indices,
            1.0,
            temperature,
            barostat_interval,
            seed
        )

        x0 = minimized_coords
        v0 = np.zeros_like(minimized_coords)

        model = estimator_abfe.FreeEnergyModel(
            unbound_potentials,
            endpoint_correct,
            self.client,
            self.host_box,
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

        return dG

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
