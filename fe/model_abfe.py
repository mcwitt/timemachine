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

from md.barostat.utils import get_group_indices, get_bond_list


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
        prefix):

        print(f"Minimizing the host structure to remove clashes.")
        minimized_coords, _, _ = minimizer.minimize_host_4d(
            [mol],
            self.host_system,
            self.host_coords,
            self.ff,
            self.host_box
        )

        afe = free_energy.AbsoluteFreeEnergy(mol, self.ff)

        unbound_potentials, sys_params, masses = afe.prepare_host_edge(
            ff_params,
            self.host_system,
        )

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

        bond_list = get_bond_list(unbound_potentials[0])
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
