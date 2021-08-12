# Test that we can adjust parameters to make the conversion dG correspond to
# a desired value


import os
import pickle
import argparse
import numpy as np

from fe.free_energy_rabfe import construct_absolute_lambda_schedule_complex, construct_absolute_lambda_schedule_solvent, construct_conversion_lambda_schedule, get_romol_conf, setup_relative_restraints_using_smarts
from fe.utils import convert_uM_to_kJ_per_mole
from fe import model_rabfe
from fe.free_energy_rabfe import RABFEResult

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from ff.handlers.serialize import serialize_handlers
from parallel.client import CUDAPoolClient, GRPCClient
from parallel.utils import get_gpu_count

import multiprocessing
from training.dataset import Dataset
from rdkit import Chem

from timemachine.potentials import rmsd
from md import builders, minimizer
from rdkit.Chem import rdFMCS
from fe.atom_mapping import CompareDistNonterminal
from fe.utils import get_romol_conf

with open('ff/params/smirnoff_1_1_0_ccc.py') as f:
    ff_handlers = deserialize_handlers(f.read())

default_forcefield = Forcefield(ff_handlers)
import timemachine
from pathlib import Path

root = Path(timemachine.__file__).parent.parent
path_to_hif2a = root.joinpath('datasets/fep-benchmark/hif2a')

def get_ligands(ligand_sdf):
    suppl = Chem.SDMolSupplier(ligand_sdf, removeHs=False)
    mols = [x for x in suppl]
    return mols


class SolventConversion():
    def __init__(self, mol, mol_ref,
                 temperature=300, pressure=1.0, dt=2.5*1e-3,
                 num_equil_steps=100, num_prod_steps=1000,
                 num_windows=10, client=CUDAPoolClient(1),
                 initial_forcefield=default_forcefield):

        self.mol = mol
        self.mol_ref = mol_ref
        self.temperature = temperature
        self.pressure = pressure
        self.dt = dt
        self.num_equil_steps = num_equil_steps
        self.num_prod_steps = num_prod_steps
        self.schedule = construct_conversion_lambda_schedule(num_windows)
        self.client = client
        self.initial_forcefield = initial_forcefield
        self.solvent_system, self.solvent_coords, self.solvent_box, self.solvent_topology = builders.build_water_system(
            4.0)

        self.conversion_model = model_rabfe.AbsoluteConversionModel(
            self.client,
            self.initial_forcefield,
            self.solvent_system,
            self.schedule,
            self.solvent_topology,
            self.temperature,
            self.pressure,
            self.dt,
            self.num_equil_steps,
            self.num_prod_steps
        )

    def predict(self, flat_params):
        ordered_params = self.unflatten(flat_params)

        raise NotImplementedError


        # solvent
        min_solvent_coords = minimizer.minimize_host_4d([mol], solvent_system, solvent_coords, forcefield, solvent_box)
        solvent_x0 = np.concatenate([min_solvent_coords, mol_coords])
        solvent_box0 = solvent_box
        dG_solvent_conversion, dG_solvent_conversion_error = binding_model_solvent_conversion.predict(
            ordered_params,
            mol,
            solvent_x0,
            solvent_box0,
            prefix='solvent_conversion_' + mol_name + "_" + str(epoch)
        )

        return dG_solvent_conversion

def test_rabfe_conversion_trainable():
    """does not test for correctness, just that unconverged simulations"""

    ligand_sdf = str(path_to_hif2a.joinpath('ligands.sdf').resolve())
    mols = get_ligands(ligand_sdf)
    mol, mol_ref = mols[:2]


    solvent_conversion = SolventConversion(mol, mol_ref, solvent_system, solvent_topology)

    flat_params = solvent_conversion.flatten(default_forcefield.get_ordered_params())

    initial_prediction = solvent_conversion.predict()


    raise NotImplementedError
