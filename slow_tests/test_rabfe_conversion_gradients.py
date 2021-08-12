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

with open('ff/params/smirnoff_1_1_0_ccc.py') as f:
    ff_handlers = deserialize_handlers(f.read())

default_forcefield = Forcefield(ff_handlers)


class SolventConversion():
    def __init__(self, mol, mol_ref, solvent_system, solvent_topology,
                 temperature=300, pressure=1.0, dt=2.5*1e-3,
                 num_equil_steps=100, num_prod_steps=1000,
                 num_windows=10, client=CUDAPoolClient(1), forcefield=default_forcefield):

        self.mol = mol
        self.mol_ref = mol_ref

        self.schedule = construct_conversion_lambda_schedule(num_windows)

        conversion_model = model_rabfe.AbsoluteConversionModel(
            client,
            forcefield,
            solvent_system,
            self.schedule,
            solvent_topology,
            temperature,
            pressure,
            dt,
            num_equil_steps,
            num_prod_steps
        )
        self.conversion_model = conversion_model

        mcs_params = rdFMCS.MCSParameters()
        mcs_params.AtomTyper = CompareDistNonterminal()
        mcs_params.BondTyper = rdFMCS.BondCompare.CompareAny
        self.mcs_params = mcs_params

    def initialize_restraints(self, complex_coords, complex_ref_x0, complex_ref_box0):
        result = rdFMCS.FindMCS(
            [self.mol, self.mol_ref],
            self.mcs_params
        )

        core_smarts = result.smartsString

        print("core_smarts", core_smarts)

        # generate the core_idxs
        core_idxs = setup_relative_restraints_using_smarts(self.mol, self.mol_ref, core_smarts)
        mol_coords = get_romol_conf(self.mol)  # original coords

        num_complex_atoms = complex_coords.shape[0]

        # Use core_idxs to generate
        R, t = rmsd.get_optimal_rotation_and_translation(
            x1=complex_ref_x0[num_complex_atoms:][core_idxs[:, 1]],  # reference core atoms
            x2=mol_coords[core_idxs[:, 0]],  # mol core atoms
        )

        aligned_mol_coords = rmsd.apply_rotation_and_translation(mol_coords, R, t)

        ref_coords = complex_ref_x0[num_complex_atoms:]
        complex_host_coords = complex_ref_x0[:num_complex_atoms]
        complex_box0 = complex_ref_box0

        raise NotImplementedError




    def predict(self, flat_params):
        ordered_params = self.unflatten(flat_params)

        raise NotImplementedError


        # solvent
        min_solvent_coords = minimizer.minimize_host_4d([mol], solvent_system, solvent_coords, forcefield, solvent_box)
        solvent_x0 = np.concatenate([min_solvent_coords, mol_coords])
        solvent_box0 = solvent_box
        dG_solvent_conversion, dG_solvent_conversion_error = binding_model_solvent_conversion.predict(
            params,
            mol,
            solvent_x0,
            solvent_box0,
            prefix='solvent_conversion_' + mol_name + "_" + str(epoch)
        )

def test_rabfe_conversion_trainable():

    raise NotImplementedError
