# This script repeatedly estimates the relative binding free energy of a single edge, along with the gradient of the
# estimate with respect to force field parameters, and adjusts the force field parameters to improve tha accuracy
# of the free energy prediction.

from rdkit import Chem
import argparse
import numpy as np
import jax
from jax import numpy as jnp

from fe.free_energy import construct_absolute_lambda_schedule
from fe.utils import convert_uIC50_to_kJ_per_mole
from fe import model_abfe
from md import builders

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from ff.handlers.serialize import serialize_handlers
from ff.handlers.nonbonded import AM1CCCHandler, LennardJonesHandler
from parallel.client import CUDAPoolClient
from parallel.utils import get_gpu_count

from typing import Union, Optional, Iterable, Any, Tuple, Dict

from optimize.step import truncated_step

import multiprocessing

array = Union[np.array, jnp.array]
Handler = Union[AM1CCCHandler, LennardJonesHandler] # TODO: do these all inherit from a Handler class already?

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(
        description="Relative Binding Free Energy Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        help="number of gpus",
        default=get_gpu_count()
    )

    parser.add_argument(
        "--num_complex_windows",
        type=int,
        help="number of vacuum lambda windows",
        required=True
    )

    parser.add_argument(
        "--num_solvent_windows",
        type=int,
        help="number of solvent lambda windows",
        required=True
    )

    parser.add_argument(
        "--num_equil_steps",
        type=int,
        help="number of equilibration steps for each lambda window",
        required=True
    )

    parser.add_argument(
        "--num_prod_steps",
        type=int,
        help="number of production steps for each lambda window",
        required=True
    )

    cmd_args = parser.parse_args()

    client = CUDAPoolClient(max_workers=cmd_args.num_gpus)

    path_to_ligand = 'tests/data/ligands_40.sdf'
    suppl = Chem.SDMolSupplier(path_to_ligand, removeHs=False)

    with open('ff/params/smirnoff_1_1_0_ccc.py') as f:
        ff_handlers = deserialize_handlers(f.read())

    forcefield = Forcefield(ff_handlers)

    for mol in suppl:

        label_dG = convert_uIC50_to_kJ_per_mole(float(mol.GetProp("IC50[uM](SPA)")))

        # if mol.GetProp("_Name") != "234":
            # continue

        print("mol", mol.GetProp("_Name"), "binding dG", label_dG)

        # construct lambda schedules for complex and solvent
        complex_schedule = construct_absolute_lambda_schedule(cmd_args.num_complex_windows)
        solvent_schedule = construct_absolute_lambda_schedule(cmd_args.num_solvent_windows)

        # build the protein system.
        complex_system, complex_coords, _, _, complex_box, complex_topology = builders.build_protein_system(
            'tests/data/hif2a_nowater_min.pdb')

        # build the water system.
        solvent_system, solvent_coords, solvent_box, solvent_topology = builders.build_water_system(4.0)

        # client = None

        binding_model_complex = model_abfe.AbsoluteModel(
            client,
            forcefield,
            complex_system,
            complex_coords,
            complex_box,
            complex_schedule,
            complex_topology,
            cmd_args.num_equil_steps,
            cmd_args.num_prod_steps
        )

        binding_model_solvent = model_abfe.AbsoluteModel(
            client,
            forcefield,
            solvent_system,
            solvent_coords,
            solvent_box,
            solvent_schedule,
            solvent_topology,
            cmd_args.num_equil_steps,
            cmd_args.num_prod_steps
        )

        ordered_params = forcefield.get_ordered_params()
        ordered_handles = forcefield.get_ordered_handles()

        def loss_fn(params, mol, label_dG_bind, epoch, cr, sr):
            dG_complex, cr = binding_model_complex.predict(params, mol, restraints=True, prefix='complex_'+str(epoch), cache_results=cr)
            dG_solvent, sr = binding_model_solvent.predict(params, mol, restraints=False, prefix='solvent_'+str(epoch), cache_results=sr)
            pred_dG_bind = dG_solvent - dG_complex # deltaG of binding, move from solvent into complex

            loss = jnp.abs(pred_dG_bind - label_dG_bind)
            print("dG_complex", dG_complex, "dG_solvent", dG_solvent, "dG_pred", pred_dG_bind, "dG_label", label_dG_bind)
            return loss, (cr, sr)
            # return dG_complex, (None, None)

        complex_results = None
        solvent_results = None

        epoch = 0

        loss, (complex_results, solvent_results) = loss_fn(
            ordered_params,
            mol,
            label_dG,
            epoch,
            complex_results,
            solvent_results
        )

        print("mol", mol.GetProp("_Name"), "loss", loss)
