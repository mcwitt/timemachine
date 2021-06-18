# this script validates the endpoint correction protocol on relative hydration free energies

# This script repeatedly estimates the relative binding free energy of a single edge, along with the gradient of the
# estimate with respect to force field parameters, and adjusts the force field parameters to improve tha accuracy
# of the free energy prediction.


import argparse
import numpy as np
import jax
from jax import numpy as jnp

from fe.free_energy_rabfe import construct_absolute_lambda_schedule
from fe.utils import convert_uIC50_to_kJ_per_mole
from fe import model_rabfe
from md import builders

from testsystems.relative import hif2a_ligand_pair

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from ff.handlers.serialize import serialize_handlers
from ff.handlers.nonbonded import AM1CCCHandler, LennardJonesHandler
from parallel.client import CUDAPoolClient, GRPCClient
from parallel.utils import get_gpu_count

from typing import Union, Optional, Iterable, Any, Tuple, Dict

from optimize.step import truncated_step

import multiprocessing
from training.dataset import Dataset
from rdkit import Chem

array = Union[np.array, jnp.array]
Handler = Union[AM1CCCHandler, LennardJonesHandler] # TODO: do these all inherit from a Handler class already?

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(
        description="Absolute Binding Free Energy Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--hosts",
        nargs="*",
        default=None,
        help="Hosts running GRPC worker to use for compute"
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        help="number of gpus",
        default=get_gpu_count()
    )

    parser.add_argument(
        "--num_windows",
        type=int,
        help="number of solvent lambda windows",
        required=True
    )

    parser.add_argument(
        "--num_equil_steps",
        type=int,
        help="number of equilibration steps for each solvent lambda window",
        required=True
    )

    parser.add_argument(
        "--num_prod_steps",
        type=int,
        help="number of production steps for each solvent lambda window",
        required=True
    )

    cmd_args = parser.parse_args()

    if not cmd_args.hosts:
        num_gpus = cmd_args.num_gpus
        # set up multi-GPU client
        client = CUDAPoolClient(max_workers=num_gpus)
    else:
        # Setup GRPC client
        print("Connecting to GRPC workers...")
        client = GRPCClient(hosts=cmd_args.hosts)
    client.verify()

    path_to_ligand = 'tests/data/ligands_40.sdf'
    suppl = Chem.SDMolSupplier(path_to_ligand, removeHs=False)

    with open('ff/params/smirnoff_1_1_0_ccc.py') as f:
        ff_handlers = deserialize_handlers(f.read())

    forcefield = Forcefield(ff_handlers)
    mols = [x for x in suppl]

    dataset = Dataset(mols)

    absolute_solvent_schedule = construct_absolute_lambda_schedule(cmd_args.num_windows)
    relative_solvent_schedule = construct_absolute_lambda_schedule(cmd_args.num_windows-1)
    solvent_system, solvent_coords, solvent_box, solvent_topology = builders.build_water_system(4.0)

    # pick the largest mol as the blocker
    largest_size = 0
    ref_mol = None
    for mol in mols:
        if mol.GetNumAtoms() > largest_size:
            largest_size = mol.GetNumAtoms()
            ref_mol = mol

    print("Reference Molecule:", ref_mol.GetProp("_Name"), Chem.MolToSmiles(ref_mol))

    model_relative = model_rabfe.RelativeHydrationModel(
        client,
        forcefield,
        solvent_system,
        solvent_coords,
        solvent_box,
        relative_solvent_schedule,
        solvent_topology,
        cmd_args.num_equil_steps,
        cmd_args.num_prod_steps
    )

    # assert 0
    model_absolute = model_rabfe.AbsoluteHydrationModel(
        client,
        forcefield,
        solvent_system,
        solvent_coords,
        solvent_box,
        absolute_solvent_schedule,
        solvent_topology,
        cmd_args.num_equil_steps,
        cmd_args.num_prod_steps
    )

    ordered_params = forcefield.get_ordered_params()
    ordered_handles = forcefield.get_ordered_handles()

    M = len(dataset.data)

    for epoch in range(100):

        for i in range(M):

            for j in range(i+1, M):

                if i != 0 or j != 3:
                    continue

                mol_a = dataset.data[i]
                mol_b = dataset.data[j]

                ddG_ab = model_relative.predict(
                    ordered_params,
                    mol_a,
                    mol_b,
                    prefix='epoch_'+str(epoch)+'_solvent_relative_'+mol_a.GetProp('_Name')+'_'+mol_b.GetProp('_Name')
                )

                dG_a = model_absolute.predict(ordered_params, mol_a, prefix='solvent_absolute_'+mol_a.GetProp('_Name'))
                dG_b = model_absolute.predict(ordered_params, mol_b, prefix='solvent_absolute_'+mol_b.GetProp('_Name'))

                print("mol_i", i, mol_a.GetProp("_Name"), "mol_j", j, mol_b.GetProp("_Name"), "ddG_ab", ddG_ab, "dG_a-dG_b", dG_a-dG_b)
