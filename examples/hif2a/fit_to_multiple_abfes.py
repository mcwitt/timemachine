# This script repeatedly estimates the relative binding free energy of a single edge, along with the gradient of the
# estimate with respect to force field parameters, and adjusts the force field parameters to improve tha accuracy
# of the free energy prediction.


import argparse
import numpy as np
import jax
from jax import numpy as jnp

from fe.free_energy import construct_absolute_lambda_schedule
from fe.utils import convert_uIC50_to_kJ_per_mole
from fe import model_abfe
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

    # client = CUDAPoolClient(max_workers=cmd_args.num_gpus)

    if not cmd_args.hosts:
        num_gpus = cmd_args.num_gpus
        # set up multi-GPU client
        client = CUDAPoolClient(max_workers=num_gpus)
    else:
        # Setup GRPC client
        client = GRPCClient(hosts=cmd_args.hosts)
    client.verify()

    path_to_ligand = 'tests/data/ligands_40.sdf'
    suppl = Chem.SDMolSupplier(path_to_ligand, removeHs=False)

    with open('ff/params/smirnoff_1_1_0_ccc.py') as f:
        ff_handlers = deserialize_handlers(f.read())

    forcefield = Forcefield(ff_handlers)
    mols = [x for x in suppl]

    dataset = Dataset(mols)

    # construct lambda schedules for complex and solvent
    complex_schedule = construct_absolute_lambda_schedule(cmd_args.num_complex_windows)
    solvent_schedule = construct_absolute_lambda_schedule(cmd_args.num_solvent_windows)

    # build the protein system.
    complex_system, complex_coords, _, _, complex_box, complex_topology = builders.build_protein_system(
        'tests/data/hif2a_nowater_min.pdb')

    # build the water system.
    solvent_system, solvent_coords, solvent_box, solvent_topology = builders.build_water_system(4.0)

    client = None

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

    handle_types_being_optimized = [AM1CCCHandler, LennardJonesHandler]

    def flatten(params) -> Tuple[np.array, callable]:
        """Turn params dict into flat array, with an accompanying unflatten function

        TODO: note that the result is going to be in the order given by ordered_handles (filtered by presence in handle_types)
            rather than in the order they appear in handle_types_being_optimized

        TODO: maybe leave out the reference to handle_types_being optimized altogether

        TODO: does Jax have a pytree-based flatten / unflatten utility?
        """

        theta_list = []
        _shapes = dict()
        _handle_types = []

        for param, handle in zip(params, ordered_handles):
            assert handle.params.shape == param.shape
            key = type(handle)

            if key in handle_types_being_optimized:
                theta_list.append(param.flatten())
                _shapes[key] = param.shape
                _handle_types.append(key)

        theta = np.hstack(theta_list)

        def unflatten(theta: array) -> Dict[Handler, array]:
            params = dict()
            i = 0
            for key in _handle_types:
                shape = _shapes[key]
                num_params = int(np.prod(shape))
                params[key] = np.array(theta[i: i + num_params]).reshape(shape)
                i += num_params
            return params

        return theta, unflatten


    # in each optimization step, don't step so far that you think you're jumping to
    #   loss_next = relative_improvement_bound * loss_current
    relative_improvement_bound = 0.95

    flat_theta_traj = []
    flat_grad_traj = []
    loss_traj = []

    # arbitrary right now
    dG_reorg = 10

    def safe_repr(x):
        try:
            # get rid of ConcreteArray string shenanigans
            return x.aval.val
        except:
            return x

    def loss_fn(params, mol, label_dG_bind, epoch, cr, sr):
        dG_complex, cr = binding_model_complex.predict(params, mol, restraints=True, prefix='complex_'+str(epoch), cache_results=cr)
        # dG_solvent, sr = binding_model_solvent.predict(params, mol, restraints=False, prefix='solvent_'+str(epoch), cache_results=sr)
        # pred_dG_bind = dG_solvent - dG_complex + dG_reorg # deltaG of binding, move from solvent into complex

        # loss = jnp.abs(pred_dG_bind - label_dG_bind)
        # print("mol", mol.GetProp("_Name"), "dG_complex", safe_repr(dG_complex), "dG_solvent", safe_repr(dG_solvent), "dG_pred", safe_repr(pred_dG_bind), "dG_label", label_dG_bind)
        # return loss, (cr, sr)
        return dG_complex, (None, None)

    vg_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

    complex_results = None
    solvent_results = None

    for epoch in range(1000):
        epoch_params = serialize_handlers(ordered_handles)

        dataset.shuffle()

        for mol in dataset.data:

            if mol.GetProp("_Name") != '254':
                continue

            label_dG = convert_uIC50_to_kJ_per_mole(float(mol.GetProp("IC50[uM](SPA)")))

            print("processing mol", mol.GetProp("_Name"), "with binding dG", label_dG)

            (loss, (complex_results, solvent_results)), loss_grad = vg_fn(
                ordered_params,
                mol,
                label_dG,
                epoch,
                complex_results,
                solvent_results
            )

            print("epoch", epoch, "mol", mol.GetProp("_Name"), "loss", loss)

            continue

            # note: unflatten_grad and unflatten_theta have identical definitions for now
            flat_loss_grad, unflatten_grad = flatten(loss_grad)
            flat_theta, unflatten_theta = flatten(ordered_params)

            step_lower_bound = loss * relative_improvement_bound
            theta_increment = truncated_step(flat_theta, loss, flat_loss_grad, step_lower_bound=step_lower_bound)
            param_increments = unflatten_theta(theta_increment)

            # for any parameter handler types being updated, update in place
            for handle in ordered_handles:
                handle_type = type(handle)
                if handle_type in param_increments:
                    print(f'updating {handle_type.__name__}')

                    print(f'\tbefore update: {handle.params}')
                    handle.params += param_increments[handle_type] # TODO: careful -- this must be a "+=" or "-=" not an "="!
                    print(f'\tafter update:  {handle.params}')

                    # useful for debugging to dump out the grads
                    # for smirks, dp in zip(handle.smirks, loss_grad):
                        # if np.any(dp) > 0:
                            # print(smirks, dp)

            # checkpoint results to npz (overwrite
            flat_theta_traj.append(np.array(flat_theta))
            flat_grad_traj.append(flat_loss_grad)
            loss_traj.append(loss)

            path_to_npz = 'results_checkpoint.npz'
            print(f'saving theta, grad, loss trajs to {path_to_npz}')
            np.savez(
                path_to_npz,
                theta_traj=np.array(flat_theta_traj),
                grad_traj=np.array(flat_grad_traj),
                loss_traj=np.array(loss_traj)
            )

            # write ff parameters after each epoch
            path_to_ff_checkpoint = f"checkpoint_{epoch}.py"
            print(f'saving force field parameter checkpoint to {path_to_ff_checkpoint}')
            with open(path_to_ff_checkpoint, 'w') as fh:
                fh.write(epoch_params)
