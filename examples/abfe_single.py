# This script repeatedly estimates the relative binding free energy of a single edge, along with the gradient of the
# estimate with respect to force field parameters, and adjusts the force field parameters to improve tha accuracy
# of the free energy prediction.
import multiprocessing
import mdtraj

import argparse
import numpy as np
from simtk.openmm import app
import jax
from jax import numpy as jnp

from fe import pdb_writer
from fe.free_energy import construct_lambda_schedule
from fe.utils import convert_uIC50_to_kJ_per_mole
from fe import model_v2
from md import builders

from rdkit import Chem
from rdkit.Chem import rdmolops
from testsystems import relative

from ff.handlers.serialize import serialize_handlers
from ff.handlers.nonbonded import AM1CCCHandler, LennardJonesHandler
from parallel.client import CUDAPoolClient
from parallel.utils import get_gpu_count

from typing import Union, Optional, Iterable, Any, Tuple, Dict

from optimize.step import truncated_step

import tempfile
import scipy

array = Union[np.array, jnp.array]
Handler = Union[AM1CCCHandler, LennardJonesHandler] # TODO: do these all inherit from a Handler class already?

def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm

def generate_core(mol_a, mol_b):
    """
    Generate a core using non-complete bipartite graph matching.

    The smaller mol will be fully mapped.
    """
    xi = get_romol_conf(mol_a)
    xj = get_romol_conf(mol_b)
    xi = np.expand_dims(xi, axis=0)
    xj = np.expand_dims(xj, axis=1)
    xij = xi - xj
    # use square distance to penalize far apart pairs
    d2ij = np.sum(xij*xij, axis=-1)
    row_idxs, col_idxs = scipy.optimize.linear_sum_assignment(d2ij)
    core = np.stack([col_idxs, row_idxs], axis=-1)
    return core


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
                # read
                rd_mols.append(Chem.MolFromPDBFile(fp.name, removeHs=False))

        if isinstance(obj, Chem.Mol):
            rd_mols.append(obj)

    combined_mol = rd_mols[0]
    for mol in rd_mols[1:]:
        combined_mol = Chem.CombineMols(combined_mol, mol)

    # with tempfile.NamedTemporaryFile(mode='w') as fp:
    fp = open(out_filename, "w")
    # write
    Chem.MolToPDBFile(combined_mol, out_filename)
    fp.flush()
    # read
    combined_pdb = app.PDBFile(out_filename)
    return combined_pdb.topology



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

    mol, _, _, forcefield = relative._setup_hif2a_ligand_pair()

    core = []
    for a_idx, a in enumerate(mol.GetAtoms()):
        if a.IsInRing():
            core.append(a_idx)

    core = np.array(core, dtype=np.int32)

    print("initial core", core)

    # compute ddG label from mol_a, mol_b
    # TODO: add label upon testsystem construction
    # (ytz): these are *binding* free energies, i.e. values that are less than zero.
    label_dG = convert_uIC50_to_kJ_per_mole(float(mol.GetProp("IC50[uM](SPA)")))
    # label_dG_b = convert_uIC50_to_kJ_per_mole(float(mol_b.GetProp("IC50[uM](SPA)")))
    # label_ddG = label_dG_b - label_dG_a  # complex - solvent
    print("binding dG", label_dG)
    # print("binding dG_b", label_dG_b)

    # hif2a_ligand_pair.label = label_ddG

    # construct lambda schedules for complex and solvent
    complex1_schedule, complex0_schedule = construct_lambda_schedule(cmd_args.num_complex_windows)
    solvent_schedule, _ = construct_lambda_schedule(cmd_args.num_solvent_windows)

    # build the protein system.
    complex_system, complex_coords, _, _, complex_box, complex_topology = builders.build_protein_system(
        'tests/data/hif2a_nowater_min.pdb')
    complex_box += np.eye(3) * 0.1  # BFGS this later

    # build the water system.
    solvent_system, solvent_coords, solvent_box, solvent_topology = builders.build_water_system(4.0)
    solvent_box += np.eye(3) * 0.1  # BFGS this later

    complex_topology = generate_topology([complex_topology, mol], complex_coords, "complex.pdb")
    solvent_topology = generate_topology([solvent_topology, mol], solvent_coords, "solvent.pdb")

    k_core = 50.0

    print("k_core", k_core)
    # k_core = 10000.0

    binding_model = model_v2.ABFEModel(
        client,
        forcefield,
        complex_system,
        complex_coords,
        complex_box,
        complex0_schedule,
        complex1_schedule,
        complex_topology,
        solvent_system,
        solvent_coords,
        solvent_box,
        solvent_schedule,
        solvent_topology,
        cmd_args.num_equil_steps,
        cmd_args.num_prod_steps,
        k_core
    )

    ordered_params = forcefield.get_ordered_params()
    ordered_handles = forcefield.get_ordered_handles()

    for epoch in range(100):
        epoch_params = serialize_handlers(ordered_handles)

        pred, aux = binding_model.predict(ordered_params, mol, core, epoch)

        print(pred)

        continue

        # print("epoch", epoch, "stage", stage, "dG", dG)

        for (stage, results), lambda_schedule, topology in zip(aux, [complex_schedule, solvent_schedule], [complex_topology, solvent_topology]):
            avg_du_dls = []

            combined_lambda_schedule = np.concatenate([
                np.array([1000.0]),
                lambda_schedule[::-1],
                lambda_schedule,
                np.array([1000.0]),
            ])

            for lamb_idx, (lamb, sim_res) in enumerate(zip(combined_lambda_schedule[1:-1], results[1:-1])):

                md_topology = mdtraj.Topology.from_openmm(topology)
                traj = mdtraj.Trajectory(sim_res.xs, md_topology)
                traj.save_xtc(stage+"_lambda_"+str(lamb_idx)+".xtc")

                print(stage, "lambda", lamb, "<du/dl>", np.mean(sim_res.du_dls), "std(du/dl)", np.std(sim_res.du_dls))
                avg_du_dls.append(np.mean(sim_res.du_dls))

            dG = np.trapz(avg_du_dls, combined_lambda_schedule[1:-1])

            print("epoch", epoch, "stage", stage, "dG", dG)

        continue

        print("epoch", epoch, "loss", loss)

        assert 0

        # note: unflatten_grad and unflatten_theta have identical definitions for now
        flat_loss_grad, unflatten_grad = flatten(loss_grad)
        flat_theta, unflatten_theta = flatten(ordered_params)

        step_lower_bound = loss * relative_improvement_bound
        theta_increment = truncated_step(flat_theta, loss, flat_loss_grad, step_lower_bound=step_lower_bound)
        param_increments= unflatten_theta(theta_increment)

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
