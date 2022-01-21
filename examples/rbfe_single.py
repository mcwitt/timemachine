# This script estimates the relative binding free energy of a single edge


import argparse

from fe.free_energy import construct_lambda_schedule
from fe.utils import convert_uIC50_to_kJ_per_mole
from fe import model
from md import builders

from testsystems.relative import hif2a_ligand_pair

from parallel.client import CUDAPoolClient
from parallel.utils import get_gpu_count


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Relative Binding Free Energy Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--num_gpus", type=int, help="number of gpus", default=get_gpu_count())

    parser.add_argument("--num_complex_windows", type=int, help="number of vacuum lambda windows", required=True)

    parser.add_argument("--num_solvent_windows", type=int, help="number of solvent lambda windows", required=True)

    parser.add_argument(
        "--num_equil_steps", type=int, help="number of equilibration steps for each lambda window", required=True
    )

    parser.add_argument(
        "--num_prod_steps", type=int, help="number of production steps for each lambda window", required=True
    )

    cmd_args = parser.parse_args()

    client = CUDAPoolClient(max_workers=cmd_args.num_gpus)

    # fetch mol_a, mol_b, core, forcefield from testsystem
    mol_a, mol_b, core = hif2a_ligand_pair.mol_a, hif2a_ligand_pair.mol_b, hif2a_ligand_pair.core
    forcefield = hif2a_ligand_pair.ff

    # compute ddG label from mol_a, mol_b
    # (ytz): these are *binding* free energies, i.e. values that are less than zero.
    label_dG_a = convert_uIC50_to_kJ_per_mole(float(mol_a.GetProp("IC50[uM](SPA)")))
    label_dG_b = convert_uIC50_to_kJ_per_mole(float(mol_b.GetProp("IC50[uM](SPA)")))
    label_ddG = label_dG_b - label_dG_a  # complex - solvent

    print("binding dG_a", label_dG_a)
    print("binding dG_b", label_dG_b)

    hif2a_ligand_pair.label = label_ddG

    # construct lambda schedules for complex and solvent
    complex_schedule = construct_lambda_schedule(cmd_args.num_complex_windows)
    solvent_schedule = construct_lambda_schedule(cmd_args.num_solvent_windows)

    # build the protein system.
    complex_system, complex_coords, _, _, complex_box, _ = builders.build_protein_system(
        "tests/data/hif2a_nowater_min.pdb"
    )

    # build the water system.
    solvent_system, solvent_coords, solvent_box, _ = builders.build_water_system(4.0)

    binding_model = model.RBFEModel(
        client,
        forcefield,
        complex_system,
        complex_coords,
        complex_box,
        complex_schedule,
        solvent_system,
        solvent_coords,
        solvent_box,
        solvent_schedule,
        cmd_args.num_equil_steps,
        cmd_args.num_prod_steps,
    )

    ordered_params = forcefield.get_ordered_params()
    ordered_handles = forcefield.get_ordered_handles()

    ddg_predict = binding_model.predict(ordered_params, mol_a, mol_b, core)

    print(f'predicted ddg = {ddg_predict}')
    print(f'label ddg = {label_ddG}')
