import os
from pathlib import Path
import timemachine
from rdkit import Chem
import io
from contextlib import redirect_stdout
from fe.free_energy_rabfe import RABFEResult
import numpy as np
np.random.seed(0)

# TODO: running the example generates a lot of files in the working directory
#   and preparing to run the example generates a file in the slow_tests/ directory
#   --> use tempfiles / directories!

def render_command_line_arguments(cmd_arg_dict):
    """ "--{key}={value}" for each key-value pair in dictionary"""
    cmds = [f"--{key}={value}" for key, value in cmd_arg_dict.items()]
    return " ".join(cmds)


def test_validate_relative_binding():
    script_name = "examples/validate_relative_binding.py"
    n_windows = 10
    n_steps = 1001
    n_gpus = 2

    root = Path(timemachine.__file__).parent.parent
    path_to_hif2a = root.joinpath('datasets/fep-benchmark/hif2a')

    protein_pdb = str(path_to_hif2a.joinpath('5tbm_prepared.pdb').resolve())

    # TODO: use just 1 or 2 of these, rather than the whole file
    ligand_sdf = str(path_to_hif2a.joinpath('ligands.sdf').resolve())

    blocker_name = "338"  # arbitrary choice

    params = dict(
        property_field='"IC50[uM](SPA)"',
        property_units='"uM"',
        num_gpus=n_gpus,
        num_complex_conv_windows=n_windows,
        num_complex_windows=n_windows,
        num_solvent_conv_windows=n_windows,
        num_solvent_windows=n_windows,
        num_complex_equil_steps=n_steps,
        num_complex_prod_steps=n_steps,
        num_solvent_equil_steps=n_steps,
        num_solvent_prod_steps=n_steps,
        num_complex_preequil_steps=n_steps,
        blocker_name=blocker_name,
        protein_pdb=protein_pdb,
        ligand_sdf=ligand_sdf,
        num_replicates=1,
    )

    cmd_line_args = render_command_line_arguments(params)
    cmd = f"python {script_name} {cmd_line_args}"
    exit_status = os.system(cmd)
    assert exit_status == 0


def make_cache_log(ligand_sdf, name_property="_Name"):


    root = Path(timemachine.__file__).parent.parent
    path_to_tests = root.joinpath('slow_tests')
    cache_log = str(path_to_tests.joinpath('random_cache_log_for_testing.txt').resolve())

    suppl = Chem.SDMolSupplier(ligand_sdf, removeHs=False)
    mols = [x for x in suppl]

    rabfe_results = []
    for mol in mols:
        mol_name = mol.GetProp(name_property)
        rabfe_results.append(RABFEResult(mol_name, *np.random.randn(4)))

    f = io.StringIO()
    with redirect_stdout(f):
        for result in rabfe_results:
            result.log()
    log = f.getvalue()

    print(f'saving to {cache_log}...')
    with open(cache_log, 'w') as f:
        f.write(log)

    return cache_log


def test_train_relative_binding():
    script_name = "examples/train_relative_binding.py"
    n_windows = 10
    n_steps = 10001
    n_gpus = 2

    root = Path(timemachine.__file__).parent.parent
    path_to_hif2a = root.joinpath('datasets/fep-benchmark/hif2a')

    protein_pdb = str(path_to_hif2a.joinpath('5tbm_prepared.pdb').resolve())

    # TODO: use just 1 or 2 of these, rather than the whole file
    ligand_sdf = str(path_to_hif2a.joinpath('ligands.sdf').resolve())

    blocker_name = "338"  # arbitrary choice

    cache_log = make_cache_log(ligand_sdf)

    params = dict(
        property_field='"IC50[uM](SPA)"',
        property_units='"uM"',
        num_gpus=n_gpus,
        num_complex_conv_windows=n_windows,
        num_solvent_conv_windows=n_windows,
        num_complex_equil_steps=n_steps,
        num_complex_prod_steps=n_steps,
        num_solvent_equil_steps=n_steps,
        num_solvent_prod_steps=n_steps,
        num_complex_preequil_steps=n_steps,
        blocker_name=blocker_name,
        protein_pdb=protein_pdb,
        ligand_sdf=ligand_sdf,
        cache_log=cache_log,
        num_epochs=1,
    )

    cmd_line_args = render_command_line_arguments(params)
    cmd = f"python {script_name} {cmd_line_args}"
    exit_status = os.system(cmd)
    assert exit_status == 0
