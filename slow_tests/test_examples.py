import os
from pathlib import Path
import timemachine


def render_command_line_arguments(cmd_arg_dict):
    """ "--{key}={value}" for each key-value pair in dictionary"""
    cmds = [f"--{key}={value}" for key, value in cmd_arg_dict.items()]
    return " ".join(cmds)


def test_validate_relative_binding():
    script_name = "examples/validate_relative_binding.py"
    n_windows = 50
    n_steps = 100
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
    )

    cmd_line_args = render_command_line_arguments(params)
    cmd = f"python {script_name} {cmd_line_args}"
    exit_status = os.system(cmd)
    assert exit_status == 0
