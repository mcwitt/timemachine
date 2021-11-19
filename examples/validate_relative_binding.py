# This script runs the rabfe code in "inference" mode.
# There are multiple stages to this protocol:

# For the complex setup, we proceed as follows:
# 1) Conversion of the ligand parameters into a ff-independent state
# 2) Adding restraints to the ligand to the non-interacting "blocker" molecule
# 3) Compute the free energy of the swapping the ligand with the blocker
# 4) Release the restraints attached to the blocker
# Note that 2) and 4) are done directly via an endpoint correction.

# For the solvent setup, we proceed as follows:
# 1) Conversion of the ligand parameters into a ff-independent state.
# 2) Run an absolute hydration free energy of the ff-independent state.
import os
import pickle
import argparse
import numpy as np

from pathlib import Path

from fe.free_energy_rabfe import (
    construct_absolute_lambda_schedule_complex,
    construct_absolute_lambda_schedule_solvent,
    construct_conversion_lambda_schedule,
    get_romol_conf,
    setup_relative_restraints_by_distance,
    RABFEResult,
)
from fe.utils import convert_uM_to_kJ_per_mole
from fe import model_rabfe
from fe.frames import endpoint_frames_only, all_frames

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from parallel.client import CUDAPoolClient, GRPCClient
from parallel.utils import get_gpu_count


from training.dataset import Dataset
from rdkit import Chem

import timemachine
from timemachine.potentials import rmsd
from md import builders, minimizer

ALL_FRAMES = "all"
ENDPOINTS_ONLY = "endpoints"


def cache_wrapper(cache_path: str, fxn: callable, overwrite: bool = False) -> callable:
    """Given a path and a function, will either write the result of the function call
    to the file or read form the file. The output of the function is expected to be picklable.

    Parameters
    ----------
    cache_path: string
        Path to write pickled results to and to read results from


    fxn: callable
        Function whose result to cache. Used for expensive functions whose results
        should be stored between runs.

    overwrite: boolean
        Overwrite the cache if the file already exists. Useful is the cache file was incorrectly
        generated or the implementation of fxn has changed.

    Returns
    -------

    wrapped_function: callable
        A function with the same signature as the original fxn whose results will be stored to
        the cached file.
    """

    def fn(*args, **kwargs):
        if overwrite or not os.path.isfile(cache_path):
            print(f"Caching {fxn.__name__} result to {cache_path}")
            val = fxn(*args, **kwargs)
            with open(cache_path, "wb") as ofs:
                pickle.dump(val, ofs)
        else:
            with open(cache_path, "rb") as ifs:
                val = pickle.load(ifs)
        return val

    return fn


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Relatively absolute Binding Free Energy Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--epochs", default=1, help="Number of Epochs", type=int)

    parser.add_argument(
        "--property_field",
        help="Property field to convert to kcals/mols",
        default=None,
    )

    parser.add_argument(
        "--property_units",
        help="must be either nM or uM",
        required=True,
        choices=["nM", "uM"],
    )

    parser.add_argument("--hosts", nargs="*", default=None, help="Hosts running GRPC worker to use for compute")

    parser.add_argument("--num_gpus", type=int, help="number of gpus", default=get_gpu_count())

    parser.add_argument(
        "--num_complex_conv_windows", type=int, help="number of lambda windows for complex conversion", required=True
    )

    parser.add_argument("--num_complex_windows", type=int, help="number of vacuum lambda windows", required=True)

    parser.add_argument(
        "--num_solvent_conv_windows", type=int, help="number of lambda windows for solvent conversion", required=True
    )

    parser.add_argument("--num_solvent_windows", type=int, help="number of solvent lambda windows", required=True)

    parser.add_argument(
        "--num_complex_equil_steps",
        type=int,
        help="number of equilibration steps for each complex lambda window",
        required=True,
    )

    parser.add_argument(
        "--num_complex_prod_steps",
        type=int,
        help="number of production steps for each complex lambda window",
        required=True,
    )

    parser.add_argument(
        "--num_solvent_equil_steps",
        type=int,
        help="number of equilibration steps for each solvent lambda window",
        required=True,
    )

    parser.add_argument(
        "--num_solvent_prod_steps",
        type=int,
        help="number of production steps for each solvent lambda window",
        required=True,
    )

    parser.add_argument(
        "--num_complex_preequil_steps",
        type=int,
        help="number of pre-equilibration steps for each complex lambda window",
        required=True,
    )

    parser.add_argument(
        "--blocker_name", type=str, help="Name of the ligand the sdf file to be used as a blocker", required=True
    )
    parser.add_argument("--frames_written", choices=[ALL_FRAMES, ENDPOINTS_ONLY], default=ENDPOINTS_ONLY)

    parser.add_argument("--protein_pdb", type=str, help="Path to the target pdb", required=True)

    parser.add_argument("--ligand_sdf", type=str, help="Path to the ligand's sdf", required=True)

    parser.add_argument("--shuffle", action="store_true", help="Shuffle ligand order")

    cmd_args = parser.parse_args()

    print("cmd_args", cmd_args)

    if not cmd_args.hosts:
        num_gpus = cmd_args.num_gpus
        # set up multi-GPU client
        client = CUDAPoolClient(max_workers=num_gpus)
    else:
        # Setup GRPC client
        print("Connecting to GRPC workers...")
        client = GRPCClient(hosts=cmd_args.hosts)
    client.verify()

    path_to_ligand = cmd_args.ligand_sdf
    suppl = Chem.SDMolSupplier(path_to_ligand, removeHs=False)
    root = Path(timemachine.__file__).parent.parent

    with open(root.joinpath("ff/params/smirnoff_1_1_0_ccc.py")) as f:
        ff_handlers = deserialize_handlers(f.read())

    forcefield = Forcefield(ff_handlers)
    mols = [x for x in suppl]

    dataset = Dataset(mols)

    # construct lambda schedules for complex and solvent
    complex_absolute_schedule = construct_absolute_lambda_schedule_complex(cmd_args.num_complex_windows)
    solvent_absolute_schedule = construct_absolute_lambda_schedule_solvent(cmd_args.num_solvent_windows)

    # build the protein system.
    complex_system, complex_coords, _, _, complex_box, complex_topology = builders.build_protein_system(
        cmd_args.protein_pdb
    )

    solvent_system, solvent_coords, solvent_box, solvent_topology = builders.build_water_system(4.0)

    blocker_mol = None

    for mol in mols:
        if mol.GetProp("_Name") == cmd_args.blocker_name:
            # we should only have one copy.
            assert blocker_mol is None
            blocker_mol = mol

    assert blocker_mol is not None

    frame_filter = None
    if cmd_args.frames_written == ALL_FRAMES:
        frame_filter = all_frames
    elif cmd_args.frames_written == ENDPOINTS_ONLY:
        frame_filter = endpoint_frames_only
    assert frame_filter is not None, f"Unknown frame writing mode: {cmd_args.frames_written}"

    print("Reference Molecule:", blocker_mol.GetProp("_Name"), Chem.MolToSmiles(blocker_mol))

    temperature = 300.0
    pressure = 1.0
    dt = 2.5e-3

    # cached_equil_path = "equil_{}_blocker_{}.pkl".format(
    #     os.path.basename(cmd_args.protein_pdb).split(".")[0], cmd_args.blocker_name
    # ).replace(" ", "_")

    # # Generate an equilibrated reference structure to use.
    # complex_ref_x0, complex_ref_box0 = cache_wrapper(cached_equil_path, minimizer.equilibrate_complex)(
    #     blocker_mol,
    #     complex_system,
    #     complex_coords,
    #     temperature,
    #     pressure,
    #     forcefield,
    #     complex_box,
    #     cmd_args.num_complex_preequil_steps,
    # )

    # complex_conversion_schedule = construct_conversion_lambda_schedule(cmd_args.num_complex_conv_windows)
    # complex models
    # binding_model_complex_conversion = model_rabfe.AbsoluteConversionModel(
    #     client,
    #     forcefield,
    #     complex_system,
    #     complex_conversion_schedule,
    #     complex_topology,
    #     temperature,
    #     pressure,
    #     dt,
    #     cmd_args.num_complex_equil_steps,
    #     cmd_args.num_complex_prod_steps,
    #     frame_filter=frame_filter,
    # )

    # binding_model_complex_decouple = model_rabfe.RelativeBindingModel(
    #     client,
    #     forcefield,
    #     complex_system,
    #     complex_absolute_schedule,
    #     complex_topology,
    #     temperature,
    #     pressure,
    #     dt,
    #     cmd_args.num_complex_equil_steps,
    #     cmd_args.num_complex_prod_steps,
    #     frame_filter=frame_filter,
    # )

    # solvent models.
    # solvent_conversion_schedule = construct_conversion_lambda_schedule(cmd_args.num_solvent_conv_windows)

    # binding_model_solvent_conversion = model_rabfe.AbsoluteConversionModel(
    #     client,
    #     forcefield,
    #     solvent_system,
    #     solvent_conversion_schedule,
    #     solvent_topology,
    #     temperature,
    #     pressure,
    #     dt,
    #     cmd_args.num_solvent_equil_steps,
    #     cmd_args.num_solvent_prod_steps,

    ref_vals = [0.0,
        0.030602248958530254,
         0.04343921065458984,
         0.053286962874056605,
         0.061530500238056327,
         0.06874727143589167,
         0.07523159584747947,
         0.08115977734926053,
         0.08673559270006116,
         0.09197355858262861,
         0.0969144153961157,
         0.10159645025291364,
         0.10605939310314867,
         0.1103340040687108,
         0.11444283570500403,
         0.11840380773941236,
         0.12222621867887551,
         0.12590621489743328,
         0.12947504831763063,
         0.13299387815576555,
         0.13636145441131187,
         0.13957203783801678,
         0.1427408444583851,
         0.14592107849903435,
         0.1489925598194029,
         0.15200062172473455,
         0.15505279334028343,
         0.15792573401056345,
         0.1607395469926778,
         0.16356099493445256,
         0.16620056446808032,
         0.16897275173092222,
         0.17159610748988727,
         0.17417292092268263,
         0.17674146103119037,
         0.17914299078288928,
         0.18164789797849124,
         0.18391826664913077,
         0.18637732136770624,
         0.1885821065845263,
         0.191022715593469,
         0.19319584264582332,
         0.1955766612624234,
         0.19767308202503384,
         0.1999739720926449,
         0.20198698137575316,
         0.20420273396213723,
         0.20622434396706107,
         0.20842794933641354,
         0.21052336987945022,
         0.21274670771096021,
         0.2149248581260875,
         0.2172515796738345,
         0.2195808848777371,
         0.22211414715154548,
         0.22457724420555747,
         0.22743460652624772,
         0.23036511964282108,
         0.23359480722524123,
         0.23744927805741772,
         0.2419763171071928,
         0.2476993102185118,
         0.2559914311487634,
         0.27228673430098493,
         0.31186658687684155,
         0.36654268499663656,
         0.4878873054204437,
         0.7378873054201813,
         0.9878873054190915,
         1.0]
    ref_vals = np.asarray(ref_vals)
    def custom_solvent_decouple_sched(num_wins: int):
        return np.interp(np.linspace(0, 1, num_wins), np.linspace(0, 1, len(ref_vals)), ref_vals)



    solvent_absolute_schedule = custom_solvent_decouple_sched(64)
    #     frame_filter=frame_filter,
    # )

    binding_model_solvent_decouple = model_rabfe.AbsoluteStandardHydrationModel(
        client,
        forcefield,
        solvent_system,
        solvent_absolute_schedule,
        solvent_topology,
        temperature,
        pressure,
        dt,
        cmd_args.num_solvent_equil_steps,
        cmd_args.num_solvent_prod_steps,
        frame_filter=frame_filter,
    )

    ordered_params = forcefield.get_ordered_params()
    ordered_handles = forcefield.get_ordered_handles()
    # print("Foobar", solvent_absolute_schedule.shape)
    print(solvent_absolute_schedule)
    # import matplotlib.pyplot as plt
    # plt.plot(construct_absolute_lambda_schedule_solvent(len(ref_vals)), ".-", label="Ref")
    # plt.plot(ref_vals, label="A")
    # plt.plot(solvent_absolute_schedule, label="B")
    # plt.legend()
    # plt.show()
    # raise ValueError("Foo")

    def simulate_pair(epoch: int, blocker: Chem.Mol, mol: Chem.Mol):
        mol_name = mol.GetProp("_Name")

        # generate the core_idxs
        # core_idxs = setup_relative_restraints_by_distance(mol, blocker)
        mol_coords = get_romol_conf(mol)  # original coords

        # num_complex_atoms = complex_coords.shape[0]

        # Use core_idxs to generate
        # R, t = rmsd.get_optimal_rotation_and_translation(
        #     x1=complex_ref_x0[num_complex_atoms:][core_idxs[:, 1]],  # reference core atoms
        #     x2=mol_coords[core_idxs[:, 0]],  # mol core atoms
        # )

        # aligned_mol_coords = rmsd.apply_rotation_and_translation(mol_coords, R, t)

        # ref_coords = complex_ref_x0[num_complex_atoms:]
        # complex_host_coords = complex_ref_x0[:num_complex_atoms]
        # complex_box0 = complex_ref_box0

        # compute the free energy of swapping an interacting mol with a non-interacting reference mol
        # complex_decouple_x0 = minimizer.minimize_host_4d(
        #     [mol, blocker_mol],
        #     complex_system,
        #     complex_host_coords,
        #     forcefield,
        #     complex_box0,
        #     [aligned_mol_coords, ref_coords],
        # )
        # complex_decouple_x0 = np.concatenate([complex_decouple_x0, aligned_mol_coords, ref_coords])

        # # compute the free energy of conversion in complex
        # complex_conversion_x0 = minimizer.minimize_host_4d(
        #     [mol],
        #     complex_system,
        #     complex_host_coords,
        #     forcefield,
        #     complex_box0,
        #     [aligned_mol_coords],
        # )
        # complex_conversion_x0 = np.concatenate([complex_conversion_x0, aligned_mol_coords])

        min_solvent_coords = minimizer.minimize_host_4d([mol], solvent_system, solvent_coords, forcefield, solvent_box)
        solvent_x0 = np.concatenate([min_solvent_coords, mol_coords])
        solvent_box0 = solvent_box

        suffix = f"{mol_name}_{epoch}"

        return {
            "solvent_decouple": binding_model_solvent_decouple.simulate_futures(
                ordered_params,
                mol,
                solvent_x0,
                solvent_box0,
                prefix="solvent_decouple_" + suffix,
            ),
            # "solvent_conversion": binding_model_solvent_conversion.simulate_futures(
            #     ordered_params,
            #     mol,
            #     solvent_x0,
            #     solvent_box0,
            #     prefix="solvent_conversion_" + suffix,
            # ),
            # "complex_conversion": binding_model_complex_conversion.simulate_futures(
            #     ordered_params,
            #     mol,
            #     complex_conversion_x0,
            #     complex_box0,
            #     prefix="complex_conversion_" + suffix,
            # ),
            # "complex_decouple": binding_model_complex_decouple.simulate_futures(
            #     ordered_params,
            #     mol,
            #     blocker_mol,
            #     core_idxs,
            #     complex_decouple_x0,
            #     complex_box0,
            #     prefix="complex_decouple_" + suffix,
            # ),
            "mol": mol,
            "blocker": blocker,
            "epoch": epoch,
        }

    def predict_dG(results: dict):
        # dG_complex_decouple, dG_complex_decouple_error = binding_model_complex_decouple.predict_from_futures(
        #     results["complex_decouple"][0],
        #     results["mol"],
        #     results["blocker"],
        #     results["complex_decouple"][1],
        #     results["complex_decouple"][2],
        # )
        # dG_complex_conversion, dG_complex_conversion_error = binding_model_complex_conversion.predict_from_futures(
        #     results["complex_conversion"][0],
        #     results["mol"],
        #     results["complex_conversion"][1],
        #     results["complex_conversion"][2],
        # )
        dG_solvent_decouple, dG_solvent_decouple_error = binding_model_solvent_decouple.predict_from_futures(
            results["solvent_decouple"][0],
            results["mol"],
            results["solvent_decouple"][1],
            results["solvent_decouple"][2],
        )

        # dG_solvent_conversion, dG_solvent_conversion_error = binding_model_solvent_conversion.predict_from_futures(
        #     results["solvent_conversion"][0],
        #     results["mol"],
        #     results["solvent_conversion"][1],
        #     results["solvent_conversion"][2],
        # )

        return dG_solvent_decouple, dG_solvent_decouple_error

        rabfe_result = RABFEResult(
            mol_name=mol_name,
            dG_complex_conversion=dG_complex_conversion,
            dG_complex_decouple=dG_complex_decouple,
            dG_solvent_conversion=dG_solvent_conversion,
            dG_solvent_decouple=dG_solvent_decouple,
        )
        rabfe_result.log()
        dG_err = np.sqrt(
            dG_complex_conversion_error ** 2
            + dG_complex_decouple_error ** 2
            + dG_solvent_conversion_error ** 2
            + dG_solvent_decouple_error ** 2
        )

        return rabfe_result.dG_bind, dG_err

    runs = []
    for epoch in range(cmd_args.epochs):
        if cmd_args.shuffle:
            dataset.shuffle()
        for mol in dataset.data:
            try:
                runs.append(simulate_pair(epoch, blocker_mol, mol))
            except Exception as e:
                mol_name = mol.GetProp("_Name")
                print(f"Error simulating Mol: {mol_name}: {e}")
    for i in range(len(runs)):
        # Pop off futures to avoid accumulating memory.
        run = runs.pop(0)
        mol = run["mol"]
        epoch = run["epoch"]
        mol_name = mol.GetProp("_Name")

        label_dG = "'N/A'"
        if cmd_args.property_field is not None:
            try:
                concentration = float(mol.GetProp(cmd_args.property_field))

                if cmd_args.property_units == "uM":
                    label_dG = convert_uM_to_kJ_per_mole(concentration)
                elif cmd_args.property_units == "nM":
                    label_dG = convert_uM_to_kJ_per_mole(concentration / 1000)
                else:
                    assert 0, "Unknown property units"
            except Exception as e:
                print(f"Unable to find property {cmd_args.property_field}: {e}")
        print(f"Epoch: {epoch}, Processing Mol: {mol_name}, Label: {label_dG}")
        try:
            dG, dG_err = predict_dG(run)
            print(f"Epoch: {epoch}, Mol: {mol_name}, Predicted dG: {dG}, dG Err:" f" {dG_err}, Label: {label_dG}")
        except Exception as e:
            print(f"Error processing Mol: {mol_name}: {e}")
