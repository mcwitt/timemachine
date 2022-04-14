from importlib import resources
from typing import Dict, TypeAlias

from rdkit.Chem import Mol, SDMolSupplier

MolName: TypeAlias = str


def fetch_freesolv() -> Dict[MolName, Mol]:
    with resources.path("timemachine.datasets.freesolv", "freesolv.sdf") as freesolv_path:
        supplier = SDMolSupplier(str(freesolv_path), removeHs=False)

    mol_dict = {mol.GetProp("_Name"): mol for mol in supplier}
    return mol_dict
