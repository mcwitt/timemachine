from timemachine.datasets import fetch_freesolv
from timemachine.md.enhanced import identify_rotatable_bonds


def test_identify_rotatable_bonds_runs_on_freesolv():
    """pass if no runtime errors are encountered"""
    mol_dict = fetch_freesolv()
    mols = list(mol_dict.values())

    for mol in mols:
        _ = identify_rotatable_bonds(mol)
