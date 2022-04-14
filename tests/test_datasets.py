from timemachine.datasets import fetch_freesolv


def test_fetch_freesolv():
    """assert expected number of molecules loaded -- with unique names and expected property annotations"""
    mol_dict = fetch_freesolv()
    mols = list(mol_dict.values())

    # expected number of mols loaded
    assert len(mols) == 642

    # expected mol properties present, interpretable as floats
    for mol in mols:
        props = mol.GetPropsAsDict()
        _, _ = float(props["dG"]), float(props["dG_err"])

    # unique names
    names = [mol.GetProp("_Name") for mol in mols]
    assert len(set(names)) == len(names)

    # dictionary keys consistent with _Name property
    for name, mol in mol_dict.items():
        assert name == mol.GetProp("_Name")
