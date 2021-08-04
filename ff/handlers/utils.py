import numpy as np

import timemachine
from pathlib import Path
from ff import Forcefield

from rdkit import Chem
from ff.handlers import nonbonded, AM1CCCHandler
from ff.handlers.deserialize import deserialize_handlers


def sort_tuple(arr):

    container_type = type(arr)

    if len(arr) == 0:
        raise Exception("zero sized array")
    elif len(arr) == 1:
        return arr
    elif arr[0] > arr[-1]:
        return container_type(reversed(arr))
    else:
        return arr

def match_smirks(mol, smirks):
    """
    Notes
    -----
    * See also implementations of match_smirks in
        * bootstrap_am1.py, which is identical
        * bcc_aromaticity.py, which uses OpenEye instead of RDKit
    """
    
    # Make a copy of the molecule
    rdmol = Chem.Mol(mol)
    # Use designated aromaticity model
    Chem.SanitizeMol(rdmol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY)
    Chem.SetAromaticity(rdmol, Chem.AromaticityModel.AROMATICITY_MDL)
    
    # Set up query.
    qmol = Chem.MolFromSmarts(smirks)  #cannot catch the error
    if qmol is None:
        raise ValueError('RDKit could not parse the SMIRKS string "{}"'.format(smirks))

    # Create atom mapping for query molecule
    idx_map = dict()
    for atom in qmol.GetAtoms():
        smirks_index = atom.GetAtomMapNum()
        if smirks_index != 0:
            idx_map[smirks_index - 1] = atom.GetIdx()
    map_list = [idx_map[x] for x in sorted(idx_map)]

    # Perform matching
    matches = list()
    for match in rdmol.GetSubstructMatches(qmol, uniquify=False):
        mas = [match[x] for x in map_list]
        matches.append(tuple(mas))

    return matches


def get_default_am1ccc_handler():
    # load ff
    root = Path(timemachine.__file__).parent.parent
    path_to_ff = str(root.joinpath('ff/params/smirnoff_1_1_0_ccc.py'))

    with open(path_to_ff) as f:
        ff_handlers = deserialize_handlers(f.read())
    ff = Forcefield(ff_handlers)

    # get AM1CCC handler
    ordered_handles = ff.get_ordered_handles()
    components = [handle.__class__ for handle in ordered_handles]
    handles = dict(zip(components, ordered_handles))
    am1ccc_handler = handles[AM1CCCHandler]

    return am1ccc_handler


def assert_am1ccc_am1bcc_consistency(mols, abs_tolerance=1e-3):
    """Assert that the partial charges assigned by ff/params/smirnoff_1_1_0_ccc.py
    are close to those assigned by AM1BCCELF10, for all atoms in a collection of mols"""
    # methods to compare
    am1ccc_handler = get_default_am1ccc_handler()

    def am1ccc_parameterize(mol):
        return am1ccc_handler.parameterize(mol)

    def am1bcc_parameterize(mol):
        return nonbonded.oe_assign_charges(mol, 'AM1BCCELF10')

    # run both methods on all mols
    inlined_constant = np.sqrt(timemachine.constants.ONE_4PI_EPS0)

    for mol in mols:
        ref = am1bcc_parameterize(mol) / inlined_constant
        test = am1ccc_parameterize(mol) / inlined_constant
        difference = np.max(np.abs(ref - test))
        assert difference < abs_tolerance, f'{difference:.3f} e > {abs_tolerance} e'
