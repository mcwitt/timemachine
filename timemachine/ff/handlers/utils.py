import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem


def canonicalize_bond(arr):
    """
    Canonicalize a bonded interaction. If arr[0] < arr[-1] then arr is
    returned, else if arr[0] > arr[-1], then arr[::-1] is returned. If
    arr[0] == arr[-1] then an exception is thrown.

    Parameters
    ----------
    arr: list of int
        Bond indices.

    Returns
    -------
    arr: list of int
        Canonicalized bond indices.

    """
    container_type = type(arr)

    if len(arr) == 0:
        raise ValueError("zero sized array")
    elif len(arr) == 1:
        return arr
    elif arr[0] > arr[-1]:
        return container_type(reversed(arr))
    elif arr[0] == arr[-1]:
        raise ValueError("Invalid bond with first and last indices equal")
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
    qmol = Chem.MolFromSmarts(smirks)  # cannot catch the error
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


def get_symmetry_classes(mol: Chem.Mol) -> NDArray:
    """Assign arbitrary integer indices to atoms in mol, such that:
    * symmetric atoms are assigned distinct indices
    * indices are contiguous from 0 to n_classes
    """
    symmetry_classes = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True))
    n_classes = len(set(symmetry_classes))

    # make contiguous from 0 to n_classes
    idx_map = {old_idx: new_idx for (new_idx, old_idx) in enumerate(set(symmetry_classes))}
    symmetry_classes = [idx_map[old_idx] for old_idx in symmetry_classes]
    assert set(symmetry_classes) == set(range(n_classes))

    return np.array(symmetry_classes)


def symmetrize(per_particle_params: NDArray, mol: Chem.Mol) -> NDArray:
    """Replace per_particle_params[atom] with mean([per_particle_params[a] for a in symmetry_class(atom)])"""

    # compute the mean per symmetry class
    symmetry_classes = get_symmetry_classes(mol)
    class_sums = jax.ops.segment_sum(per_particle_params, symmetry_classes)
    class_means = class_sums / np.bincount(symmetry_classes)

    # assign all members of a symmetry class the class's mean value
    symmetrized = class_means[symmetry_classes]
    assert symmetrized.shape == per_particle_params.shape

    return symmetrized
