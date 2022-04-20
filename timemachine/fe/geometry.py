# Utility functions to help assign and identify local geometry points

from enum import Enum
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import HybridizationType


class LocalGeometry(Enum):
    G0_ION = 0
    G1_TERMINAL = 1  # R-X
    G2_KINK = 2  # R-X-H
    G2_LINEAR = 3  # R-X#N
    G3_PLANAR = 4  # R-X(=O)O
    G3_PYRAMIDAL = 5  # R-X(-H)H
    G4_TETRAHEDRAL = 6  # R-X(-H)(-H)H


def assign_atom_geometry(atom, core):
    """
    Heuristic using hybridization information to assign local description
    of geometry. Atoms not in the core are not considered for assignment.
    """
    assert atom.GetIdx() in core

    real_nbrs = []
    for nbr_atom in atom.GetNeighbors():
        if nbr_atom.GetIdx() in core:
            real_nbrs.append(nbr_atom)

    num_nbrs = len(real_nbrs)
    hybridization = atom.GetHybridization()
    if num_nbrs == 0:
        assert 0, "Ion not supported"
    elif num_nbrs == 1:
        return LocalGeometry.G1_TERMINAL
    elif num_nbrs == 2:
        if hybridization == HybridizationType.SP3:
            return LocalGeometry.G2_KINK
        elif hybridization == HybridizationType.SP2:
            return LocalGeometry.G2_KINK
        elif hybridization == HybridizationType.SP:
            return LocalGeometry.G2_LINEAR
        else:
            assert 0, "Unknown 2-nbr geometry!"
    elif num_nbrs == 3:
        if hybridization == HybridizationType.SP3:
            return LocalGeometry.G3_PYRAMIDAL
        elif hybridization == HybridizationType.SP2:
            return LocalGeometry.G3_PLANAR
        else:
            assert 0, "Unknown 3-nbr geometry"
    elif num_nbrs == 4:
        if hybridization == HybridizationType.SP3:
            return LocalGeometry.G4_TETRAHEDRAL
        else:
            assert 0, "Unknown 4-nbr geometry"
    else:
        assert 0, "Too many neighbors"


def classify_geometry(mol: Chem.Mol, core: List[int] = None) -> List[LocalGeometry]:
    """
    Identify the local geometry of the molecule. This current uses a heuristic but we
    should really be generating this from gas-phase simulations of the real forcefield.

    Currently, 3D coordinates are not required, but this may change in the future.

    Parameters
    ----------
    mol: Chem.Mol
        Input molecule.

    core: List[Int] or None
        Core indices, if None then all atoms are considered to be in the core.

    Returns
    -------
    List[LocalGeometry]
        List of per atom geometries. Dummy atoms are None



    """

    if core is None:
        core = np.arange(mol.GetNumAtoms())

    geometry_types = []
    for a in mol.GetAtoms():
        if a.GetIdx() in core:
            geometry_types.append(assign_atom_geometry(a, core))
        else:
            geometry_types.append(None)

    return geometry_types
