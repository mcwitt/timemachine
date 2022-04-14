# Utility functions to help assign and identify local geometry points

from enum import Enum
from typing import List

from rdkit import Chem

SP = Chem.HybridizationType.SP
SP2 = Chem.HybridizationType.SP2
SP3 = Chem.HybridizationType.SP3


class LocalGeometry(Enum):
    G1_TERMINAL = 0  # R-X
    G2_KINK = 1  # R-X-H
    G2_LINEAR = 2  # R-X#N
    G3_PLANAR = 3  # R-X(=O)O
    G3_PYRAMIDAL = 4  # R-X(-H)H
    G4_TETRAHEDRAL = 5  # R-X(-H)(-H)H


def assign_atom_geometry(atom):
    """
    Heuristic using hybridization information to assign local description
    of geometry.
    """
    nbrs = len(list(atom.GetNeighbors()))
    hybr = atom.GetHybridization()

    two_neighbor_geometries = {
        SP3: LocalGeometry.G2_KINK,
        SP2: LocalGeometry.G2_KINK,
        SP: LocalGeometry.G2_LINEAR,
    }
    three_neighbor_geometries = {
        SP3: LocalGeometry.G3_PYRAMIDAL,
        SP2: LocalGeometry.G3_PLANAR,
    }
    four_neighbor_geometries = {
        SP3: LocalGeometry.G4_TETRAHEDRAL,
    }

    if nbrs == 0:
        assert 0, "Ion not supported"
    elif nbrs == 1:
        return LocalGeometry.G1_TERMINAL
    elif nbrs == 2:
        return two_neighbor_geometries[hybr]
    elif nbrs == 3:
        return three_neighbor_geometries[hybr]
    elif nbrs == 4:
        return four_neighbor_geometries[hybr]
    else:
        assert 0, "Too many neighbors"


def classify_geometry(mol: Chem.Mol) -> List[LocalGeometry]:
    """
    Identify the local geometry of the molecule. This current uses a heuristic but we
    should really be generating this from gas-phase simulations of the real forcefield.

    Currently, 3D coordinates are not required, but this may change in the future.

    Parameters
    ----------
    mol: Chem.Mol
        Input molecule.

    Returns
    -------
    List[LocalGeometry]
        List of per atom geometries

    """

    geometry_types = []
    for a in mol.GetAtoms():
        geometry_types.append(assign_atom_geometry(a))

    return geometry_types
