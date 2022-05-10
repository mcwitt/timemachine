# Utility functions to help assign and identify local geometry points

from enum import Enum
from typing import List
from timemachine.ff import Forcefield

import numpy as np
from rdkit import Chem
from rdkit.Chem import HybridizationType
from timemachine.fe import topology

class LocalGeometry(Enum):
    G0_ION = 0
    G1_TERMINAL = 1  # R-X
    G2_KINK = 2  # R-X-H
    G2_LINEAR = 3  # R-X#N
    G3_PLANAR = 4  # R-X(=O)O
    G3_PYRAMIDAL = 5  # R-X(-H)H
    G4_TETRAHEDRAL = 6  # R-X(-H)(-H)H

def bond_idxs_to_nblist(bond_idxs):
    num_atoms = np.amax(bond_idxs) + 1
    cmat = np.zeros((num_atoms, num_atoms))
    for i,j in bond_idxs:
        cmat[i][j] = 1
        cmat[j][i] = 1

    nblist = []
    for i in range(num_atoms):
        nbs = []
        for j in range(num_atoms):
            if cmat[i][j]:
                nbs.append(j)
        nblist.append(nbs)

    return nblist

def label_stereo(
    bond_idxs,
    bond_params,
    angle_idxs,
    angle_params,
    proper_idxs,
    proper_params,
    improper_idxs,
    improper_params):
    # stereo atom
    # 4 neighbors - always stereo
    # 3 neighbors - if improper torsions are present

    # stereo bond 
    # src, dst atoms must have 2 or 3 neighbors
    # a planarizing torsion is present (period=2, phase=3.1415, k > 10kJ/mol)

    # list of list representation
    nblist = bond_idxs_to_nblist(bond_idxs)

    # do atoms
    atom_stereo_flags = []
    atom_geometries = []
    for atom_idx, atom_nbs in enumerate(nblist):
        if len(atom_nbs) == 4:
            atom_stereo_flags.append(True)
            atom_geometries.append(LocalGeometry.G4_TETRAHEDRAL)
        elif len(atom_nbs) == 3:
            # check for impropers
            is_stereo = True
            local_geometry = LocalGeometry.G3_PYRAMIDAL

            # impropers are centered around the first atom.
            for i,_,_,_ in improper_idxs:
                if i == atom_idx:
                    is_stereo = False
                    print("PLANAR GROUP FOUND", atom_idx)
                    local_geometry = LocalGeometry.G3_PLANAR
                    break

            atom_geometries.append(local_geometry)
            atom_stereo_flags.append(is_stereo)
        elif len(atom_nbs) == 2:
            # check angle terms:
            local_geometry = LocalGeometry.G2_KINK
            for (i,j,k), (_,angle) in zip(angle_idxs, angle_params):
                if abs(angle) < 0.05:
                    assert 0
                ii, kk = atom_nbs[0], atom_nbs[1]
                if j == atom_idx:
                    # if j == 10:

                    if (i,k) == (ii,kk) or (i,k) == (kk,ii):
                        if abs(angle-np.pi) < 0.05:
                            local_geometry = LocalGeometry.G2_LINEAR
                            break
            atom_geometries.append(local_geometry)
            atom_stereo_flags.append(False)
        elif len(atom_nbs) == 1:
            atom_geometries.append(LocalGeometry.G1_TERMINAL)
            atom_stereo_flags.append(False)
    
    # do bonds
    bond_stereo_flags = []
    for i,j in bond_idxs:
        i_nbs = nblist[i]
        j_nbs = nblist[j]
        if (len(i_nbs) == 2 or len(i_nbs) == 3) and (len(j_nbs) == 2 or len(j_nbs) == 3):
            # check for proper torsions
            is_stereo = False
            for (ii,jj,kk,ll), (k, phase, period) in zip(proper_idxs, proper_params):
                if (jj,kk) == (i,j) or (jj,kk) == (j,i):
                    if period == 2 and (phase - np.pi) < 0.05:
                        is_stereo = True
                        break
            
            bond_stereo_flags.append(is_stereo)
        else:
            bond_stereo_flags.append(False)

    return atom_geometries, np.array(atom_stereo_flags, dtype=int), np.array(bond_stereo_flags, dtype=int)



def classify_geometry(mol: Chem.Mol, ff:Forcefield = None) -> List[LocalGeometry]:
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
    if ff is None:
        ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    bt = topology.BaseTopology(mol, ff)

    bond_params, hb = bt.parameterize_harmonic_bond(ff.hb_handle.params)
    angle_params, ha = bt.parameterize_harmonic_angle(ff.ha_handle.params)
    proper_params, pt = bt.parameterize_proper_torsion(ff.pt_handle.params)
    improper_params, it = bt.parameterize_improper_torsion(ff.pt_handle.params)

    bond_idxs = hb.get_idxs()
    angle_idxs = ha.get_idxs()
    proper_idxs = pt.get_idxs()
    improper_idxs = it.get_idxs()

    atom_geometries, atom_stereo, bond_stereo = label_stereo(
        bond_idxs, bond_params, angle_idxs, angle_params, proper_idxs, proper_params, improper_idxs, improper_params
    )

    return atom_geometries

    # if core is None:
    #     core = np.arange(mol.GetNumAtoms())

    # geometry_types = []
    # for a in mol.GetAtoms():
    #     if a.GetIdx() in core:
    #         geometry_types.append(assign_atom_geometry(a, core))
    #     else:
    #         geometry_types.append(None)

    # return geometry_types
