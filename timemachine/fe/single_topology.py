from collections import defaultdict
from collections.abc import Iterable
from enum import Enum

import numpy as np

from timemachine.fe import dummy, geometry, topology, utils
from timemachine.fe.geometry import LocalGeometry


def is_planarizing(force_constant, phase, period):
    """
    Determine if
    """
    return period == 2 and abs(phase - np.pi) < 0.05 and force_constant > 10.0


def identify_bonds_spanned_by_planar_torsions(proper_idxs, proper_params):
    """
    Identify bonds that are spanned by planar torsions and returns a dict of bonds
    and associated torsions that span it.
    """
    planar_bonds = dict()

    # collect unique idxs first
    torsions = defaultdict(list)
    for idxs, params in zip(proper_idxs, proper_params):
        idxs = tuple(idxs.tolist())
        torsions[idxs].append(params)

    for (i, j, k, l), all_params in torsions.items():
        # planarizing torsions should be unique
        if len(all_params) == 1:
            force, phase, period = all_params[0]
            if is_planarizing(force, phase, period):
                canon_jk = dummy.canonicalize_bond((j, k))
                if canon_jk not in planar_bonds:
                    planar_bonds[canon_jk] = []
                planar_bonds[canon_jk].append((i, j, k, l))

    return planar_bonds


def recursive_map(items, mapping):
    """recursively replace items in a list of list
    mapping = np.arange(100)[::-1]
    items = [[0,2,3], [5,1,[2,5,6]], 3]
    result = recursive_map(items, mapping)

    """
    if isinstance(items, Iterable):
        res = []
        for item in items:
            res.append(recursive_map(item, mapping))
        return tuple(res)
    else:
        return mapping[items]


def find_chiral_bonds(ring_bonds, proper_idxs, proper_params, mol):
    """
    Find chiral bonds - note that a chiral bond is a local descriptor. i.e. only the neighboring
    atoms to the bond are considered. Furthermore, all atoms are considered unique, even if they are hydrogens!

    (ytz): we can simplify the API to just bond_idxs, and infer ring atoms and degree from the
    connectivity alone without resorting to rdkit.

    Parameters
    ----------
    ring_bonds: list of 2-tuple
        Bonds that are part of a ring system.

    proper_idxs: list of 4-tuple
        Proper torsion idxs

    proper_params: list of 3-tuple
        Proper torsion parameters

    mol: Chem.Mol
        RDKit molecule

    Returns
    -------
    set
        Set of chiral bonds.

    """
    # a stereo bond is defined as a bond that
    # 1) has a single proper torsion term with k > 10 kJ/mol, period=2, phase=3.1415
    # 2) is not part of a ring system.
    # the reason why 2) is present is because the planar torsions spanning a
    # ring system are not used to enforce stereochemistry, since if we simply
    # disabled them, we would *still* get the correct stereochemistry due to steric effects.
    # consider a benzene devoid of any torsions (proper or improper), or non-bonded
    # terms, and only angles and bonds are present. The hydrogens would still be correctly placed.

    canonical_ring_bonds = set()
    for ij in ring_bonds:
        canonical_ring_bonds.add(dummy.canonicalize_bond(ij))

    planar_bonds_kv = identify_bonds_spanned_by_planar_torsions(proper_idxs, proper_params)
    canonical_chiral_bonds = set()
    for k in planar_bonds_kv.keys():
        if k not in canonical_ring_bonds:
            src, dst = k
            src = mol.GetAtomWithIdx(src)
            dst = mol.GetAtomWithIdx(dst)
            if src.GetDegree() == 4 or dst.GetDegree() == 4:
                # if either atom is tetrahedral then it has no pi bonds to conjugate over
                # exception: weird sulfur groups
                continue
            else:
                canonical_chiral_bonds.add(k)

    return canonical_chiral_bonds


def find_chiral_atoms(mol):
    """
    Find chiral atoms in a molecule. Chirality is perceived locally, without taking
    graph-symmetry into account. Eg. even something like CH4 would be considered
    to be chiral since the hydrogens are distinguishable.

    Returns
    -------
    set
        Set of chiral atoms.

    """
    mol_geom = geometry.classify_geometry(mol)
    chiral_atoms = set()
    for a in mol.GetAtoms():
        a_idx = a.GetIdx()
        nbs = a.GetNeighbors()
        if len(nbs) == 4:
            chiral_atoms.add(a_idx)
        elif len(nbs) == 3 and mol_geom[a_idx] == LocalGeometry.G3_PYRAMIDAL:
            # if in ring, or is sulfur or phosphorus, this may not be guaranteed by
            if a.IsInRing() or a.GetAtomicNum() == 16 or a.GetAtomicNum() == 15:
                chiral_atoms.add(a_idx)
        elif len(nbs) > 4:
            assert 0, "More than four neighbors found"
    return chiral_atoms


def check_bond_stability(j, k, bond_idxs, bond_params):
    # tbd this should really be based on the pdf, we should expect to see
    # near zero probability at bond lengths < 0.5 Angstroms
    canonical_bonds = dict()
    for idxs, params in zip(bond_idxs, bond_params):
        canonical_bonds[dummy.canonicalize_bond(idxs)] = params

    jk = dummy.canonicalize_bond((j, k))

    if jk not in canonical_bonds:
        return False
    elif canonical_bonds[jk][0] < 50.0:
        return False
    elif canonical_bonds[jk][1] < 0.05:
        return False

    return True


def check_angle_stability(j, k, l, angle_idxs, angle_params):

    canonical_angles = dict()
    for idxs, params in zip(angle_idxs, angle_params):
        k, t = params
        assert k > 0.0
        assert t >= 0.0
        assert t <= np.pi
        canonical_angles[dummy.canonicalize_bond(idxs)] = params

    jkl = dummy.canonicalize_bond((j, k, l))

    if jkl not in canonical_angles:
        return False
    elif canonical_angles[jkl][0] < 50.0:
        return False
    elif abs(canonical_angles[jkl][1] - 3.1415) < 0.05:
        return False
    elif abs(canonical_angles[jkl][1] - 0) < 0.05:
        assert 0, "angle term of zero rad found."

    return True


def check_bond_angle_stability(j, k, l, bond_idxs, bond_params, angle_idxs, angle_params):
    jk_stable = check_bond_stability(j, k, bond_idxs, bond_params)
    kl_stable = check_bond_stability(k, l, bond_idxs, bond_params)
    jkl_stable = check_angle_stability(j, k, l, angle_idxs, angle_params)
    return jk_stable and kl_stable and jkl_stable


def find_attached_dummy_atoms(dg, bond_idxs, anchor):
    attached_dummy_atoms = []
    for a, b in bond_idxs:
        if a == anchor and b in dg:
            attached_dummy_atoms.append(b)
        elif b == anchor and a in dg:
            attached_dummy_atoms.append(a)
    return attached_dummy_atoms


def find_junction_bonds(anchor, bond_idxs):
    jbs = set()
    for i, j in bond_idxs:
        assert i != j
        if i == anchor or j == anchor:
            jbs.add(dummy.canonicalize_bond((i, j)))
    return jbs


def enumerate_anchor_groups(anchor_idx, bond_idxs, core_idxs):

    # enumerate all 1 and 2 neighbor anchor atoms to form valid anchor groups.

    assert anchor_idx in core_idxs
    assert anchor_idx in [x[0] for x in bond_idxs] or anchor_idx in [x[1] for x in bond_idxs]

    nbs_1 = set()
    for src, dst in bond_idxs:
        if src == anchor_idx and dst in core_idxs:
            nbs_1.add(dst)
        elif dst == anchor_idx and src in core_idxs:
            nbs_1.add(src)

    nbs_2 = set()  # ordered tuple!
    for atom in nbs_1:
        for src, dst in bond_idxs:
            if src == atom and dst in core_idxs and dst != anchor_idx:
                nbs_2.add((atom, dst))
            elif dst == atom and src in core_idxs and src != anchor_idx:
                nbs_2.add((atom, src))

    return nbs_1, nbs_2


def setup_dummy_interactions(ff, mol_a, mol_b, core, dummy_group, anchor):
    """
    Add restraints between dummy atoms in a dummy_group and core atoms. Interactions involving
    mol_b's dummy atoms are added in a factorizable, and numerically stable way relative to both
    mol_b and mol_a.

    Parameters
    ----------

    mol_a: Chem.Mol
        mol_a's parameters are mainly used to check for numerical stability.

    mol_b: Chem.Mol
        Source of the dummy atoms

    core: np.array(N,2)
        Core atoms

    dummy_group: set
        A set of dummy atoms tethered to an anchor

    Returns
    -------
    (idxs, params)
        A set of bond, angle, proper, improper, c_angle, and x_angle restraints

    """
    core_b_to_a = {b: a for a, b in core}

    assert len(dummy_group) > 0
    assert anchor in core_b_to_a, "anchor must be in core"

    dummy_group = list(dummy_group)
    for d in dummy_group:
        assert d not in core_b_to_a, "dummy " + str(d) + " must not be in core"

    # these idxs/params can and should be cached, but is repeated here to keep the api simple
    mol_a_top = topology.BaseTopology(mol_a, ff)
    mol_b_top = topology.BaseTopology(mol_b, ff)

    mol_a_bond_params, mol_a_hb = mol_a_top.parameterize_harmonic_bond(ff.hb_handle.params)
    mol_a_angle_params, mol_a_ha = mol_a_top.parameterize_harmonic_angle(ff.ha_handle.params)

    mol_b_bond_params, mol_b_hb = mol_b_top.parameterize_harmonic_bond(ff.hb_handle.params)
    mol_b_angle_params, mol_b_ha = mol_b_top.parameterize_harmonic_angle(ff.ha_handle.params)
    mol_b_proper_params, mol_b_pt = mol_b_top.parameterize_proper_torsion(ff.pt_handle.params)
    mol_b_improper_params, mol_b_it = mol_b_top.parameterize_improper_torsion(ff.it_handle.params)

    mol_a_bond_idxs = mol_a_hb.get_idxs()
    mol_a_angle_idxs = mol_a_ha.get_idxs()

    mol_b_bond_idxs = mol_b_hb.get_idxs()
    mol_b_angle_idxs = mol_b_ha.get_idxs()
    mol_b_proper_idxs = mol_b_pt.get_idxs()
    mol_b_improper_idxs = mol_b_it.get_idxs()

    mol_b_core = core_b_to_a.keys()
    mol_b_full_geometry = geometry.classify_geometry(mol_b)
    mol_b_core_geometry = geometry.classify_geometry(mol_b, core=mol_b_core)

    # pick an arbitrary atom in the dummy_group and find the anchors, there may be
    # multiple anchors, eg (d=dummy, c=core):
    #   d...d
    #   |   |
    #   c---c
    root_anchors = dummy.identify_root_anchors(mol_b_bond_idxs, mol_b_core, dummy_group[0])
    assert anchor in root_anchors
    # (ytz): we can relax this assertion later on.
    assert len(root_anchors) == 1, "multiple root anchors found."

    mol_b_ring_bonds = []
    for b in mol_b.GetBonds():
        if b.IsInRing():
            mol_b_ring_bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))

    stereo_bonds = find_chiral_bonds(mol_b_ring_bonds, mol_b_proper_idxs, mol_b_proper_params, mol_b)
    stereo_atoms = find_chiral_atoms(mol_b)

    junction_bonds = find_junction_bonds(anchor, mol_b_bond_idxs)
    # (ytz): the mapping code should hopefully be able to guarantee this
    assert junction_bonds.isdisjoint(stereo_bonds)

    # generic forcefield restraints
    restraint_bond_idxs = []
    restraint_bond_params = []

    restraint_angle_idxs = []
    restraint_angle_params = []

    restraint_proper_idxs = []
    restraint_proper_params = []

    restraint_improper_idxs = []
    restraint_improper_params = []

    # copy restraints that involve only anchor and dummy atoms. These restraints are always
    # numerically stable and factorizable. In order to enhance sampling, dummy atoms are designed
    # to sort of mimic the "enhanced" state. Where by:
    # 1) bond, angle, and improper torsions are copied over.
    # 2) stereo bonds are copied over, but generic rotatable bonds are not.
    # 3) dummy-dummy nonbonded interactions are fully disabled.
    dummy_groupa = dummy_group + [anchor]
    for idxs, params in zip(mol_b_bond_idxs, mol_b_bond_params):
        if np.all([a in dummy_groupa for a in idxs]):
            restraint_bond_idxs.append(tuple([int(x) for x in idxs]))  # tuples are hashable etc.
            restraint_bond_params.append(params)
    for idxs, params in zip(mol_b_angle_idxs, mol_b_angle_params):
        if np.all([a in dummy_groupa for a in idxs]):
            restraint_angle_idxs.append(tuple([int(x) for x in idxs]))
            restraint_angle_params.append(params)
    for idxs, params in zip(mol_b_proper_idxs, mol_b_proper_params):
        _, jj, kk, _ = idxs
        # only copy over proper torsions responsible for stereo bonds
        if np.all([a in dummy_groupa for a in idxs]) and (dummy.canonicalize_bond((jj, kk)) in stereo_bonds):
            restraint_proper_idxs.append(tuple([int(x) for x in idxs]))
            restraint_proper_params.append(params)
    for idxs, params in zip(mol_b_improper_idxs, mol_b_improper_params):
        if np.all([a in dummy_groupa for a in idxs]):
            restraint_improper_idxs.append(tuple([int(x) for x in idxs]))
            restraint_improper_params.append(params)

    anchor_core_geometry = mol_b_core_geometry[anchor]
    anchor_full_geometry = mol_b_full_geometry[anchor]

    # specialized restraints that are factorizable
    restraint_cross_angle_idxs = []
    restraint_cross_angle_params = []

    restraint_centroid_angle_idxs = []
    restraint_centroid_angle_params = []

    nbs_1, _ = enumerate_anchor_groups(anchor, mol_b_bond_idxs, mol_b_core)

    if anchor_core_geometry == LocalGeometry.G1_TERMINAL:
        # type i-j, i and j are core atoms, j being the anchor
        assert len(nbs_1) == 1
        i = list(nbs_1)[0]  # core atom next to anchor
        j = anchor

        # require that the core bond i,j is stable in both mol_a and mol_b
        # note: there is no recovery here, so assert when this fails.
        assert check_bond_stability(i, j, mol_b_bond_idxs, mol_b_bond_params)
        assert check_bond_stability(core_b_to_a[i], core_b_to_a[j], mol_a_bond_idxs, mol_a_bond_params)

        if anchor_full_geometry == LocalGeometry.G2_KINK:
            # diagram:
            #     k
            #    .
            # i-j
            # add one angle between (i,j,k)
            atoms = find_attached_dummy_atoms(dummy_group, mol_b_bond_idxs, anchor)
            assert len(atoms) == 1
            k = atoms[0]
            # add one angle
            restraint_angle_idxs.append((i, j, k))
            restraint_angle_params.append((100.0, (2.0 / 3.0) * np.pi))
        elif anchor_full_geometry == LocalGeometry.G2_LINEAR:
            # diagram:
            # i-j.k
            # add one angle between (i,j,k)
            atoms = find_attached_dummy_atoms(dummy_group, mol_b_bond_idxs, anchor)
            assert len(atoms) == 1
            k = atoms[0]
            restraint_angle_idxs.append((i, j, k))
            restraint_angle_params.append((100.0, np.pi))
        elif anchor_full_geometry == LocalGeometry.G3_PYRAMIDAL:
            # diagram:
            # i-j.k0
            #    .
            #     k1
            # add two angles: (i,j,k0) and (i,j,k1)
            atoms = find_attached_dummy_atoms(dummy_group, mol_b_bond_idxs, anchor)
            assert len(atoms) == 2
            k0, k1 = atoms
            if anchor in stereo_atoms:
                restraint_angle_idxs.append((i, j, k0))
                restraint_angle_params.append((100.0, 1.91))  # 109 degrees
                restraint_angle_idxs.append((i, j, k1))
                restraint_angle_params.append((100.0, 1.91))  # 109 degrees
            else:
                restraint_angle_idxs.append((i, j, k0))
                restraint_angle_params.append((100.0, (2.0 / 3.0) * np.pi))
                restraint_angle_idxs.append((i, j, k1))
                restraint_angle_params.append((100.0, (2.0 / 3.0) * np.pi))
        elif anchor_full_geometry == LocalGeometry.G3_PLANAR:
            # diagram:
            #     k0
            #    .
            # i-j
            #    .
            #     k1
            # add two angles: (i,j,k0) and (i,j,k1)
            # typically we'd have to worry about stereochemistry, but we're
            # pretty confident here we don't have any stereo issues.
            atoms = find_attached_dummy_atoms(dummy_group, mol_b_bond_idxs, anchor)
            assert len(atoms) == 2
            k0, k1 = atoms
            restraint_angle_idxs.append((i, j, k0))
            restraint_angle_params.append((100.0, (2.0 / 3.0) * np.pi))
            restraint_angle_idxs.append((i, j, k1))
            restraint_angle_params.append((100.0, (2.0 / 3.0) * np.pi))
        elif anchor_full_geometry == LocalGeometry.G4_TETRAHEDRAL:
            # diagram:
            #     k0
            #    .
            # i-j . k2
            #    .
            #     k1
            # add three angles: (i,j,k0) and (i,j,k1) and (i,j,k2)
            assert anchor in stereo_atoms
            atoms = find_attached_dummy_atoms(dummy_group, mol_b_bond_idxs, anchor)
            assert len(atoms) == 3
            k0, k1, k2 = atoms
            restraint_angle_idxs.append((i, j, k0))
            restraint_angle_params.append((100.0, 1.91))
            restraint_angle_idxs.append((i, j, k1))
            restraint_angle_params.append((100.0, 1.91))
            restraint_angle_idxs.append((i, j, k2))
            restraint_angle_params.append((100.0, 1.91))
        else:
            assert 0, "Illegal Geometry"
    elif anchor_core_geometry == LocalGeometry.G2_KINK:
        # type a
        #       \
        #        j
        #       /
        #      b
        # a and b are core atoms, j being the anchor

        j = anchor

        if anchor_full_geometry == LocalGeometry.G3_PYRAMIDAL:
            # type
            #    a - j
            #       / .
            #      b   k
            # find stable core atoms next to anchor that we can build angle restraints off of.
            a, b = nbs_1

            assert check_bond_angle_stability(
                a, j, b, mol_b_bond_idxs, mol_b_bond_params, mol_b_angle_idxs, mol_b_angle_params
            )
            assert check_bond_angle_stability(
                core_b_to_a[a],
                core_b_to_a[j],
                core_b_to_a[b],
                mol_a_bond_idxs,
                mol_a_bond_params,
                mol_a_angle_idxs,
                mol_a_angle_params,
            )

            # a and b are core atoms, j being the anchor
            atoms = find_attached_dummy_atoms(dummy_group, mol_b_bond_idxs, anchor)
            assert len(atoms) == 1
            k = atoms[0]

            if anchor in stereo_atoms:
                # two options
                # 1) two wells at zero and pi
                # 2) one well explicitly encoding the stereo - probably better!
                # implement 1 for now?
                restraint_cross_angle_idxs.append(((j, a), (j, b), (j, k)))
                restraint_cross_angle_params.append(1000.0)
            else:
                # planarize so we can enhance sample both stereoisomers using a centroid
                # type a
                #       \
                #      c.j.k <- angle (c,j,k) = 0.0
                #       /
                #      b
                restraint_centroid_angle_idxs.append((tuple(sorted((a, b))), j, k))
                restraint_centroid_angle_params.append((5000.0, np.pi))

        elif anchor_full_geometry == LocalGeometry.G3_PLANAR:
            # same as G3_PYRAMIDAL non-stereo
            assert anchor not in stereo_atoms
            a, b = nbs_1
            assert check_bond_angle_stability(
                a, j, b, mol_b_bond_idxs, mol_b_bond_params, mol_b_angle_idxs, mol_b_angle_params
            )
            assert check_bond_angle_stability(
                core_b_to_a[a],
                core_b_to_a[j],
                core_b_to_a[b],
                mol_a_bond_idxs,
                mol_a_bond_params,
                mol_a_angle_idxs,
                mol_a_angle_params,
            )
            # a and b are core atoms, j being the anchor
            atoms = find_attached_dummy_atoms(dummy_group, mol_b_bond_idxs, anchor)
            assert len(atoms) == 1
            k = atoms[0]
            restraint_centroid_angle_idxs.append((tuple(sorted((a, b))), j, k))
            restraint_centroid_angle_params.append((5000.0, np.pi))

        elif anchor_full_geometry == LocalGeometry.G4_TETRAHEDRAL:
            #            l
            #           .
            # type a - j . k
            #           \
            #            b
            a, b = nbs_1
            # core-atoms a,j,b
            assert check_bond_angle_stability(
                a, j, b, mol_b_bond_idxs, mol_b_bond_params, mol_b_angle_idxs, mol_b_angle_params
            )

            assert check_bond_angle_stability(
                core_b_to_a[a],
                core_b_to_a[j],
                core_b_to_a[b],
                mol_a_bond_idxs,
                mol_a_bond_params,
                mol_a_angle_idxs,
                mol_a_angle_params,
            )

            atoms = find_attached_dummy_atoms(dummy_group, mol_b_bond_idxs, anchor)
            assert len(atoms) == 2
            k, l = atoms
            # dummy-atoms l,j,k
            assert check_bond_angle_stability(
                l, j, k, mol_b_bond_idxs, mol_b_bond_params, mol_b_angle_idxs, mol_b_angle_params
            )
            restraint_cross_angle_idxs.append(((j, a), (j, b), (j, k)))
            restraint_cross_angle_params.append(1000.0)
            restraint_cross_angle_idxs.append(((j, a), (j, b), (j, l)))
            restraint_cross_angle_params.append(1000.0)
        else:
            assert 0, "Illegal Geometry"
    elif anchor_core_geometry == LocalGeometry.G2_LINEAR:
        assert 0, "Illegal Geometry"
    elif anchor_core_geometry == LocalGeometry.G3_PLANAR:
        assert 0, "Illegal Geometry"
    elif anchor_core_geometry == LocalGeometry.G3_PYRAMIDAL:
        j = anchor
        if anchor_full_geometry == LocalGeometry.G4_TETRAHEDRAL:
            #            c
            #           /
            # type a - j . k
            #           \
            #            b
            #
            # we have some choices here:
            # 1) if there is no ring-opening and closing, then we use a centroid angle restraint defined by [a,b,c],j,k
            # 2) if there is ring-opening and closing, then we need to enumerate possible cross-product based restraints.
            # it turns out that 1) is quite difficult, since rapidly converting pyramidal groups (eg. ammonium) do not have
            # reasonable centroids due to rapid conversion. So we should always do 2) as it's more robust and enables
            # more kinds of transformations.

            a, b, c = nbs_1

            # enumerate all possible anchors, break when we find a stable one
            x, y = None, None
            for xx, yy in [(a, b), (a, c), (b, c)]:

                b_okay = check_bond_angle_stability(
                    xx, j, yy, mol_b_bond_idxs, mol_b_bond_params, mol_b_angle_idxs, mol_b_angle_params
                )
                a_okay = check_bond_angle_stability(
                    core_b_to_a[xx],
                    core_b_to_a[j],
                    core_b_to_a[yy],
                    mol_a_bond_idxs,
                    mol_a_bond_params,
                    mol_a_angle_idxs,
                    mol_a_angle_params,
                )

                if b_okay and a_okay:
                    x = xx
                    y = yy
                    break

            atoms = find_attached_dummy_atoms(dummy_group, mol_b_bond_idxs, anchor)
            assert len(atoms) == 1
            k = atoms[0]

            restraint_cross_angle_idxs.append(((j, x), (j, y), (j, k)))
            restraint_cross_angle_params.append(1000.0)

        else:
            assert 0, "Illegal Geometry"

    else:
        assert 0, "Illegal Geometry"

    all_idxs = (
        restraint_bond_idxs,
        restraint_angle_idxs,
        restraint_proper_idxs,
        restraint_improper_idxs,
        restraint_cross_angle_idxs,
        restraint_centroid_angle_idxs,
    )
    all_params = (
        restraint_bond_params,
        restraint_angle_params,
        restraint_proper_params,
        restraint_improper_params,
        restraint_cross_angle_params,
        restraint_centroid_angle_params,
    )
    return all_idxs, all_params


def setup_end_state(ff, mol_a, mol_b, core, a_to_c, b_to_c):
    """
    Setup end-state for mol_a with dummy atoms of mol_b attached. The mapped indices will correspond
    to the alchemical molecule with dummy atoms.

    Parameters
    ----------
    mol_a: Chem.Mol

    mol_b: Chem.Mol

    core: list of 2-tuples

    a_to_c: dict or array, supports []
        mapping from a into c

    b_to_c: dict or array, supports []
        mapping from b into c

    Returns
    -------
    (all_idxs, all_parameters)
        A tuple of bonded, angle, proper, c_angle, x_angle idxs and parameters.

    """
    mol_b_top = topology.BaseTopology(mol_b, ff)

    _, mol_b_hb = mol_b_top.parameterize_harmonic_bond(ff.hb_handle.params)
    mol_b_bond_idxs = mol_b_hb.get_idxs()

    mol_b_core = core[:, 1]
    mol_b_bond_idxs = utils.get_romol_bonds(mol_b)
    dgs = dummy.identify_dummy_groups(mol_b_bond_idxs, mol_b_core)

    all_dummy_bond_idxs, all_dummy_bond_params = [], []
    all_dummy_angle_idxs, all_dummy_angle_params = [], []
    all_dummy_proper_idxs, all_dummy_proper_params = [], []
    all_dummy_improper_idxs, all_dummy_improper_params = [], []
    all_dummy_x_angle_idxs, all_dummy_x_angle_params = [], []
    all_dummy_c_angle_idxs, all_dummy_c_angle_params = [], []
    # gotta add 'em all!
    for dg in dgs:
        dg = list(dg)
        root_anchors = dummy.identify_root_anchors(mol_b_bond_idxs, mol_b_core, dg[0])
        anchor = root_anchors[0]
        all_idxs, all_params = setup_dummy_interactions(ff, mol_a, mol_b, core, dg, anchor=anchor)

        all_dummy_bond_idxs.extend(all_idxs[0])
        all_dummy_angle_idxs.extend(all_idxs[1])
        all_dummy_proper_idxs.extend(all_idxs[2])
        all_dummy_improper_idxs.extend(all_idxs[3])
        all_dummy_x_angle_idxs.extend(all_idxs[4])
        all_dummy_c_angle_idxs.extend(all_idxs[5])

        all_dummy_bond_params.extend(all_params[0])
        all_dummy_angle_params.extend(all_params[1])
        all_dummy_proper_params.extend(all_params[2])
        all_dummy_improper_params.extend(all_params[3])
        all_dummy_x_angle_params.extend(all_params[4])
        all_dummy_c_angle_params.extend(all_params[5])

    # generate parameters for mol_a
    mol_a_top = topology.BaseTopology(mol_a, ff)
    mol_a_bond_params, mol_a_hb = mol_a_top.parameterize_harmonic_bond(ff.hb_handle.params)
    mol_a_angle_params, mol_a_ha = mol_a_top.parameterize_harmonic_angle(ff.ha_handle.params)
    mol_a_proper_params, mol_a_pt = mol_a_top.parameterize_proper_torsion(ff.pt_handle.params)
    mol_a_improper_params, mol_a_it = mol_a_top.parameterize_improper_torsion(ff.it_handle.params)
    mol_a_nbpl_params, mol_a_nbpl = mol_a_top.parameterize_nonbonded_pairlist(ff.q_handle.params, ff.lj_handle.params)

    mol_a_bond_params = mol_a_bond_params.tolist()
    mol_a_angle_params = mol_a_angle_params.tolist()
    mol_a_proper_params = mol_a_proper_params.tolist()
    mol_a_improper_params = mol_a_improper_params.tolist()
    mol_a_nbpl_params = mol_a_nbpl_params.tolist()

    mol_a_bond_idxs = recursive_map(mol_a_hb.get_idxs(), a_to_c)
    mol_a_angle_idxs = recursive_map(mol_a_ha.get_idxs(), a_to_c)
    mol_a_proper_idxs = recursive_map(mol_a_pt.get_idxs(), a_to_c)
    mol_a_improper_idxs = recursive_map(mol_a_it.get_idxs(), a_to_c)
    mol_a_nbpl_idxs = recursive_map(mol_a_nbpl.get_idxs(), a_to_c)

    all_dummy_bond_idxs = recursive_map(all_dummy_bond_idxs, b_to_c)
    all_dummy_angle_idxs = recursive_map(all_dummy_angle_idxs, b_to_c)
    all_dummy_proper_idxs = recursive_map(all_dummy_proper_idxs, b_to_c)
    all_dummy_improper_idxs = recursive_map(all_dummy_improper_idxs, b_to_c)
    all_dummy_x_angle_idxs = recursive_map(all_dummy_x_angle_idxs, b_to_c)
    all_dummy_c_angle_idxs = recursive_map(all_dummy_c_angle_idxs, b_to_c)

    # parameterize the combined molecule
    mol_c_bond_idxs = mol_a_bond_idxs + all_dummy_bond_idxs
    mol_c_bond_params = mol_a_bond_params + all_dummy_bond_params

    mol_c_angle_idxs = mol_a_angle_idxs + all_dummy_angle_idxs
    mol_c_angle_params = mol_a_angle_params + all_dummy_angle_params

    mol_c_proper_idxs = mol_a_proper_idxs + all_dummy_proper_idxs
    mol_c_proper_params = mol_a_proper_params + all_dummy_proper_params

    mol_c_improper_idxs = mol_a_improper_idxs + all_dummy_improper_idxs
    mol_c_improper_params = mol_a_improper_params + all_dummy_improper_params

    mol_c_nbpl_idxs = mol_a_nbpl_idxs
    num_combined_atoms = mol_a.GetNumAtoms() + mol_b.GetNumAtoms() - len(core)
    mol_c_nbpl_params = np.zeros((num_combined_atoms, 3))
    for a_idx, a_param in enumerate(mol_a_nbpl_params):
        mol_c_nbpl_params[a_to_c[a_idx]] = a_param

    # no need to parameterize dummy atoms of mol_b for the pairlist.
    mol_c_x_angle_idxs = all_dummy_x_angle_idxs
    mol_c_x_angle_params = all_dummy_x_angle_params

    mol_c_c_angle_idxs = all_dummy_c_angle_idxs
    mol_c_c_angle_params = all_dummy_c_angle_params

    return (
        (mol_c_bond_idxs, mol_c_bond_params),
        (mol_c_angle_idxs, mol_c_angle_params),
        (mol_c_proper_idxs, mol_c_proper_params),
        (mol_c_improper_idxs, mol_c_improper_params),
        (
            mol_c_nbpl_idxs,
            mol_a_nbpl.get_rescale_mask(),
            mol_a_nbpl.get_beta(),
            mol_a_nbpl.get_cutoff(),
            mol_c_nbpl_params,
        ),
        (mol_c_x_angle_idxs, mol_c_x_angle_params),
        (mol_c_c_angle_idxs, mol_c_c_angle_params),
    )


class AtomMembership(Enum):
    CORE = 0
    DUMMY_A = 1
    DUMMY_B = 2


class SingleTopologyV2:
    def __init__(self, mol_a, mol_b, core, forcefield):
        """
        SingleTopology combines two molecules through a common core. The combined mol has
        atom indices laid out such that mol_a is identically mapped to the combined mol indices.
        The atoms in the mol_b's R-group is then glued on to resulting molecule.

        Parameters
        ----------
        mol_a: ROMol
            First ligand

        mol_b: ROMol
            Second ligand

        core: np.array (C, 2)
            Atom mapping from mol_a to to mol_b.

        # ff: ff.Forcefield
            # Forcefield to be used for parameterization.

        """
        assert mol_a is not None
        assert mol_b is not None
        self.mol_a = mol_a
        self.mol_b = mol_b
        self.core = core
        self.ff = forcefield

        assert core.shape[1] == 2

        # map into idxs in the combined molecule
        self.a_to_c = np.arange(mol_a.GetNumAtoms(), dtype=np.int32)  # identity
        self.b_to_c = np.zeros(mol_b.GetNumAtoms(), dtype=np.int32) - 1

        self.NC = mol_a.GetNumAtoms() + mol_b.GetNumAtoms() - len(core)

        # mark membership in the combined molecule:
        # 0: CORE
        # 1: DUMMY_A (default)
        # 2: DUMMY_B
        self.c_flags = np.ones(self.get_num_atoms(), dtype=np.int32)

        for a, b in core:
            self.c_flags[a] = 0
            self.b_to_c[b] = a

        iota = self.mol_a.GetNumAtoms()
        for b_idx, c_idx in enumerate(self.b_to_c):
            if c_idx == -1:
                self.b_to_c[b_idx] = iota
                self.c_flags[iota] = 2
                iota += 1

        # test for uniqueness in core idxs for each mol
        assert len(set(tuple(core[:, 0]))) == len(core[:, 0])
        assert len(set(tuple(core[:, 1]))) == len(core[:, 1])

    def get_num_atoms(self):
        return self.NC

    def lookup_a_in_c(self, core_idx):
        for a, c in enumerate(self.a_to_c):
            if c == core_idx:
                return a

    def lookup_b_in_c(self, core_idx):
        for b, c in enumerate(self.b_to_c):
            if c == core_idx:
                return b

    def generate_end_state_mol_a(self):
        return setup_end_state(self.ff, self.mol_a, self.mol_b, self.core, self.a_to_c, self.b_to_c)

    def generate_end_state_mol_b(self):
        core = self.core[:, [1, 0]]  # swap
        return setup_end_state(self.ff, self.mol_b, self.mol_a, core, self.b_to_c, self.a_to_c)

    def combine_confs(self, x_a, x_b):
        assert x_a.shape == (self.mol_a.GetNumAtoms(), 3)
        assert x_b.shape == (self.mol_b.GetNumAtoms(), 3)
        x0 = np.zeros((self.get_num_atoms(), 3))
        for src, dst in enumerate(self.a_to_c):
            x0[dst] = x_a[src]
        for src, dst in enumerate(self.b_to_c):
            x0[dst] = x_b[src]
        return x0
