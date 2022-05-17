import numpy as np

from timemachine.fe import dummy, geometry, topology
from timemachine.fe.geometry import LocalGeometry


def enumerate_anchor_groups(anchor_idx, bond_idxs, core_idxs):

    # enumerate all 1 and 2 neighbor anchor atoms to form valid anchor groups.

    assert anchor_idx in core_idxs
    assert anchor_idx in [x[0] for x in bond_idxs] or anchor_idx in [x[1] for x in bond_idxs]

    nbs_1 = set()
    nbs_2 = set()
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


def generate_required_torsions(dg, bond_idxs, anchor, core, proper_idxs, proper_params):
    """
    Generate planarizing torsions that are required by dummy atoms 1 or 2 bonds away
    from the anchoring atom. Typically, given a particular choice of anchor groups, we
    require that every dummy item in the returned dict has at least one term satisified.

    Parameters
    ----------
    dg: list of int
        Atoms in the dummy group

    bond_idxs: list of 2-tuples
        Bonds connecting atoms in the graph

    anchor: int
        Junction atom belonging to the core

    proper_idxs: list of 4-tuple
        Torsion idxs for proper terms

    proper_params: list of 3-tuple
        Force constant, phase, period of the proper params

    Returns
    -------
    dict
        Keys are dummy atoms that are 1 or 2 bonds away from the anchor
        Values are torsions that span into the anchor.

    """
    required_torsions_main = dict()
    required_torsions_side = dict()
    atoms = find_attached_dummy_atoms(dg, bond_idxs, anchor)
    for m in atoms:
        for (a, b, c, d), (force, phase, period) in zip(proper_idxs, proper_params):
            if period == 2 and (phase - np.pi) < 0.05 and force > 0.0:
                # found a planarizing torsion
                if (a == m and b in core and c in core and d in core) or (
                    a in core and b in core and c in core and d == m
                ):
                    if m not in required_torsions_main:
                        required_torsions_main[m] = []
                    required_torsions_main[m].append((a, b, c, d))
        side_dummies = set()
        for i, j in bond_idxs:
            if i == m and j in dg:
                side_dummies.add(j)
            elif j == m and i in dg:
                side_dummies.add(i)
        for s in side_dummies:
            for (a, b, c, d), (force, phase, period) in zip(proper_idxs, proper_params):
                if period == 2 and (phase - np.pi) < 0.05 and force > 0.0:
                    if (a == s and b == m and c in core and d in core) or (
                        a in core and b in core and c == m and d == s
                    ):
                        if s not in required_torsions_side:
                            required_torsions_side[s] = []
                        required_torsions_side[s].append((a, b, c, d))

    return required_torsions_main, required_torsions_side


def identify_bonds_spanned_by_planar_torsions(proper_idxs, proper_params):
    """
    Identify bonds that are spanned by planar torsions and returns a dict of bonds
    and associated torsions that span it.
    """
    planar_bonds = dict()

    for (i, j, k, l), (force, phase, period) in zip(proper_idxs, proper_params):
        if period == 2 and (phase - np.pi) < 0.05 and force > 5.0:
            canon_jk = dummy.canonicalize_bond((j, k))
            if canon_jk not in planar_bonds:
                planar_bonds[canon_jk] = []
            planar_bonds[canon_jk].append((i, j, k, l))

    return planar_bonds


def find_stereo_bonds(ring_bonds, proper_idxs, proper_params):
    # a stereo bond is defined as a bond that
    # 1) has a proper torsion term that has k > 0, period=2, phase=3.1415
    # 2) a bond that is not part of a ring system.
    # 3)
    # the reason why 2) is present is because the planar torsions spanning a
    # ring system are not used to enforce stereochemistry, since if we simply
    # disabled them, we would *still* get the correct stereochemistry.
    # consider a benzene devoid of any torsions (proper or improper), or non-bonded
    # terms, and only angles and bonds are present. The hydrogens would still be correctly
    # placed since the angles intrinsically restrain them.

    canonical_ring_bonds = set()
    for ij in ring_bonds:
        canonical_ring_bonds.add(dummy.canonicalize_bond(ij))

    planar_bonds_kv = identify_bonds_spanned_by_planar_torsions(proper_idxs, proper_params)
    canonical_stereo_bonds = set()
    for k in planar_bonds_kv.keys():
        if k not in canonical_ring_bonds:
            canonical_stereo_bonds.add(k)

    return canonical_stereo_bonds


def find_stereo_atoms(mol):
    mol_geom = geometry.classify_geometry(mol)
    stereo_atoms = set()
    for a in mol.GetAtoms():
        a_idx = a.GetIdx()
        nbs = a.GetNeighbors()
        if len(nbs) == 4:
            stereo_atoms.add(a_idx)
        elif len(nbs) == 3 and mol_geom[a_idx] == LocalGeometry.G3_PYRAMIDAL:
            # if in ring, or is sulfur or phosphorus, this may not be guaranteed by
            if a.IsInRing() or a.GetAtomicNum() == 16 or a.GetAtomicNum() == 15:
                stereo_atoms.add(a_idx)
        elif len(nbs) > 4:
            assert 0
    return stereo_atoms


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
        canonical_angles[dummy.canonicalize_bond(idxs)] = params

    jkl = dummy.canonicalize_bond((j, k, l))

    if jkl not in canonical_angles:
        return False
    elif canonical_angles[jkl][0] < 50.0:
        return False
    elif abs(canonical_angles[jkl][1] - 0) < 0.05:
        return False
    elif abs(canonical_angles[jkl][1] - 3.1415) < 0.05:
        return False

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


def find_dummy_atoms_one_away(dg, bond_idxs, dummy_next_to_anchor):
    other_dummy_atoms = []
    for a, b in bond_idxs:
        if a == dummy_next_to_anchor and b in dg:
            other_dummy_atoms.append(b)
        elif b == dummy_next_to_anchor and a in dg:
            other_dummy_atoms.append(a)
    return other_dummy_atoms


def find_junction_bonds(anchor, bond_idxs):
    jbs = set()
    for i, j in bond_idxs:
        assert i != j
        if i == anchor or j == anchor:
            jbs.add(dummy.canonicalize_bond((i, j)))
    return jbs


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

    # tbd, refactor into a function that setups up orientational restraints at an anchor site.
    @staticmethod
    def setup_orientational_restraints(ff, mol_a, mol_b, core, dg, anchor):
        """
        Add restraints between dummy atoms in a dummy_group and core atoms.
        """
        core_b_to_a = dict()
        for a, b in core:
            core_b_to_a[b] = a

        assert anchor in core_b_to_a
        for d in dg:
            assert d not in core_b_to_a

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

        dg = list(dg)
        # pick an arbitrary atom in the dummy_group and find the anchors, there may be
        # multiple anchors, eg (d=dummy, c=core):
        #   d...d
        #   |   |
        #   c---c
        root_anchors = dummy.identify_root_anchors(mol_b_bond_idxs, mol_b_core, dg[0])
        assert anchor in root_anchors
        # (ytz): we can relax this assertion later on.
        assert len(root_anchors) == 1, "multiple root anchors found."

        mol_b_ring_bonds = []
        for b in mol_b.GetBonds():
            if b.IsInRing():
                mol_b_ring_bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))

        stereo_bonds = find_stereo_bonds(mol_b_ring_bonds, mol_b_proper_idxs, mol_b_proper_params)
        stereo_atoms = find_stereo_atoms(mol_b)

        junction_bonds = find_junction_bonds(anchor, mol_b_bond_idxs)
        # (ytz): the mapping code should hopefully be able to guarantee this
        assert junction_bonds.intersection(stereo_bonds) == set()

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
        # numerically stable and factorizable.
        dga = dg + [anchor]
        for idxs, params in zip(mol_b_bond_idxs, mol_b_bond_params):
            # core/anchor + exactly one anchor atom.
            if np.all([a in dga for a in idxs]) and np.sum([a == anchor for a in idxs]) == 1:
                restraint_bond_idxs.append(tuple([int(x) for x in idxs]))  # tuples are hashable etc.
                restraint_bond_params.append(params)
        for idxs, params in zip(mol_b_angle_idxs, mol_b_angle_params):
            if np.all([a in dga for a in idxs]) and np.sum([a == anchor for a in idxs]) == 1:
                restraint_angle_idxs.append(tuple([int(x) for x in idxs]))
                restraint_angle_params.append(params)
        for idxs, params in zip(mol_b_proper_idxs, mol_b_proper_params):
            if np.all([a in dga for a in idxs]) and np.sum([a == anchor for a in idxs]) == 1:
                restraint_proper_idxs.append(tuple([int(x) for x in idxs]))
                restraint_proper_params.append(params)
        for idxs, params in zip(mol_b_improper_idxs, mol_b_improper_params):
            if np.all([a in dga for a in idxs]) and np.sum([a == anchor for a in idxs]) == 1:
                restraint_improper_idxs.append(tuple([int(x) for x in idxs]))
                restraint_improper_params.append(params)

        anchor_core_geometry = mol_b_core_geometry[anchor]
        anchor_full_geometry = mol_b_full_geometry[anchor]
        # print(anchor_core_geometry, anchor_full_geometry)
        # specialized restraints that are factorizable
        restraint_cross_angle_idxs = []
        restraint_cross_angle_params = []

        restraint_centroid_angle_idxs = []
        restraint_centroid_angle_params = []

        nbs_1, nbs_2 = enumerate_anchor_groups(anchor, mol_b_bond_idxs, mol_b_core)

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
                # type:
                #     k
                #    .
                # i-j
                # add one angle between (i,j,k)
                atoms = find_attached_dummy_atoms(dg, mol_b_bond_idxs, anchor)
                assert len(atoms) == 1
                k = atoms[0]
                # add one angle
                restraint_angle_idxs.append((i, j, k))
                restraint_angle_params.append((100.0, (2.0 / 3.0) * np.pi))
            elif anchor_full_geometry == LocalGeometry.G2_LINEAR:
                # type:
                # i-j.k
                # add one angle between (i,j,k)
                atoms = find_attached_dummy_atoms(dg, mol_b_bond_idxs, anchor)
                assert len(atoms) == 1
                k = atoms[0]
                restraint_angle_idxs.append((i, j, k))
                restraint_angle_params.append((100.0, np.pi))
            elif anchor_full_geometry == LocalGeometry.G3_PYRAMIDAL:
                # type:
                # i-j.k0
                #    .
                #     k1
                # add two angles: (i,j,k0) and (i,j,k1)
                atoms = find_attached_dummy_atoms(dg, mol_b_bond_idxs, anchor)
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
                # type:
                #     k0
                #    .
                # i-j
                #    .
                #     k1
                # add two angles: (i,j,k0) and (i,j,k1)
                # typically we'd have to worry about stereochemistry, but we're
                # pretty confident here we don't have any stereo issues.
                atoms = find_attached_dummy_atoms(dg, mol_b_bond_idxs, anchor)
                assert len(atoms) == 2
                k0, k1 = atoms
                restraint_angle_idxs.append((i, j, k0))
                restraint_angle_params.append((100.0, (2.0 / 3.0) * np.pi))
                restraint_angle_idxs.append((i, j, k1))
                restraint_angle_params.append((100.0, (2.0 / 3.0) * np.pi))
            elif anchor_full_geometry == LocalGeometry.G4_TETRAHEDRAL:
                # type:
                #     k0
                #    .
                # i-j . k2
                #    .
                #     k1
                # add three angles: (i,j,k0) and (i,j,k1) and (i,j,k2)
                assert anchor in stereo_atoms
                atoms = find_attached_dummy_atoms(dg, mol_b_bond_idxs, anchor)
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
                atoms = find_attached_dummy_atoms(dg, mol_b_bond_idxs, anchor)
                assert len(atoms) == 1
                k = atoms[0]

                if anchor in stereo_atoms:
                    # two options
                    # 1) two wells at zero and pi
                    # 2) one well explicitly encoding the stereo - probably better!
                    # implement 1 for now?
                    restraint_cross_angle_idxs.append(((j, a), (j, b), (j, k)))
                    restraint_cross_angle_params.append((100.0, 0.0))
                else:
                    # planarize so we can enhance sample both stereoisomers using a centroid
                    # type a
                    #       \
                    #      c.j.k <- angle (c,j,k) = 0.0
                    #       /
                    #      b
                    restraint_centroid_angle_idxs.append((tuple(sorted(a, b)), j, k))
                    restraint_centroid_angle_params.append((100.0, 0.0))

            elif anchor_full_geometry == LocalGeometry.G3_PLANAR:
                # same as G3_PYRAMIDAL non-stereo
                # print("anchor", anchor, "st", stereo_atoms)
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
                atoms = find_attached_dummy_atoms(dg, mol_b_bond_idxs, anchor)
                assert len(atoms) == 1
                k = atoms[0]
                restraint_centroid_angle_idxs.append((tuple(sorted((a, b))), j, k))
                restraint_centroid_angle_params.append((100.0, 0.0))

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

                atoms = find_attached_dummy_atoms(dg, mol_b_bond_idxs, anchor)
                assert len(atoms) == 2
                k, l = atoms
                # dummy-atoms l,j,k
                assert check_bond_angle_stability(
                    l, j, k, mol_b_bond_idxs, mol_b_bond_params, mol_b_angle_idxs, mol_b_angle_params
                )
                restraint_cross_angle_idxs.append(((j, a), (j, b), (j, k)))
                restraint_cross_angle_params.append((100.0, 0.0))
                restraint_cross_angle_idxs.append(((j, a), (j, b), (j, l)))
                restraint_cross_angle_params.append((100.0, 0.0))
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
                # we currently support only 1) right now. But it would not be difficult to implement 2).

                a, b, c = nbs_1
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
                # core-atoms a,j,c
                assert check_bond_angle_stability(
                    a, j, c, mol_b_bond_idxs, mol_b_bond_params, mol_b_angle_idxs, mol_b_angle_params
                )
                assert check_bond_angle_stability(
                    core_b_to_a[a],
                    core_b_to_a[j],
                    core_b_to_a[c],
                    mol_a_bond_idxs,
                    mol_a_bond_params,
                    mol_a_angle_idxs,
                    mol_a_angle_params,
                )
                # core-atoms b,j,c
                assert check_bond_angle_stability(
                    b, j, c, mol_b_bond_idxs, mol_b_bond_params, mol_b_angle_idxs, mol_b_angle_params
                )
                assert check_bond_angle_stability(
                    core_b_to_a[b],
                    core_b_to_a[j],
                    core_b_to_a[c],
                    mol_a_bond_idxs,
                    mol_a_bond_params,
                    mol_a_angle_idxs,
                    mol_a_angle_params,
                )

                atoms = find_attached_dummy_atoms(dg, mol_b_bond_idxs, anchor)
                assert len(atoms) == 1
                k = atoms[0]

                restraint_centroid_angle_idxs.append((tuple(sorted((a, b, c))), j, k))
                restraint_centroid_angle_params.append((100.0, 0.0))

            else:
                assert 0, "Illegal Geometry"

        else:
            assert 0, "Illegal Geometry"

        return (
            restraint_bond_idxs,
            restraint_bond_params,
            restraint_angle_idxs,
            restraint_angle_params,
            restraint_proper_idxs,
            restraint_proper_params,
            restraint_improper_idxs,
            restraint_improper_params,
            restraint_cross_angle_idxs,
            restraint_cross_angle_params,
            restraint_centroid_angle_idxs,
            restraint_centroid_angle_params,
        )
