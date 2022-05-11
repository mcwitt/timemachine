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


class SingleTopologyV2:
    def core_b_to_a(self, b_idx):
        for a, b in self.core:
            if b == b_idx:
                return a

    def core_b_to_a(self, a_idx):
        for a, b in self.core:
            if a == a_idx:
                return b

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

        # classify geometry of the full molecule

        # for each dummy group:
        # 1) interactions involving only dummy atoms stays on.
        # 2) interactions involving dummy atoms and the anchor stays on.

        # build bond_idxs of the combined molecule
        # self.combined_bond_idxs = set()

        # for bond in self.mol_a.GetBonds():
        #     src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        #     src, dst = dummy.canonicalize_bond((src, dst))
        #     self.combined_bond_idxs.add((src, dst))

        # for bond in self.mol_b.GetBonds():
        #     src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        #     src, dst = dummy.canonicalize_bond((self.b_to_c[src], self.b_to_c[dst]))
        #     self.combined_bond_idxs.add((src, dst))

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

    # @staticmethod
    # def _find_stereo_bonds(proper_torsion_idxs, proper_torsion_params):
    #     for bond_idxs, params in zip(proper_torsion_idxs, proper_torsion_params):
    #         k, phase, period = params
    #         stereo_bond

    def get_mol_b_dummy_anchor_ixns(self):
        """
        At src, R-A atoms are real, i.e. fully interacting. R-B is in a dummy state, where:

        1) Dummy B atoms do not interact with the environment via nonbonded interactions.
        2) Dummy B atoms interact with the anchoring atom and a specific set of bonded interactins:
            a) Bonds
            b) Angles
            c) Stereo Torsions
            d) Stereo Angles

            Dummy B atoms are not allowed to interact with R-A atoms. While the choice of core atoms
            is arbitrary, the dummy restraints must be all-or-nothing turned on, and numerically stable.

        This does not include core-core or dummy-dummy interactions. Those are expected to be processed
        externally.
        """

        # parameterize mol_a and mol_b independently.
        mol_a_top = topology.BaseTopology(self.mol_a, self.ff)
        mol_b_top = topology.BaseTopology(self.mol_b, self.ff)

        mol_a_bond_params, mol_a_hb = mol_a_top.parameterize_harmonic_bond(self.ff.hb_handle.params)
        mol_a_angle_params, mol_a_ha = mol_a_top.parameterize_harmonic_angle(self.ff.ha_handle.params)
        mol_a_proper_params, mol_a_pt = mol_a_top.parameterize_proper_torsion(self.ff.pt_handle.params)
        mol_a_improper_params, mol_a_it = mol_a_top.parameterize_proper_torsion(self.ff.pt_handle.params)

        mol_b_bond_params, mol_b_hb = mol_b_top.parameterize_harmonic_bond(self.ff.hb_handle.params)
        mol_b_angle_params, mol_b_ha = mol_b_top.parameterize_harmonic_angle(self.ff.ha_handle.params)
        mol_b_proper_params, mol_b_pt = mol_b_top.parameterize_proper_torsion(self.ff.pt_handle.params)
        mol_b_improper_params, mol_b_it = mol_b_top.parameterize_improper_torsion(self.ff.it_handle.params)

        mol_a_bond_idxs = mol_a_hb.get_idxs()
        mol_a_angle_idxs = mol_a_ha.get_idxs()
        mol_a_proper_idxs = mol_a_pt.get_idxs()
        mol_a_improper_idxs = mol_a_it.get_idxs()

        mol_b_bond_idxs = mol_b_hb.get_idxs()
        mol_b_angle_idxs = mol_b_ha.get_idxs()
        mol_b_proper_idxs = mol_b_pt.get_idxs()
        mol_b_improper_idxs = mol_b_it.get_idxs()

        mol_b_core = self.core[:, 1]
        mol_b_full_geometry = geometry.classify_geometry(self.mol_b)
        mol_b_core_geometry = geometry.classify_geometry(self.mol_b, core=mol_b_core)

        dummy_groups_b = dummy.identify_dummy_groups(mol_b_bond_idxs, mol_b_core)

        restraint_bond_idxs = []
        restraint_bond_params = []

        restraint_angle_idxs = []
        restraint_angle_params = []

        restraint_proper_idxs = []
        restraint_proper_params = []

        restraint_improper_idxs = []
        restraint_improper_params = []

        for dg in dummy_groups_b:

            dg = list(dg)
            # pick an arbitrary atom in the dummy_group
            root_anchors = dummy.identify_root_anchors(mol_b_bond_idxs, mol_b_core, dg[0])
            assert len(root_anchors) == 1
            anchor = root_anchors[0]

            # first process interactions that involve only the anchor atom and dummy atoms. These are always safe to include.
            # rule: either 
            dga = dg + [anchor]
            for idxs, params in zip(mol_b_bond_idxs, mol_b_bond_params):
                # core/anchor + exactly one anchor atom.
                if np.all([a in dga for a in idxs]) and np.sum([a == anchor for a in idxs]) == 1:
                    restraint_bond_idxs.append(idxs)
                    restraint_bond_params.append(params)
            for idxs, params in zip(mol_b_angle_idxs, mol_b_angle_params):
                if np.all([a in dga for a in idxs]) and np.sum([a == anchor for a in idxs]) == 1:
                    restraint_angle_idxs.append(idxs)
                    restraint_angle_params.append(params)
            for idxs, params in zip(mol_b_proper_idxs, mol_b_proper_params):
                if np.all([a in dga for a in idxs]) and np.sum([a == anchor for a in idxs]) == 1:
                    restraint_proper_idxs.append(idxs)
                    restraint_proper_params.append(params)
            for idxs, params in zip(mol_b_improper_idxs, mol_b_improper_params):
                if np.all([a in dga for a in idxs]) and np.sum([a == anchor for a in idxs]) == 1:
                    restraint_improper_idxs.append(idxs)
                    restraint_improper_params.append(params)

            anchor_core_geometry = mol_b_core_geometry[anchor]
            anchor_full_geometry = mol_b_full_geometry[anchor]

            nbs_1, nbs_2 = enumerate_anchor_groups(anchor, mol_b_bond_idxs, mol_b_core)
            print("nbs_1", nbs_1)
            print("nbs_2", nbs_2)

            if anchor_core_geometry == LocalGeometry.G1_TERMINAL:
                assert len(nbs_1) == 1
                i = list(nbs_1)[0]
                j = anchor
                # ring opening and closing on terminal bonds are illegal.
                if anchor_full_geometry == LocalGeometry.G2_KINK:
                    atoms = find_attached_dummy_atoms(dg, mol_b_bond_idxs, anchor)
                    assert len(atoms) == 1
                    k = atoms[0]
                    restraint_angle_idxs.append((i,j,k))
                    restraint_angle_params.append((100.0, 2*np.pi/3))
                elif anchor_full_geometry == LocalGeometry.G2_LINEAR:
                    atoms = find_attached_dummy_atoms(dg, mol_b_bond_idxs, anchor)
                    assert len(atoms) == 1
                    k = atoms[0]
                    restraint_angle_idxs.append((i,j,k))
                    restraint_angle_params.append((100.0, np.pi))
                elif anchor_full_geometry == LocalGeometry.G3_PYRAMIDAL:
                    atoms = find_attached_dummy_atoms(dg, mol_b_bond_idxs, anchor)
                    assert len(atoms) == 2
                    k0, k1 = atoms
                    restraint_angle_idxs.append((i,j,k0))
                    restraint_angle_params.append((100.0, 1.91)) # 109 degrees
                    restraint_angle_idxs.append((i,j,k1))
                    restraint_angle_params.append((100.0, 1.91)) # 109 degrees
                elif anchor_full_geometry == LocalGeometry.G3_PLANAR:
                    atoms = find_attached_dummy_atoms(dg, mol_b_bond_idxs, anchor)
                    assert len(atoms) == 2
                    k0, k1 = atoms
                    restraint_angle_idxs.append((i,j,k0))
                    restraint_angle_params.append((100.0, 2*np.pi/3))
                    restraint_angle_idxs.append((i,j,k1))
                    restraint_angle_params.append((100.0, 2*np.pi/3))

                    assert 0
                    # need to add missing torsions for these.
                elif anchor_full_geometry == LocalGeometry.G4_TETRAHEDRAL:
                    atoms = find_attached_dummy_atoms(dg, mol_b_bond_idxs, anchor)
                    assert len(atoms) == 3
                    k0, k1, k2 = atoms
                    print("3X!")
                    restraint_angle_idxs.append((i,j,k0))
                    restraint_angle_params.append((100.0, 1.91))
                    restraint_angle_idxs.append((i,j,k1))
                    restraint_angle_params.append((100.0, 1.91))
                    restraint_angle_idxs.append((i,j,k2))
                    restraint_angle_params.append((100.0, 1.91))
                else:
                    assert 0, "Illegal Geometry"
            elif anchor_core_geometry == LocalGeometry.G2_KINK:
                if anchor_full_geometry == LocalGeometry.G3_PYRAMIDAL:
                    pass
                elif anchor_full_geometry == LocalGeometry.G3_PLANAR:

                    atoms = find_attached_dummy_atoms(dg, mol_b_bond_idxs, anchor)
                    assert len(atoms) == 1
                    i = atoms[0]
                    j = anchor # j for junction atom

                    # find atoms we can anchor dummy atoms against 
                    # _, nbs_2 = enumerate_anchor_groups(anchor, mol_b_bond_idxs, mol_b_core)
                    assert len(nbs_2) > 0

                    found = False
                    
                    # for each anchor group, see what kind of restraints are needed.
                    for k, l in nbs_2:

                        # check that end-states are numerically stable
                        mol_b_stable = check_bond_angle_stability(
                            j,
                            k,
                            l,
                            mol_b_bond_idxs,
                            mol_b_bond_params,
                            mol_b_angle_idxs,
                            mol_b_angle_params,
                        )

                        mol_a_stable = check_bond_angle_stability(
                            self.core_b_to_a(j),
                            self.core_b_to_a(k),
                            self.core_b_to_a(l),
                            mol_a_bond_idxs,
                            mol_a_bond_params,
                            mol_a_angle_idxs,
                            mol_a_angle_params,
                        )

                        if mol_b_stable and mol_a_stable:
                            # add one angle term
                            ijk = dummy.canonicalize_bond((i,j,k))
                            restraint_angle_idxs.append(ijk) # tbd: canonicalize?
                            restraint_angle_params.append((100.0, 2.09))

                            for (a, b, c, d), (force_constant, phase, period) in zip(mol_b_proper_idxs, mol_b_proper_params):
                                if abs(phase - np.pi) < 0.05 and period == 2:
                                    # dummy-dummy-core-core, fwd/rev matches
                                    # dummy-core-core-core, fwd/rev matches
                                    if (a in dg and b in dg and c == j and d == k) or \
                                       (d in dg and c in dg and b == j and a == k) or \
                                       (a in dg and b == j and c == k and d == l) or \
                                       (d in dg and c == j and b == k and a == l):

                                        abcd = dummy.canonicalize_bond((a,b,c,d))
                                        assert abcd not in restraint_proper_idxs
                                        restraint_proper_idxs.append(abcd)
                                        restraint_proper_params.append((50.0, np.pi, 2))

                            # found a stable set of restraints, break from anchor group loop
                            break

                elif anchor_full_geometry == LocalGeometry.G4_TETRAHEDRAL:
                    pass
                else:
                    assert 0, "Illegal Geometry"
            elif anchor_core_geometry == LocalGeometry.G2_LINEAR:
                assert 0, "Illegal Geometry"
            elif anchor_core_geometry == LocalGeometry.G3_PLANAR:
                assert 0, "Illegal Geometry"
            elif anchor_core_geometry == LocalGeometry.G3_PYRAMIDAL:
                assert 0, "Illegal Geometry"
            else:
                assert 0, "Illegal Geometry"

            print("dg", dg, "anchor", anchor, "core_geom", anchor_core_geometry, "full_geom", anchor_full_geometry)
        
        print("restraint_proper_idxs", restraint_proper_idxs)
        print("restraint_improper_idxs", restraint_improper_idxs)
        print("restraint_angle_idxs", restraint_angle_idxs)
        print("restraint_bond_idxs", restraint_bond_idxs)
