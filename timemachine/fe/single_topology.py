import numpy as np

from timemachine.fe import dummy, geometry


class SingleTopologyV2:
    def __init__(self, mol_a, mol_b, core):
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
        # self.ff = ff
        self.core = core

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
        self.mol_a_geometry = geometry.classify_geometry(self.mol_a)
        self.mol_b_geometry = geometry.classify_geometry(self.mol_b)

        # classify gemoetry of the cores
        self.core_a_geometry = geometry.classify_geometry(self.mol_a, core[:, 0])
        self.core_b_geometry = geometry.classify_geometry(self.mol_b, core[:, 1])

        # for each dummy group:
        # 1) interactions involving only dummy atoms stays on.
        # 2) interactions involving dummy atoms and the anchor stays on.

        # build bond_idxs of the combined molecule
        self.combined_bond_idxs = set()

        for bond in self.mol_a.GetBonds():
            src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src, dst = dummy.canonicalize_bond((src, dst))
            self.combined_bond_idxs.add((src, dst))

        for bond in self.mol_b.GetBonds():
            src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src, dst = dummy.canonicalize_bond((self.b_to_c[src], self.b_to_c[dst]))
            self.combined_bond_idxs.add((src, dst))

        self.combined_bond_idxs = list(self.combined_bond_idxs)

        # important tbd:
        # assert core geometries are identical between the two molecules
        # assert that connectivity of the core geometries are identical.

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

    def add_restraints_src(self):
        core_src = []
        for idx, flag in enumerate(self.c_flags):
            if flag == 0 or flag == 1:
                core_src.append(idx)

        dummy_groups = dummy.identify_dummy_groups(self.combined_bond_idxs, core_src)
        for dg in dummy_groups:
            dg = list(dg)
            root_anchors = dummy.identify_root_anchors(self.combined_bond_idxs, core_src, dg[0])
            assert len(root_anchors) == 1
            anchor = root_anchors[0]

            attached_dummy_atoms = []
            for dg_atom in dg:
                if dummy.canonicalize_bond((dg_atom, anchor)) in self.combined_bond_idxs:
                    attached_dummy_atoms.append(dg_atom)

            core_geometry = self.core_a_geometry[self.lookup_a_in_c(anchor)]
            full_geometry = self.mol_b_geometry[self.lookup_b_in_c(anchor)]

            restraint_bond_idxs = []
            restraint_bond_params = []

            restraint_angle_idxs = []
            restraint_angle_params = []

            restraint_cross_product_idxs = []  # i,j,k,l [ij x ik, i, l]
            restraint_cross_product_params = []  # k, a0

            restraint_torsion_idxs = []
            restraint_torsion_params = []

            if core_geometry == geometry.G1_TERMINAL:
                if full_geometry == geometry.G1_TERMINAL:
                    assert len(dummy_groups) == 0
                elif full_geometry == geometry.G2_PLANAR:
                    assert len(dummy_groups) == 1
                    # required: 1 bond @ 1nm
                    #         : 1 angle restraint @ 120 or 109.5
                    # optional: 1 torsion if bond is not rotatable.
                    #         : dummy-core is not rotatable, check bond order.
                    #         : core-core is not rotatable, check bond order.
                    #         : add special case for amide bond.
                elif full_geometry == geometry.G2_LINEAR:
                    # required: 1 bond @ 1nm
                    #         : 1 angle @ 180
                    # extend geometry not supported
                    assert len(dummy_groups) == 1
                elif full_geometry == geometry.G3_PLANAR:
                    # required: 2 bonds set to 1nm, 3 angles set to 120 (or copied).
                    assert len(dummy_groups) == 2
                elif full_geometry == geometry.G3_PYRAMIDAL:
                    # required: StrictStereo=Off 2 bonds set to 1nm, 3 angles set to 120 degrees
                    #           StrictStereo=On 2 bonds set to 1nm, 3 angles set to 109.5
                    assert len(dummy_groups) == 2
                elif full_geometry == geometry.G4_TETRAHEDRAL:
                    # required: 3 bonds, 6 angles set to 120 degrees with stereo matching
                    assert len(dummy_groups) == 3
            elif core_geometry == geometry.G2_KINK:
                if full_geometry == geometry.G1_TERMINAL:
                    # invalid
                    assert 0
                elif full_geometry == geometry.G2_KINK:
                    # no need for dummy atoms
                    pass
                elif full_geometry == geometry.G2_LINEAR:
                    assert 0  # mapped cores must be identical in geometry
                elif full_geometry == geometry.G3_PLANAR:
                    assert len(dummy_groups) == 1
                    # required: 1 bond @ 1nm, 1 angle @ 120.
                elif full_geometry == geometry.G3_PYRAMIDAL:
                    assert len(dummy_groups) == 1
                    # required: 1 bond @ 1nm, 1 angle @ 109.5
                elif full_geometry == geometry.G4_TETRAHEDRAL:
                    assert len(dummy_groups) == 2
                    # 1 bond @ 1nm, 2 angles, 2 cross-product based restraints
            elif core_geometry == geometry.G2_LINEAR:
                if full_geometry == geometry.G1_TERMINAL:
                    assert 0
                elif full_geometry == geometry.G2_KINK:
                    assert 0
                    # invalid geometry
                elif full_geometry == geometry.G2_LINEAR:
                    # do thing
                    pass
                elif full_geometry == geometry.G3_PLANAR:
                    assert 0  # unsupported morph
                elif full_geometry == geometry.G3_PYRAMIDAL:
                    assert 0  # unsupported morph
                elif full_geometry == geometry.G4_TETRAHEDRAL:
                    assert 0  # unsupported morph
            elif core_geometry == geometry.G3_PLANAR:
                if full_geometry == geometry.G1_TERMINAL:
                    assert 0
                elif full_geometry == geometry.G2_KINK:
                    assert 0
                elif full_geometry == geometry.G2_LINEAR:
                    assert 0
                elif full_geometry == geometry.G3_PLANAR:
                    # do
                    pass
                elif full_geometry == geometry.G3_PYRAMIDAL:
                    assert 0
                elif full_geometry == geometry.G4_TETRAHEDRAL:
                    # attach chiral restraints.
                    pass
            elif core_geometry == geometry.G3_PYRAMIDAL:
                if full_geometry == geometry.G1_TERMINAL:
                    assert 0
                elif full_geometry == geometry.G2_LINEAR:
                    assert 0
                elif full_geometry == geometry.G3_PLANAR:
                    assert 0
                elif full_geometry == geometry.G3_PYRAMIDAL:
                    pass
                elif full_geometry == geometry.G4_TETRAHEDRAL:
                    # attach chiral restraints.
                    pass
            elif core_geometry == geometry.G4_TETRAHEDRAL:
                # can't have dummy atoms here.
                assert 0

            print(
                "restraining dummy atoms",
                attached_dummy_atoms,
                "to anchor",
                anchor,
                "full_geometry",
                full_geometry,
                "core_geometry",
                core_geometry,
            )
