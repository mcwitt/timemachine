import functools

import numpy as np
from rdkit import Chem

from timemachine.fe import single_topology, utils, topology, geometry
from timemachine.ff import Forcefield

def test_nblist_conversion():
    mol = Chem.MolFromSmiles("CC1CC1C(C)(C)C")
    bond_idxs = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    nblist = geometry.bond_idxs_to_nblist(bond_idxs)

    expected = [
            [1],
            [0,2,3],
            [1,3],
            [1,2,4],
            [3,5,6,7],
            [4],
            [4],
            [4]
    ]

    np.testing.assert_array_equal(nblist, expected)


def test_flag_stereo():
    mol = Chem.MolFromSmiles("c1ccccc1C")
    mol = Chem.AddHs(mol)

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    bt = topology.BaseTopology(mol, ff)

    bond_params, hb = bt.parameterize_harmonic_bond(ff.hb_handle.params)
    angle_params, ha = bt.parameterize_harmonic_angle(ff.ha_handle.params)
    proper_params, pt = bt.parameterize_proper_torsion(ff.pt_handle.params)
    improper_params, it = bt.parameterize_proper_torsion(ff.pt_handle.params)

    bond_idxs = hb.get_idxs()
    angle_idxs = ha.get_idxs()
    proper_idxs = pt.get_idxs()
    improper_idxs = it.get_idxs()

    atom_geometries, atom_stereo, bond_stereo = geometry.label_stereo(
        bond_idxs, bond_params, angle_idxs, angle_params, proper_idxs, proper_params, improper_idxs, improper_params
    )

    np.testing.assert_array_equal(atom_stereo, [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
    # only carbon-carbon bonds should have stereo codes
    for idx, (i,j) in enumerate(bond_idxs):
        if i < 6 and j < 6:
            assert bond_stereo[idx] == 1
        else:
            assert bond_stereo[idx] == 0 
    
    print(atom_geometries)
    print(atom_stereo)
    print(bond_stereo)
    print(bond_idxs)

def test_single_carboxylic_acid():
    # mol_a = Chem.AddHs(Chem.MolFromSmiles("FC(=O)O"))
    # mol_b = Chem.AddHs(Chem.MolFromSmiles("FC(Cl)(Br)N"))

    # mol_a = Chem.AddHs(Chem.MolFromSmiles("FC(Cl)(Br)N"))
    # mol_b = Chem.AddHs(Chem.MolFromSmiles("FC(=O)O"))

    # core = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1F"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1N"))

    # core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    st = single_topology.SingleTopologyV2(mol_a, mol_b, core, ff)
    st.get_mol_b_dummy_anchor_ixns()

def test_ring_opening():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("C=CC=CC=CF"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1N"))
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = single_topology.SingleTopologyV2(mol_a, mol_b, core, ff)
    st.get_mol_b_dummy_anchor_ixns()


def test_ring_opening_extra_map():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("C=CC=CC=CF"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C"))
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    st = single_topology.SingleTopologyV2(mol_a, mol_b, core, ff)
    st.get_mol_b_dummy_anchor_ixns()


def test_enumerate_anchor_groups():

    #                                                          1  1 1
    #                          0 12  3  4    5   6  7   8  9   0  1 2
    config = Chem.SmilesParserParams()
    config.removeHs = False
    mol = Chem.MolFromSmiles(r"[H]OC1=C([H])\[N+](C([H])=C1[H])=C(\F)Cl", config)
    bond_idxs = utils.get_romol_bonds(mol)
    core_idxs = [1, 2, 3, 5, 6, 8]
    enumerate_fn = functools.partial(single_topology.enumerate_anchor_groups, bond_idxs=bond_idxs, core_idxs=core_idxs)

    nbs_1, nbs_2 = enumerate_fn(1)
    assert nbs_1 == set([2])
    assert nbs_2 == set([(2, 3), (2, 8)])

    nbs_1, nbs_2 = enumerate_fn(2)
    assert nbs_1 == set([1, 3, 8])
    assert nbs_2 == set([(3, 5), (8, 6)])

    nbs_1, nbs_2 = enumerate_fn(3)
    assert nbs_1 == set([2, 5])
    assert nbs_2 == set([(2, 8), (2, 1), (5, 6)])

    nbs_1, nbs_2 = enumerate_fn(5)
    assert nbs_1 == set([3, 6])
    assert nbs_2 == set([(3, 2), (6, 8)])

    nbs_1, nbs_2 = enumerate_fn(6)
    assert nbs_1 == set([5, 8])
    assert nbs_2 == set([(5, 3), (8, 2)])

    nbs_1, nbs_2 = enumerate_fn(8)
    assert nbs_1 == set([2, 6])
    assert nbs_2 == set([(2, 1), (2, 3), (6, 5)])


def test_check_stability():

    # wrong idxs
    result = single_topology.check_bond_stability(0, 1, bond_idxs=[[1, 2]], bond_params=[[1000.0, 0.2]])
    assert result == False

    # weak force constant
    result = single_topology.check_bond_stability(0, 1, bond_idxs=[[1, 0]], bond_params=[[10.0, 0.0]])
    assert result == False

    # bond length too short
    result = single_topology.check_bond_stability(0, 1, bond_idxs=[[1, 0]], bond_params=[[1000.0, 0.04]])
    assert result == False

    # okay bond
    result = single_topology.check_bond_stability(0, 1, bond_idxs=[[1, 0]], bond_params=[[1000.0, 0.1]])
    assert result == True

    # wrong idxs
    result = single_topology.check_angle_stability(0, 1, 2, angle_idxs=[[1, 2, 0]], angle_params=[[100.0, 0.5]])
    assert result == False

    # weak force constant
    result = single_topology.check_angle_stability(0, 1, 2, angle_idxs=[[0, 1, 2]], angle_params=[[10.0, 1.2]])
    assert result == False

    # collapsed
    result = single_topology.check_angle_stability(0, 1, 2, angle_idxs=[[0, 1, 2]], angle_params=[[100.0, 0.0]])
    assert result == False

    # linear
    result = single_topology.check_angle_stability(0, 1, 2, angle_idxs=[[0, 1, 2]], angle_params=[[100.0, 3.1415]])
    assert result == False

    # everything should be fine
    result = single_topology.check_angle_stability(0, 1, 2, angle_idxs=[[0, 1, 2]], angle_params=[[100.0, 1.2]])
    assert result == True


# def test_group_torsions():

#     # given a set of torsion idxs, group them into non-redundant sets.
#     ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
#     mol = Chem.MolFromSmiles(r"F\C(Br)=C\Cl")
#     params, idxs = ff.pt_handle.parameterize(mol)


#     print("params", params)
#     print("idxs", idxs)