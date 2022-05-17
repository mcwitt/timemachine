import functools

import numpy as np
import pytest
from rdkit import Chem

from timemachine.fe import geometry, single_topology, topology, utils
from timemachine.fe.single_topology import setup_orientational_restraints, setup_end_state, SingleTopologyV2
from timemachine.ff import Forcefield

def test_nblist_conversion():
    mol = Chem.MolFromSmiles("CC1CC1C(C)(C)C")
    bond_idxs = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    nblist = geometry.bond_idxs_to_nblist(mol.GetNumAtoms(), bond_idxs)

    expected = [[1], [0, 2, 3], [1, 3], [1, 2, 4], [3, 5, 6, 7], [4], [4], [4]]

    np.testing.assert_array_equal(nblist, expected)

def test_benzene_to_phenol_restraints():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))

    # do not allow H to map to O
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    # add [O,H] as the dummy-group
    all_idxs, _ = setup_orientational_restraints(ff, mol_a, mol_b, core, dg=[6, 12], anchor=5)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs

    assert set(bond_idxs) == set([(5, 6)])
    assert set(angle_idxs) == set([(5, 6, 12)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set([((0, 4), 5, 6)])

    # allow H to map to O
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [11, 6]])
    all_idxs, _ = setup_orientational_restraints(ff, mol_a, mol_b, core, dg=[12], anchor=6)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs
    assert set(bond_idxs) == set([(6, 12)])
    assert set(angle_idxs) == set([(5, 6, 12)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set()

def test_phenol_end_state_max_core():
    # test morphing of phenol using the largest core possible

    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))

    # do not allow H to map to O
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6,7], [7,8], [8,9], [9,10], [10,11]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    s_top = SingleTopologyV2(mol_a, mol_b, core, ff)
    idxs_and_params = s_top.generate_end_state_mol_a()

    bond_idxs, bond_params = idxs_and_params[0]

    assert set([tuple(idxs) for idxs in bond_idxs]) == set([
        (0, 1), # c-c
        (1, 2), # c-c
        (2, 3), # c-c
        (3, 4), # c-c
        (4, 5), # c-c
        (0, 5), # c-c
        (0, 6), # c-h
        (1, 7), # c-h
        (2, 8), # c-h
        (3, 9), # c-h
        (4, 10), # c-h
        (5, 11), # c-h
        (5, 12), # c-o, from dummy
        (12, 13),  # c-h from dummy
    ])

    angle_idxs, angle_params = idxs_and_params[1]

    assert set([tuple(idxs) for idxs in angle_idxs]) == set([
        (1,0,5), # c-c-c
        (1,0,6), # c-c-h
        (5,0,6), # c-c-h
        (0,1,7), # c-c-h
        (0,1,2), # c-c-c
        (2,1,7), # c-c-h
        (1,2,8), # c-c-h
        (3,2,8), # c-c-h
        (1,2,3), # c-c-c
        (2,3,9), # c-c-h
        (2,3,4), # c-c-c
        (4,3,9), # c-c-h
        (3,4,10), # c-c-h
        (3,4,5), # c-c-c
        (5,4,10), # c-c-h
        (0,5,4), # c-c-c 
        (0,5,11), # c-c-h
        (4,5,11), # c-c-h
        (5,12,13) # c-o-h, core-dummy-dummy
    ])

    # there are too many of these to enumerate out fully
    proper_idxs, proper_params = idxs_and_params[2]
    improper_idxs, improper_params = idxs_and_params[3]

    x_angle_idxs, x_angle_params = idxs_and_params[4]
    assert set(x_angle_idxs) == set()

    c_angle_idxs, c_angle_params = idxs_and_params[5]
    assert set(c_angle_idxs) == set([((0,4),5,12)])

    # generate the other end-state, grafting mol_a unto mol_b, note that core is flipped
    # core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [7,6], [8,7], [9,8], [10,9], [11,10]])
    # idxs_and_params = setup_end_state(ff, mol_b, mol_a, core, s_top.b_to_c, s_top.a_to_c)
    idxs_and_params = s_top.generate_end_state_mol_b()
    bond_idxs, bond_params = idxs_and_params[0]

    # note that bonds are the same
    assert set([tuple(idxs) for idxs in bond_idxs]) == set([
        (0, 1), # c-c
        (1, 2), # c-c
        (2, 3), # c-c
        (3, 4), # c-c
        (4, 5), # c-c
        (0, 5), # c-c
        (0, 6), # c-h
        (1, 7), # c-h
        (2, 8), # c-h
        (3, 9), # c-h
        (4, 10), # c-h
        (5, 11), # c-h <- this becomes the dummy
        (5, 12), # c-o
        (12, 13),  # c-h
    ])

    angle_idxs, angle_params = idxs_and_params[1]

    # angles differ by extra terms around 5 and 12
    assert set([tuple(idxs) for idxs in angle_idxs]) == set([
        (1,0,5), # c-c-c
        (1,0,6), # c-c-h
        (5,0,6), # c-c-h
        (0,1,7), # c-c-h
        (0,1,2), # c-c-c
        (2,1,7), # c-c-h
        (1,2,8), # c-c-h
        (3,2,8), # c-c-h
        (1,2,3), # c-c-c
        (2,3,9), # c-c-h
        (2,3,4), # c-c-c
        (4,3,9), # c-c-h
        (3,4,10), # c-c-h
        (3,4,5), # c-c-c
        (5,4,10), # c-c-h
        (0,5,4), # c-c-c 
        (0,5,12), # c-c-h
        (4,5,12), # c-c-h
        (5,12,13) # c-o-h
    ])

    c_angle_idxs, c_angle_params = idxs_and_params[5]
    assert set(c_angle_idxs) == set([((0,4),5,11)])

def test_benzene_to_benzoic_acid():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C(=O)O"))
    # map only the benzene
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    # note that the C-C bond in benzoic acid is rotatable, despite being SP2 hybridized at both ends
    # this is because of steric hindrance effects, the only planar non-rotatable bond is O=C-C-H
    all_idxs, _ = setup_orientational_restraints(ff, mol_a, mol_b, core, dg=[6, 7, 8, 14], anchor=5)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs

    assert set(bond_idxs) == set([(5, 6)])
    assert set(angle_idxs) == set([(5, 6, 7), (5, 6, 8)])
    assert set(proper_idxs) == set([(5, 6, 8, 14)])
    assert set(improper_idxs) == set([(6, 5, 7, 8), (5, 8, 7, 6), (6, 8, 5, 7)])
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set([((0, 4), 5, 6)])

    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [11, 6]])

    # this should raise because a stereo bond is present
    with pytest.raises(AssertionError):
        setup_orientational_restraints(ff, mol_a, mol_b, core, dg=[7, 8, 14], anchor=6)


def test_benzene_to_aniline():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1N"))
    # map only the benzene
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    all_idxs, _ = setup_orientational_restraints(ff, mol_a, mol_b, core, dg=[6, 12, 13], anchor=5)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs

    assert set(bond_idxs) == set([(5, 6)])
    assert set(angle_idxs) == set([(5, 6, 13), (5, 6, 12)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set([(6, 13, 5, 12), (6, 5, 12, 13), (5, 13, 12, 6)])
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set([((0, 4), 5, 6)])

    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [11, 6]])
    all_idxs, _ = setup_orientational_restraints(ff, mol_a, mol_b, core, dg=[12, 13], anchor=6)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs
    assert set(bond_idxs) == set([(6, 12), (6, 13)])
    assert set(angle_idxs) == set([(5, 6, 13), (5, 6, 12), (12, 6, 13)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()  # disable impropers
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set()


def test_benzene_to_benzonitrile():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C#N"))

    # map only the benzene
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    all_idxs, _ = setup_orientational_restraints(ff, mol_a, mol_b, core, dg=[6, 7], anchor=5)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs

    assert set(bond_idxs) == set([(5, 6)])
    assert set(angle_idxs) == set([(5, 6, 7)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set([((0, 4), 5, 6)])

    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [11, 6]])
    all_idxs, _ = setup_orientational_restraints(ff, mol_a, mol_b, core, dg=[7], anchor=6)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs
    assert set(bond_idxs) == set([(6, 7)])
    assert set(angle_idxs) == set([(5, 6, 7)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()  # disable impropers
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set()

def test_benzene_to_toluene():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C"))

    # map hydrogen to carbon
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [11, 6]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    all_idxs, _ = setup_orientational_restraints(ff, mol_a, mol_b, core, dg=[12,13,14], anchor=6)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs

    assert set(bond_idxs) == set([(6, 12),(6, 13),(6, 14)])
    assert set(angle_idxs) == set([(5, 6, 12),(5, 6, 13),(5, 6, 14),(12,6,13),(12,6,14),(13,6,14)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()  # disable impropers
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set()

def test_ethanol_to_carboxylate():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("CO"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))

    core = np.array([[0, 0], [1, 1], [5,3]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    all_idxs, _ = setup_orientational_restraints(ff, mol_a, mol_b, core, dg=[2], anchor=1)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs

    assert set(bond_idxs) == set([(1,2)])
    assert set(angle_idxs) == set()
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()  # disable impropers
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set([((0,3),1,2)])

def test_ethanol_to_ammonium():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("CO"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("CN(F)Br"))

    core = np.array([[0, 0], [1, 1], [5,3]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    all_idxs, _ = setup_orientational_restraints(ff, mol_a, mol_b, core, dg=[2], anchor=1)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs

    assert set(bond_idxs) == set([(1,2)])
    assert set(angle_idxs) == set()
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()  # disable impropers
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set([((0,3),1,2)])

def test_ammonium_to_tetrahedral():
    mol_a = Chem.MolFromSmiles("N(F)(Cl)Br")
    mol_b = Chem.MolFromSmiles("C(F)(Cl)(Br)I")
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

    all_idxs, _ = setup_orientational_restraints(ff, mol_a, mol_b, core, dg=[4], anchor=0)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs
    assert set(bond_idxs) == set([(0, 4)])
    assert set(angle_idxs) == set()
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()
    assert set(x_angle_idxs) == set()
    assert set(c_angle_idxs) == set([((1, 2, 3), 0, 4)])

    core = np.array([[0, 0], [1, 1], [2, 2]])
    all_idxs, _ = setup_orientational_restraints(ff, mol_a, mol_b, core, dg=[3, 4], anchor=0)
    bond_idxs, angle_idxs, proper_idxs, improper_idxs, x_angle_idxs, c_angle_idxs = all_idxs
    assert set(bond_idxs) == set([(0, 3), (0, 4)])
    assert set(angle_idxs) == set([(3, 0, 4)])
    assert set(proper_idxs) == set()
    assert set(improper_idxs) == set()
    assert set(x_angle_idxs) == set([((0, 1), (0, 2), (0, 3)), ((0, 1), (0, 2), (0, 4))])


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

from timemachine.fe import dummy

def test_hif2a_set():

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    suppl = Chem.SDMolSupplier("timemachine/testsystems/data/ligands_40.sdf", removeHs=False)
    mols = list(suppl)
    for idx, mol_a in enumerate(mols):
        for mol_b in mols[idx+1:]:
            
            if mol_a.GetProp("_Name") != "235" or mol_b.GetProp("_Name") != "165":
                continue
            print("attempting", mol_a.GetProp("_Name"), "->", mol_b.GetProp("_Name"))
            core = single_topology.find_core(mol_a, mol_b)
            s_top = SingleTopologyV2(mol_a, mol_b, core, ff)
            s_top.generate_end_state_mol_a()
            s_top.generate_end_state_mol_b()