import functools

import numpy as np
from rdkit import Chem

from timemachine.fe import single_topology, utils
from timemachine.ff import Forcefield

# def test_single_topology():
#     print("testing")

#     mol_a = Chem.AddHs(Chem.MolFromSmiles("FC1CCC1"))
#     mol_b = Chem.AddHs(Chem.MolFromSmiles("C1CC(F)(C#N)C1Br"))
#     core = np.array([[1, 0], [2, 1], [3, 2], [4, 6]])

#     st = single_topology.SingleTopologyV2(mol_a, mol_b, core)
#     st.add_restraints_src()


def test_single_carboxylic_acid():
    print("testing")
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
    st.add_restraints_src()


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
