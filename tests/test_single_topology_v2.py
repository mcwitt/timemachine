import functools

import numpy as np
import pytest
from rdkit import Chem

from timemachine.fe import geometry, single_topology, topology, utils
from timemachine.fe.single_topology import setup_orientational_restraints, setup_end_state, SingleTopologyV2
from timemachine.ff import Forcefield
from timemachine.integrator import simulate
from timemachine.fe import dummy
from timemachine.potentials import bonded
from timemachine.fe import pdb_writer, utils

import functools
import jax
import scipy
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem



def minimize_bfgs(x0, U_fn):
    N = x0.shape[0]
    def u_bfgs(x_flat):
        x_full = x_flat.reshape(N, 3)
        return U_fn(x_full)

    grad_bfgs_fn = jax.grad(u_bfgs)
    res = scipy.optimize.minimize(u_bfgs, x0.reshape(-1), jac=grad_bfgs_fn)
    xi = res.x.reshape(N,3)
    return xi

def simulate_idxs_and_params(idxs_and_params, x0):

    (bond_idxs, bond_params), (angle_idxs, angle_params), (proper_idxs, proper_params), (improper_idxs, improper_params), (x_angle_idxs, x_angle_params), (c_angle_idxs, c_angle_params) = idxs_and_params

    box = None
    bond_U = functools.partial(bonded.harmonic_bond, params=bond_params, box=box, lamb=0.0, bond_idxs=bond_idxs)
    angle_U = functools.partial(bonded.harmonic_angle, params=angle_params,box=box, lamb=0.0, angle_idxs=angle_idxs)
    proper_U = functools.partial(bonded.periodic_torsion, params=proper_params, box=box, lamb=0.0, torsion_idxs=proper_idxs)
    improper_U = functools.partial(bonded.periodic_torsion, params=improper_params, box=box, lamb=0.0, torsion_idxs=improper_idxs)
    c_angle_U = functools.partial(bonded.harmonic_c_angle, params=c_angle_params, box=box, lamb=0.0, angle_idxs=c_angle_idxs)
    x_angle_U = functools.partial(bonded.harmonic_x_angle, params=x_angle_params, box=box, lamb=0.0, angle_idxs=x_angle_idxs)

    def U_fn(x):
        return bond_U(x) + angle_U(x) + proper_U(x) + improper_U(x) + c_angle_U(x) + x_angle_U(x)

    num_atoms = x0.shape[0]

    x_min = minimize_bfgs(x0, U_fn)
    num_workers = 1
    num_batches = 2000
    print("starting md...")
    frames = simulate(x_min, U_fn, 300.0, np.ones(num_atoms)*4.0, 1000, num_batches, num_workers)
    # discard burn in
    # burn_in_batches = num_batches//10
    burn_in_batches = 0
    frames = frames[:, burn_in_batches:, :, :]
    # collect over all workers
    frames = frames.reshape(-1, num_atoms, 3)

    return frames


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

def measure_chiral_volume(x0, x1, x2, x3):
    
    # x0 is the center
    # x1, x2, x3 are the atoms of interest
    v0 = x1 - x0
    v1 = x2 - x0
    v2 = x3 - x0

    v0 = v0/np.linalg.norm(v0)
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)

    return np.dot(np.cross(v0, v1), v2)


def test_halomethyl_to_halomethylamine():
    # test that we preserve stereochemistry when morphing from SP3->SP3"
    mol_a = Chem.MolFromMolBlock("""
  Mrv2202 05192216353D          

  5  4  0  0  0  0            999 V2000
    0.3495    0.4000   -0.7530 C   0  0  2  0  0  0  0  0  0  0  0  0
   -0.5582   -0.1718    0.8478 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -0.8566    1.4388   -1.8390 Br  0  0  0  0  0  0  0  0  0  0  0  0
    0.9382   -1.1442   -1.7440 Br  0  0  0  0  0  0  0  0  0  0  0  0
    2.0273    1.5851   -0.2289 I   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$""", removeHs=False)

    mol_b = Chem.MolFromMolBlock("""
  Mrv2202 05192216363D          

  7  6  0  0  0  0            999 V2000
    0.3495    0.4000   -0.7530 C   0  0  2  0  0  0  0  0  0  0  0  0
   -0.5582   -0.1718    0.8478 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -0.8566    1.4388   -1.8390 Br  0  0  0  0  0  0  0  0  0  0  0  0
    0.9382   -1.1442   -1.7440 Br  0  0  0  0  0  0  0  0  0  0  0  0
    2.0273    1.5851   -0.2289 N   0  0  0  0  0  0  0  0  0  0  0  0
    2.2242    3.2477   -0.2289 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.1718    0.5547   -0.2289 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
  5  6  1  0  0  0  0
  5  7  1  0  0  0  0
M  END
$$$$""", removeHs=False)

    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    s_top = SingleTopologyV2(mol_a, mol_b, core, ff)

    x_a = utils.get_romol_conf(mol_a)
    x_b = utils.get_romol_conf(mol_b)
    x0 = s_top.combine_confs(x_a, x_b)

    # check initial chirality
    vol_a = measure_chiral_volume(x0[0], x0[1], x0[2], x0[4])
    vol_d = measure_chiral_volume(x0[0], x0[1], x0[2], x0[5])
    assert vol_a < 0 and vol_d < 0

    # re-enable visualization is desired
    # writer = pdb_writer.PDBWriter([mol_a, mol_b], "dummy_mol_a.pdb")
    idxs_and_params = s_top.generate_end_state_mol_a()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    for f in frames:
        vol_a = measure_chiral_volume(f[0], f[1], f[2], f[4])
        vol_d = measure_chiral_volume(f[0], f[1], f[2], f[5])
        # we should not be be inverting chirality here
        assert vol_a < 0 and vol_d < 0
        # new_x = pdb_writer.convert_single_topology_mols(f, s_top)
        # new_x = new_x - np.mean(new_x, axis=0)
        # writer.write_frame(new_x*10)
    # writer.close()

    idxs_and_params = s_top.generate_end_state_mol_b()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    for f in frames:
        vol_a = measure_chiral_volume(f[0], f[1], f[2], f[4])
        vol_d = measure_chiral_volume(f[0], f[1], f[2], f[5])
        assert vol_a < 0 and vol_d < 0


def test_halomethyl_to_halomethylamine_inverted():
    # test that we preserve stereochemistry when morphing from SP3->SP3, except
    # the nitrogen is assigned an alternative chirality

    mol_a = Chem.MolFromMolBlock("""
  Mrv2202 05192216353D          

  5  4  0  0  0  0            999 V2000
    0.3495    0.4000   -0.7530 C   0  0  2  0  0  0  0  0  0  0  0  0
   -0.5582   -0.1718    0.8478 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -0.8566    1.4388   -1.8390 Br  0  0  0  0  0  0  0  0  0  0  0  0
    0.9382   -1.1442   -1.7440 Br  0  0  0  0  0  0  0  0  0  0  0  0
    2.0273    1.5851   -0.2289 I   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$""", removeHs=False)

    mol_b = Chem.MolFromMolBlock("""
  Mrv2202 05192216593D          

  7  6  0  0  0  0            999 V2000
   -0.0814    0.0208   -1.3024 C   0  0  1  0  0  0  0  0  0  0  0  0
   -0.0096    0.1615    0.6181 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -1.6626    0.9097   -1.9529 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -0.1350   -1.8376   -1.8089 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -1.1201   -0.6482    0.3122 N   0  0  1  0  0  0  0  0  0  0  0  0
   -2.2975   -0.5188    1.0529 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3037   -1.9164    0.9197 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
  5  6  1  0  0  0  0
  5  7  1  0  0  0  0
M  END
$$$$""", removeHs=False)

    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    s_top = SingleTopologyV2(mol_a, mol_b, core, ff)

    x_a = utils.get_romol_conf(mol_a)
    x_b = utils.get_romol_conf(mol_b)
    x0 = s_top.combine_confs(x_a, x_b)

    # check initial chirality
    vol_a = measure_chiral_volume(x0[0], x0[1], x0[2], x0[4])
    vol_d = measure_chiral_volume(x0[0], x0[1], x0[2], x0[5])
    assert vol_a < 0 and vol_d > 0

    idxs_and_params = s_top.generate_end_state_mol_a()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    for f in frames:
        vol_a = measure_chiral_volume(f[0], f[1], f[2], f[4])
        vol_d = measure_chiral_volume(f[0], f[1], f[2], f[5])
        assert vol_a < 0 and vol_d > 0

    idxs_and_params = s_top.generate_end_state_mol_b()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    for f in frames:
        vol_a = measure_chiral_volume(f[0], f[1], f[2], f[4])
        vol_d = measure_chiral_volume(f[0], f[1], f[2], f[5])
        assert vol_a < 0 and vol_d > 0


def test_ammonium_to_chloromethyl():
    # NH3 easily interconverts between the two chiral states. In the event that we
    # morph NH3 to something that is actually chiral, we should still be able to
    # ensure enantiopurity of the end-states.
    # we expect the a-state to be:
    #   mixed stereo on vol_a, fixed stereo on vol_b
    # we expect the b-state to be:
    #   fixed stereo on vol_a, fixed stereo on vol_b

    mol_a = Chem.MolFromMolBlock("""
  Mrv2202 05192218063D          

  4  3  0  0  0  0            999 V2000
   -0.0541    0.5427   -0.3433 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.4368    0.0213    0.3859 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9636    0.0925   -0.4646 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.4652    0.3942   -1.2109 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
M  END
$$$$""", removeHs=False)

    mol_b = Chem.MolFromMolBlock("""
  Mrv2202 05192218063D          

  5  4  0  0  0  0            999 V2000
   -0.0541    0.5427   -0.3433 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4368    0.0213    0.3859 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9636    0.0925   -0.4646 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.4652    0.3942   -1.2109 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0541    2.0827   -0.3433 Cl  0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$""", removeHs=False)


    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    s_top = SingleTopologyV2(mol_a, mol_b, core, ff)

    x_a = utils.get_romol_conf(mol_a)
    x_b = utils.get_romol_conf(mol_b)
    x0 = s_top.combine_confs(x_a, x_b)

    # check initial chirality
    vol_a = measure_chiral_volume(x0[0], x0[1], x0[2], x0[3])
    vol_d = measure_chiral_volume(x0[0], x0[1], x0[2], x0[4])
    print(vol_a, vol_d)
    assert vol_a > 0 and vol_d < 0

    idxs_and_params = s_top.generate_end_state_mol_a()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    num_vol_a_pos = 0
    num_vol_a_neg = 0

    for f in frames:
        vol_a = measure_chiral_volume(f[0], f[1], f[2], f[3])
        vol_d = measure_chiral_volume(f[0], f[1], f[2], f[4])
        # ammonium is freely invertible
        if vol_a < 0:
            num_vol_a_neg += 1
        else:
            num_vol_a_pos += 1
        # but stereochemistry of the dummy is still enforced
        assert vol_d < 0
    
    # writer.close()

    # should be within 5% of 50/50
    assert abs(num_vol_a_pos/len(frames) - 0.5) < 0.05

    idxs_and_params = s_top.generate_end_state_mol_b()
    frames = simulate_idxs_and_params(idxs_and_params, x0)

    for f in frames:
        vol_a = measure_chiral_volume(f[0], f[1], f[2], f[3])
        vol_d = measure_chiral_volume(f[0], f[1], f[2], f[4])
        assert vol_a > 0 and vol_d < 0


def test_phenol_end_state_max_core():
    # test morphing of phenol using the largest core possible

    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))

    # do not allow H to map to O
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6,7], [7,8], [8,9], [9,10], [10,11]])
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    s_top = SingleTopologyV2(mol_a, mol_b, core, ff)
    idxs_and_params = s_top.generate_end_state_mol_a()

    x0 = np.random.rand(s_top.get_num_atoms(), 3)
    s_top = SingleTopologyV2(mol_a, mol_b, core, ff)
    idxs_and_params = s_top.generate_end_state_mol_a()
    frames = simulate_idxs_and_params(idxs_and_params, x0)
    writer = pdb_writer.PDBWriter([mol_a, mol_b], "dummy.pdb")

    for f in frames:
        new_x = pdb_writer.convert_single_topology_mols(f, s_top)
        new_x = new_x - np.mean(new_x, axis=0)
        writer.write_frame(new_x*10)
    
    writer.close()

    # visualize


    assert 0

    

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

    # simulate(x0, U_)

def test_hif2a_set():

    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")
    suppl = Chem.SDMolSupplier("timemachine/testsystems/data/ligands_40.sdf", removeHs=False)
    mols = list(suppl)
    for idx, mol_a in enumerate(mols):
        for mol_b in mols[idx+2:]:

            # if mol_a.GetProp("_Name") != "235" or mol_b.GetProp("_Name") != "165":
                # continue

            print(mol_a.GetProp("_Name"), "->", mol_b.GetProp("_Name"))
            core = single_topology.find_core(mol_a, mol_b)
            s_top = SingleTopologyV2(mol_a, mol_b, core, ff)

            # mol_a must be embedded
            AllChem.EmbedMolecule(mol_a)
            x0 = single_topology.embed_molecules(mol_a, mol_b, s_top)

            writer = pdb_writer.PDBWriter([mol_a, mol_b], "dummy_mol_a.pdb")
            idxs_and_params = s_top.generate_end_state_mol_a()
            frames = simulate_idxs_and_params(idxs_and_params, x0)

            vols_a = []
            for f in frames:
                vol = measure_chiral_volume(f[0], f[5], f[6], f[7])
                vols_a.append(vol)
                new_x = pdb_writer.convert_single_topology_mols(f, s_top)
                new_x = new_x - np.mean(new_x, axis=0)
                writer.write_frame(new_x*10)
            writer.close()

            plt.hist(vols_a, label="vols_a", alpha=0.5)

            writer = pdb_writer.PDBWriter([mol_a, mol_b], "dummy_mol_b.pdb")
            idxs_and_params = s_top.generate_end_state_mol_b()
            frames = simulate_idxs_and_params(idxs_and_params, x0)
            vols_b = []
            for f in frames:
                vol = measure_chiral_volume(f[0], f[5], f[6], f[7])
                vols_b.append(vol)
                new_x = pdb_writer.convert_single_topology_mols(f, s_top)
                new_x = new_x - np.mean(new_x, axis=0)
                writer.write_frame(new_x*10)


            plt.hist(vols_b, label="vols_b", alpha=0.5)
            plt.xlim(-1, 1)
            plt.legend()
            plt.show()

            writer.close()

            assert 0

            U_fn_lambda_1 = s_top.generate_end_state_mol_b()

            # def U_fn_lambda(conf, params, box, lambda):
                # return (1-lambda)*U_fn_lambda_0(conf, params, box) + lambda*U_fn_lambda_0(conf, params, box)

import cProfile