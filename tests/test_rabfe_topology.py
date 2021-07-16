# test topology classes used in the RABFE protocol.
import functools
from jax.config import config; config.update("jax_enable_x64", True)
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers

from fe import topology
from rdkit import Chem
import numpy as np

from timemachine.lib import potentials

def test_base_topology_conversion_ring_torsion():

    # test that the conversion protocol behaves as intended on a
    # simple linked cycle.

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_sc.py').read())
    ff = Forcefield(ff_handlers)

    # note that this molecule "incomplete", as we do not add graphical hydrogens
    mol = Chem.MolFromSmiles("CCC1CC1C1CC1")
    vanilla_mol_top = topology.BaseTopology(mol, ff)
    vanilla_torsion_params, _ = vanilla_mol_top.parameterize_proper_torsion(ff.pt_handle.params)

    mol_top = topology.BaseTopologyConversion(mol, ff)
    core_idxs = [2,3,4,5,6,7]

    conversion_torsion_params, torsion_potential = mol_top.parameterize_proper_torsion(ff.pt_handle.params, core_idxs)

    np.testing.assert_array_equal(vanilla_torsion_params, conversion_torsion_params)

    ring_group = [0,1,2,2,2,3,3,3]

    for torsion_idxs, mult, offset in zip(torsion_potential.get_idxs(), torsion_potential.get_lambda_mult(), torsion_potential.get_lambda_offset()):
        _, b, c, _ = torsion_idxs
        assert offset == 1

        if b not in core_idxs or c not in core_idxs:
            assert mult == 0
        elif ring_group[b] != ring_group[c]:
            assert mult == -1
        else:
            assert mult == 0

    vanilla_qlj_params, _ = vanilla_mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)
    qlj_params, nonbonded_potential = mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    assert isinstance(nonbonded_potential, potentials.NonbondedInterpolated)

    src_qlj_params = qlj_params[:len(qlj_params)//2]
    dst_qlj_params = qlj_params[len(qlj_params)//2:]

    np.testing.assert_array_equal(vanilla_qlj_params, src_qlj_params)
    np.testing.assert_array_equal(topology.standard_qlj_typer(mol), dst_qlj_params)

    
def test_base_topology_conversion_r_group():

    # check that phenol torsions are turned off if they're defined in the core
    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_sc.py').read())
    ff = Forcefield(ff_handlers)
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    mol_top = topology.BaseTopologyConversion(mol, ff)

    core_idxs = np.arange(mol.GetNumAtoms())

    result, potential = mol_top.parameterize_proper_torsion(ff.pt_handle.params, core_idxs)
    # in the conversion phase, torsions that bridge the two rings should be set to
    # be alchemically turned off.
    is_in_ring = [1,1,1,1,1,1,0,0]

    for torsion_idxs, mult, offset in zip(potential.get_idxs(), potential.get_lambda_mult(), potential.get_lambda_offset()):
        _, b, c, _ = torsion_idxs
        assert offset == 1
        if is_in_ring[b] and is_in_ring[c]:
            # should be always turned on
            assert mult == 0
        elif (not is_in_ring[b]) or (not is_in_ring[c]):
            assert mult == -1

def test_base_topology_standard_decoupling():

    # this class is typically used in the second step of the RABFE protocol for the solvent leg.
    # we expected the charges to be zero, and the lj parameters to be standardized. In addition,
    # the torsions should be turned off.
    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_sc.py').read())
    ff = Forcefield(ff_handlers)
    mol = Chem.AddHs(Chem.MolFromSmiles("CCc1ccccc1O"))
    vanilla_mol_top = topology.BaseTopology(mol, ff)
    vanilla_torsion_params, _ = vanilla_mol_top.parameterize_proper_torsion(ff.pt_handle.params)

    core_idxs = [2,3,4,5,6,7,8]

    mol_top = topology.BaseTopologyStandardDecoupling(mol, ff)

    mol_top.parameterize_proper_torsion = functools.partial(mol_top.parameterize_proper_torsion, core_idxs=core_idxs)

    decouple_torsion_params, torsion_potential = mol_top.parameterize_proper_torsion(ff.pt_handle.params)

    np.testing.assert_array_equal(vanilla_torsion_params, decouple_torsion_params)

    is_in_ring = [0,0,1,1,1,1,1,1,0,0]

    _, combined_torsion_potential = mol_top.parameterize_periodic_torsion(
        ff.pt_handle.params,
        ff.it_handle.params
    )

    assert len(combined_torsion_potential.get_lambda_mult()) == len(combined_torsion_potential.get_idxs())
    assert len(combined_torsion_potential.get_lambda_mult()) == len(combined_torsion_potential.get_lambda_offset())

    # impropers should always be turned on.
    num_proper_torsions = len(torsion_potential.get_idxs())

    for idx, (torsion_idxs, mult, offset) in enumerate(zip(
        combined_torsion_potential.get_idxs(),
        combined_torsion_potential.get_lambda_mult(),
        combined_torsion_potential.get_lambda_offset())):
        _, b, c, _ = torsion_idxs

        if idx < num_proper_torsions:
            # proper torsion
            assert mult == 0
            if b not in core_idxs or c not in core_idxs:
                assert offset == 1
            elif is_in_ring[b] and is_in_ring[c]:
                # ring bond
                assert offset == 1
            else:
                # not ring bond
                assert offset == 0
        else:
            # improper torsion
            assert offset == 1

    qlj_params, nonbonded_potential = mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    assert not isinstance(nonbonded_potential, potentials.NonbondedInterpolated)

    np.testing.assert_array_equal(topology.standard_qlj_typer(mol), qlj_params)

    np.testing.assert_array_equal(nonbonded_potential.get_lambda_plane_idxs(), np.zeros(mol.GetNumAtoms(), dtype=np.int32))
    np.testing.assert_array_equal(nonbonded_potential.get_lambda_offset_idxs(), np.ones(mol.GetNumAtoms(), dtype=np.int32))

def test_dual_topology_standard_decoupling():

    # this class is used in double decoupling stages of the RABFE protocol. It modifies the
    # DualTopology class in two ways:
    # 1) the torsions between non-ring atoms are turned off
    # 2) the nonbonded terms are standardized, but also interpolated at lambda=0 such that
    #    the epsilons are at half strength.

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_sc.py').read())
    ff = Forcefield(ff_handlers)
    mol_a = Chem.AddHs(Chem.MolFromSmiles("FOc1ccccc1O"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("FOc1ccccc1F"))
    mol_c = Chem.CombineMols(mol_a, mol_b)
    mol_top = topology.DualTopologyStandardDecoupling(mol_a, mol_b, ff)

    core_idxs_a = [2,3,4,5,6,7,8]
    core_idxs_b = [2,3,4,5,6,7]  # note the missing F

    mol_top.parameterize_proper_torsion = functools.partial(mol_top.parameterize_proper_torsion,
        core_idxs_a=core_idxs_a,
        core_idxs_b=core_idxs_b
    )

    decouple_torsion_params, torsion_potential = mol_top.parameterize_proper_torsion(ff.pt_handle.params)

    # np.testing.assert_array_equal(vanilla_torsion_params, decouple_torsion_params)
    #             F O C C C C C C O H H H H H H  F O C C C C C C F H H H H H H
    is_in_ring = [0,0,1,1,1,1,1,1,0,0,0,0,0,0,0, 0,0,1,1,1,1,1,1,0,0,0,0,0,0,0]


    combined_decouple_torsion_params, combined_torsion_potential = mol_top.parameterize_periodic_torsion(
        ff.pt_handle.params,
        ff.it_handle.params,
    )

    assert len(combined_torsion_potential.get_lambda_mult()) == len(combined_torsion_potential.get_idxs())
    assert len(combined_torsion_potential.get_lambda_mult()) == len(combined_torsion_potential.get_lambda_offset())

    NA = mol_a.GetNumAtoms()

    # impropers should always be turned on. as amides/esters are kept planar with impropers
    num_proper_torsions = len(torsion_potential.get_idxs())

    for idx, (torsion_idxs, mult, offset) in enumerate(zip(
        combined_torsion_potential.get_idxs(),
        combined_torsion_potential.get_lambda_mult(),
        combined_torsion_potential.get_lambda_offset())):

        if idx < num_proper_torsions:

            _, b, c, _ = torsion_idxs

            if b < NA and c < NA:
                if b not in core_idxs_a or c not in core_idxs_a:
                    continue
            elif b >= NA and c >= NA:
                if b not in core_idxs_b or c not in core_idxs_b:
                    continue
            else:
                # bad torsion
                assert 0

            # we must now be between two core atoms if we're reached this line.

            assert mult == 0
            if is_in_ring[b] and is_in_ring[c]:
                assert offset == 1
            else:
                assert offset == 0

        else:
            assert offset == 1

    qlj_params, nonbonded_potential = mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    assert isinstance(nonbonded_potential, potentials.NonbondedInterpolated)

    expected_qlj = topology.standard_qlj_typer(mol_c)
    expected_qlj[:, 2] = expected_qlj[:, 2]/2 # eps should be halved

    src_qlj_params = qlj_params[:len(qlj_params)//2]
    dst_qlj_params = qlj_params[len(qlj_params)//2:]

    np.testing.assert_array_equal(src_qlj_params, expected_qlj)

    expected_qlj = topology.standard_qlj_typer(mol_c)

    np.testing.assert_array_equal(dst_qlj_params, expected_qlj)

    combined_lambda_plane_idxs = nonbonded_potential.get_lambda_plane_idxs()
    combined_lambda_offset_idxs = nonbonded_potential.get_lambda_offset_idxs()

    A = mol_a.GetNumAtoms()
    B = mol_b.GetNumAtoms()
    C = mol_c.GetNumAtoms()

    np.testing.assert_array_equal(combined_lambda_plane_idxs, np.zeros(C))
    np.testing.assert_array_equal(combined_lambda_offset_idxs[:A], np.zeros(A))
    np.testing.assert_array_equal(combined_lambda_offset_idxs[A:], np.ones(B))

def test_dual_topology_minimization():

    # Identical to the vanilla Dual Topology class, except that both ligands are
    # decouple simultaneously

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_sc.py').read())
    ff = Forcefield(ff_handlers)
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1F"))
    mol_top = topology.DualTopologyMinimization(mol_a, mol_b, ff)

    C = mol_a.GetNumAtoms() + mol_b.GetNumAtoms()

    _, nonbonded_potential = mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    assert not isinstance(nonbonded_potential, potentials.NonbondedInterpolated)

    np.testing.assert_array_equal(nonbonded_potential.get_lambda_offset_idxs(), np.ones(C, dtype=np.int32))
    np.testing.assert_array_equal(nonbonded_potential.get_lambda_plane_idxs(), np.zeros(C, dtype=np.int32))


def test_dual_topology_rhfe():

    # used in testing the relative hydration protocol. The nonbonded charges and epsilons are reduced
    # to half strength

    ff_handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_sc.py').read())
    ff = Forcefield(ff_handlers)
    mol_a = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1F"))
    mol_c = Chem.CombineMols(mol_a, mol_b)
    mol_top = topology.DualTopologyRHFE(mol_a, mol_b, ff)

    C = mol_a.GetNumAtoms() + mol_b.GetNumAtoms()

    ref_qlj_params, _ = topology.BaseTopology(mol_c, ff).parameterize_nonbonded(
        ff.q_handle.params,
        ff.lj_handle.params
    )

    qlj_params, nonbonded_potential = mol_top.parameterize_nonbonded(ff.q_handle.params, ff.lj_handle.params)

    assert isinstance(nonbonded_potential, potentials.NonbondedInterpolated)

    src_qlj_params = qlj_params[:len(qlj_params)//2]
    dst_qlj_params = qlj_params[len(qlj_params)//2:]

    np.testing.assert_array_equal(src_qlj_params[:, 0], ref_qlj_params[:, 0]/2)
    np.testing.assert_array_equal(src_qlj_params[:, 1], ref_qlj_params[:, 1])
    np.testing.assert_array_equal(src_qlj_params[:, 2], ref_qlj_params[:, 2]/2)
    np.testing.assert_array_equal(dst_qlj_params, ref_qlj_params)

    combined_lambda_plane_idxs = nonbonded_potential.get_lambda_plane_idxs()
    combined_lambda_offset_idxs = nonbonded_potential.get_lambda_offset_idxs()

    A = mol_a.GetNumAtoms()
    B = mol_b.GetNumAtoms()
    C = mol_c.GetNumAtoms()

    np.testing.assert_array_equal(combined_lambda_plane_idxs, np.zeros(C))
    np.testing.assert_array_equal(combined_lambda_offset_idxs[:A], np.zeros(A))
    np.testing.assert_array_equal(combined_lambda_offset_idxs[A:], np.ones(B))
