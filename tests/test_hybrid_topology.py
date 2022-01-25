NUM_WORKERS = 12
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(NUM_WORKERS)

from jax.config import config
from timemachine import constants
import pymbar
from matplotlib import pyplot as plt
from rdkit.Chem import rdFMCS

config.update("jax_enable_x64", True)
import functools
import numpy as np
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from fe.utils import get_romol_conf, get_romol_masses
from timemachine.potentials import bonded
from timemachine import integrator
import jax

from rdkit import Chem
import hybrid_mols

from tqdm import tqdm


def identify_anchors(mol, core, dummy_atom):
    """
    Identify the core anchor(s) for a given atom.

    Parameters
    ----------
    mol: Chem.Mol
        rdkit molecule

    core: list or set or iterable
        core atoms

    dummy_atom: int
        atom we're initializing the search over

    """

    core = set(core)

    assert len(set(core)) == len(core)
    assert dummy_atom not in core

    # first convert to a dense graph
    N = mol.GetNumAtoms()
    dense_graph = np.zeros((N, N), dtype=np.int32)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dense_graph[i, j] = 1
        dense_graph[j, i] = 1

    # sparsify to simplify and speed up traversal code
    sparse_graph = []
    for row in dense_graph:
        nbs = []
        for col_idx, col in enumerate(row):
            if col == 1:
                nbs.append(col_idx)
        sparse_graph.append(nbs)

    def dfs(i, visited):
        if i in visited:
            return
        else:
            visited.add(i)
            if i not in core:
                for nb in sparse_graph[i]:
                    dfs(nb, visited)
            else:
                return

    visited = set()
    dfs(dummy_atom, visited)
    anchors = []

    for a_idx in visited:
        if a_idx in core:
            anchors.append(a_idx)

    return anchors


def test_identify_anchors():

    mol = Chem.MolFromSmiles("C1CCC1N")
    core = [0, 1, 2, 3]

    anchors = identify_anchors(mol, core, dummy_atom=4)
    assert set(anchors) == set([3])

    mol = Chem.MolFromSmiles("C1CC2NC2C1")
    core = [0, 1, 2, 4, 5]

    anchors = identify_anchors(mol, core, dummy_atom=3)
    assert set(anchors) == set([2, 4])

    mol = Chem.MolFromSmiles("C1OCC11CCCCC1")
    core = [3, 4, 5, 6, 7, 8]

    anchors = identify_anchors(mol, core, dummy_atom=0)
    assert set(anchors) == set([3])
    anchors = identify_anchors(mol, core, dummy_atom=1)
    assert set(anchors) == set([3])
    anchors = identify_anchors(mol, core, dummy_atom=2)
    assert set(anchors) == set([3])

    mol = Chem.MolFromSmiles("C1CC1.C1CCCCC1")
    core = [3, 4, 5, 6, 7, 8]

    anchors = identify_anchors(mol, core, dummy_atom=0)
    assert set(anchors) == set()
    anchors = identify_anchors(mol, core, dummy_atom=1)
    assert set(anchors) == set()
    anchors = identify_anchors(mol, core, dummy_atom=2)
    assert set(anchors) == set()

    mol = Chem.MolFromSmiles("C1CC2NC2C1")
    core = [0, 1, 2, 5]

    anchors = identify_anchors(mol, core, dummy_atom=3)
    assert set(anchors) == set([2, 5])
    anchors = identify_anchors(mol, core, dummy_atom=4)
    assert set(anchors) == set([2, 5])


class AnchorError(Exception):
    pass


def get_restraints(mol, core, k):

    bond_idxs = []
    bond_params = []

    for idx in range(mol.GetNumAtoms()):
        # dummy atom
        if idx not in core:
            anchor_idxs = identify_anchors(mol, core, idx)
            if len(anchor_idxs) != 1:
                raise AnchorError()
            anchor = anchor_idxs[0]
            bond_idxs.append([idx, anchor])
            bond_params.append([k, 0.0])

    return bond_idxs, bond_params


class HybridTopology:
    def __init__(self, mol_a, mol_b, forcefield, core):
        self.mol_a = mol_a
        self.mol_b = mol_b
        self.N_A = self.mol_a.GetNumAtoms()
        self.N_B = self.mol_b.GetNumAtoms()
        self.ff = forcefield
        self.core = core

    # works for both bonds and angles
    def combine_bonds(self, a_bond_idxs, a_bond_params, b_bond_idxs, b_bond_params):
        combined_idxs = np.concatenate([a_bond_idxs, b_bond_idxs + self.N_A])
        combined_params = np.concatenate([a_bond_params, b_bond_params])
        return combined_idxs, combined_params

    def _adjust_bonds(self, core, all_params, all_idxs, scale):

        # algorithm:
        # 1) core-core terms are alchemically interpolated
        # 2) core-dummy angle/torsion terms are turned off, bond terms are shrunken
        # 3) dummy-dummy terms are maintained
        new_params = []
        for params, idxs in zip(all_params, all_idxs):
            if np.all([x in core for x in idxs]):
                # core-core, turn off or on alchemically
                params[0] *= scale
            else:
                # core-dummy
                # TBD: special case, core is anchor (we can leave on)
                if len(idxs) == 2:
                    # bonded case, shrink bond length to zero
                    params[1] *= scale
                else:
                    params[0] *= scale

            new_params.append(params)

        return new_params

    def parameterize_bonded(self, params, handle, lamb):
        params_a, idxs_a = handle.partial_parameterize(params, self.mol_a)
        params_b, idxs_b = handle.partial_parameterize(params, self.mol_b)

        params_a = self._adjust_bonds(self.core[:, 0], params_a, idxs_a, 1 - lamb)
        params_b = self._adjust_bonds(self.core[:, 1], params_b, idxs_b, lamb)

        combined_idxs, combined_params = self.combine_bonds(idxs_a, params_a, idxs_b, params_b)

        return combined_idxs, combined_params

    def parameterize_urey_bradley(self, hb_params, ha_params, hb_handle, ha_handle, lamb):

        bond_params_a, bond_idxs_a = hb_handle.partial_parameterize(hb_params, self.mol_a)
        bond_params_b, bond_idxs_b = hb_handle.partial_parameterize(hb_params, self.mol_b)

        angle_params_a, angle_idxs_a = ha_handle.partial_parameterize(ha_params, self.mol_a, bond_idxs_a, bond_params_a)
        angle_params_b, angle_idxs_b = ha_handle.partial_parameterize(ha_params, self.mol_b, bond_idxs_b, bond_params_b)

        # params_a = bond_params_a
        # idxs_a = bond_idxs_a
        # params_b = bond_params_b
        # idxs_b = bond_idxs_b

        params_a = np.concatenate([bond_params_a, angle_params_a])
        idxs_a = np.concatenate([bond_idxs_a, angle_idxs_a])
        params_b = np.concatenate([bond_params_b, angle_params_b])
        idxs_b = np.concatenate([bond_idxs_b, angle_idxs_b])

        params_a = self._adjust_bonds(self.core[:, 0], params_a, idxs_a, 1 - lamb)
        params_b = self._adjust_bonds(self.core[:, 1], params_b, idxs_b, lamb)

        combined_idxs, combined_params = self.combine_bonds(idxs_a, params_a, idxs_b, params_b)
        combined_idxs, combined_params = self.combine_bonds(idxs_a, params_a, idxs_b, params_b)

        return combined_idxs, combined_params

    # tbd, turn into constraints
    def parameterize_core_restraints(self, k):
        core_idxs = []
        core_params = []
        for src, dst in self.core:
            core_idxs.append([src, dst + self.N_A])
            core_params.append([k, 0.0])
        return np.array(core_idxs), np.array(core_params)


def make_conformer(mol_a, mol_b, conf_a, conf_b):
    mol = Chem.CombineMols(mol_a, mol_b)
    mol.RemoveAllConformers()  # necessary!
    cc = Chem.Conformer(mol.GetNumAtoms())
    conf = np.concatenate([conf_a, conf_b])
    conf *= 10  # TODO: label this unit conversion?
    for idx, pos in enumerate(np.asarray(conf)):
        cc.SetAtomPosition(idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(cc)

    return mol


def get_U_fn(ht, ff, lamb, k_core):
    # new_bond_idxs, new_bond_params = ht.parameterize_bonded(ff.hb_handle.params, ff.hb_handle, lamb)
    # new_angle_idxs, new_angle_params = ht.parameterize_bonded(ff.ha_handle.params, ff.ha_handle, lamb)

    new_bond_idxs, new_bond_params = ht.parameterize_urey_bradley(
        ff.hb_handle.params, ff.ha_handle.params, ff.hb_handle, ff.ha_handle, lamb
    )

    # print(new_bond_idxs, new_bond_params)

    # assert 0

    # new_proper_torsion_idxs, new_proper_torsion_params = ht.parameterize_bonded(ff.pt_handle.params, ff.pt_handle, lamb)
    # new_improper_torsion_idxs, new_improper_torsion_params = ht.parameterize_bonded(
    #     ff.it_handle.params, ff.it_handle, lamb
    # )

    # new_dummy_idxs, new_dummy_params = ht.parameterize_dummy_restraints(lamb, k_dummy)
    new_core_idxs, new_core_params = ht.parameterize_core_restraints(k_core)

    box = np.eye(3) * 100

    harmonic_bond_U = functools.partial(
        bonded.harmonic_bond,
        bond_idxs=new_bond_idxs,
        box=box,
        params=new_bond_params,
        lamb=0.0,
    )

    # harmonic_angle_U = functools.partial(
    #     bonded.harmonic_angle,
    #     angle_idxs=new_angle_idxs,
    #     box=box,
    #     params=new_angle_params,
    #     lamb=0.0,
    # )

    # proper_torsion_U = functools.partial(
    #     bonded.periodic_torsion,
    #     torsion_idxs=new_proper_torsion_idxs,
    #     box=box,
    #     params=new_proper_torsion_params,
    #     lamb=0.0,
    # )

    # improper_torsion_U = functools.partial(
    #     bonded.periodic_torsion,
    #     torsion_idxs=new_improper_torsion_idxs,
    #     box=box,
    #     params=new_improper_torsion_params,
    #     lamb=0.0,
    # )

    # core restraints are left on, and can be turned into constraints
    core_bond_U = functools.partial(
        bonded.harmonic_bond, bond_idxs=new_core_idxs, box=box, params=new_core_params, lamb=0.0
    )

    def U_fn(x):
        # return harmonic_bond_U(x) + harmonic_angle_U(x) + core_bond_U(x) + proper_torsion_U(x) + improper_torsion_U(x)
        return harmonic_bond_U(x) + core_bond_U(x)

    return U_fn


def run_simulation(mol_a, mol_b, core, k_core, dt):

    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_sc.py").read())
    ff = Forcefield(ff_handlers)

    ht = HybridTopology(mol_a, mol_b, ff, core)

    # lambda_schedule = np.linspace(0.0, 1.0, 24)
    lambda_schedule = np.array([0.95652174, 1.0])
    # lambda_schedule = np.linspace(0.0, 1.0, 2)
    # lambda_schedule = np.array([0.0, 0.5, 1.0])
    # lambda_schedule = np.array([0.0, 0.05, 0.95, 1.0])
    # lambda_schedule = np.array([0.0, 1.0])

    U_fns = []
    batch_U_fns = []

    for lamb in lambda_schedule:
        U_fn = get_U_fn(ht, ff, lamb, k_core)
        U_fns.append(U_fn)
        batch_U_fns.append(jax.vmap(U_fn))

    temperature = 300.0
    beta = 1 / (constants.BOLTZ * temperature)
    N_ks = []
    all_coords = []

    burn_in_batches = 20
    num_batches = 1000  # total steps is 1000*NUM_WORKERS*steps_per_batch

    for idx, U_fn in enumerate(tqdm(U_fns)):

        x0 = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)])

        masses = np.concatenate([get_romol_masses(mol_a), get_romol_masses(mol_b)])
        coords = integrator.simulate(
            dt,
            x0,
            U_fn,
            temperature,
            masses,
            steps_per_batch=500,
            num_batches=num_batches + burn_in_batches,
            num_workers=NUM_WORKERS,
        )

        # toss away burn in batches and flatten
        coords = coords[:, burn_in_batches:, :, :].reshape(-1, x0.shape[0], 3)
        writer = Chem.SDWriter("out_" + str(idx) + ".sdf")
        all_coords.append(coords)
        N_ks.append(num_batches * NUM_WORKERS)

        for frame in coords:
            writer.write(make_conformer(mol_a, mol_b, frame[: mol_a.GetNumAtoms()], frame[mol_a.GetNumAtoms() :]))
        writer.close()

    u_kns = []
    N_C = mol_a.GetNumAtoms() + mol_b.GetNumAtoms()

    dG_estimate = 0
    dG_errs = []

    for idx in range(len(U_fns) - 1):

        fwd = batch_U_fns[idx + 1](all_coords[idx]) - batch_U_fns[idx](all_coords[idx])
        rev = batch_U_fns[idx](all_coords[idx + 1]) - batch_U_fns[idx + 1](all_coords[idx + 1])
        fwd *= beta
        rev *= beta

        # if idx == 0 or idx == len(U_fns) - 3 or idx == len(U_fns) - 2:
        # plt.hist(fwd, density=True, alpha=0.5, label="fwd")
        # plt.hist(-rev, density=True, alpha=0.5, label="-rev")
        # plt.legend()
        # plt.show()

        # print("fwd min/max", np.amin(fwd), np.amax(fwd))
        # print("rev min/max", np.amin(rev), np.amax(rev))
        # print("fwd nan count", np.sum(np.isnan(fwd)), "rev nan count", np.sum(np.isnan(rev)))

        dG, dG_err = pymbar.BAR(fwd, rev)
        dG_errs.append(dG_err)
        dG_estimate += dG

        print(idx, "->", idx + 1, dG / beta, dG_err / beta)  # in kJ

    dG_errs = np.array(dG_errs)

    all_coords = np.array(all_coords).reshape((-1, N_C, 3))
    for idx, U_batch in enumerate(batch_U_fns):
        reduced_nrg = U_batch(all_coords) * beta
        u_kns.append(reduced_nrg)

    u_kns = np.array(u_kns)
    N_ks = np.array(N_ks)

    obj = pymbar.MBAR(u_kns, N_k=N_ks)

    pbar_estimate = dG_estimate / beta
    pbar_err = np.linalg.norm(dG_errs) / beta
    mbar_estimate = (obj.f_k[-1] - obj.f_k[0]) / beta

    print(f"dt {dt} pair_bar {pbar_estimate:.3f} += {pbar_err:.3f} kJ/mol | mbar {mbar_estimate:.3f} kJ/mol")

    return dG_estimate / beta


class CompareDist(rdFMCS.MCSAtomCompare):
    def __init__(self, cutoff, *args, **kwargs):
        self.cutoff = cutoff * 10
        super().__init__(*args, **kwargs)

    def compare(self, p, mol1, atom1, mol2, atom2):
        x_i = mol1.GetConformer(0).GetPositions()[atom1]
        x_j = mol2.GetConformer(0).GetPositions()[atom2]
        if np.linalg.norm(x_i - x_j) > self.cutoff:
            return False
        else:
            return True


def get_core(mol_a: Chem.Mol, mol_b: Chem.Mol, cutoff: float = 0.1):

    ligand_coords_a = get_romol_conf(mol_a)
    ligand_coords_b = get_romol_conf(mol_b)

    mcs_params = rdFMCS.MCSParameters()
    mcs_params.AtomTyper = CompareDist(cutoff)
    mcs_params.BondCompareParameters.CompleteRingsOnly = 1
    mcs_params.BondCompareParameters.RingMatchesRingOnly = 1

    res = rdFMCS.FindMCS([mol_a, mol_b], mcs_params)

    query = Chem.MolFromSmarts(res.smartsString)

    mol_a_matches = mol_a.GetSubstructMatches(query, uniquify=False)
    mol_b_matches = mol_b.GetSubstructMatches(query, uniquify=False)

    best_match_dist = np.inf
    best_match_pairs = None
    for a_match in mol_a_matches:
        for b_match in mol_b_matches:
            dij = np.linalg.norm(ligand_coords_a[list(a_match)] - ligand_coords_b[list(b_match)])
            if dij < best_match_dist:
                best_match_dist = dij
                best_match_pairs = np.stack([a_match, b_match], axis=1)

    core_idxs = best_match_pairs

    assert len(core_idxs[:, 0]) == len(set(core_idxs[:, 0]))
    assert len(core_idxs[:, 1]) == len(set(core_idxs[:, 1]))

    return core_idxs


def test_hybrid_topology():

    suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf", removeHs=False)
    mols = [mol for mol in suppl]

    for i in range(len(mols)):
        for j in range(i + 1, len(mols)):

            mol_a = mols[i]
            mol_b = mols[j]

            if mol_a.GetProp("_Name") != "338" or mol_b.GetProp("_Name") != "67":
                continue

            try:
                print(
                    "trying",
                    mol_a.GetProp("_Name"),
                    mol_b.GetProp("_Name"),
                )
                core = get_core(mol_a, mol_b)

                print("testing...", mol_a.GetProp("_Name"), mol_b.GetProp("_Name"), "core size", core.size)

                # if k_core is too large we have numerical stability problems due to integrator error.
                # keep it around 250000
                run_simulation(mol_a, mol_b, core, k_core=100000, dt=1.5e-3)
                run_simulation(mol_a, mol_b, core, k_core=100000, dt=0.5e-3)
            except AnchorError:
                print("failed")

            assert 0
