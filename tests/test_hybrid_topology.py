NUM_WORKERS = 8
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(NUM_WORKERS)

from jax.config import config
from timemachine import constants
import pymbar
from matplotlib import pyplot as plt

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


def get_restraints(mol, core, k):

    bond_idxs = []
    bond_params = []

    for idx in range(mol.GetNumAtoms()):
        # dummy atom
        if idx not in core:
            anchor_idxs = identify_anchors(mol, core, idx)
            assert len(anchor_idxs) == 1
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

    def parameterize_dummy_restraints(self, lamb, k):
        a_bond_idxs, a_bond_params = get_restraints(self.mol_a, self.core[:, 0], lamb * k)
        b_bond_idxs, b_bond_params = get_restraints(self.mol_b, self.core[:, 1], (1 - lamb) * k)

        # this is to deal with zero sized arrays jank
        a_bond_idxs = np.array(a_bond_idxs).astype(np.int32).reshape((-1, 2))
        a_bond_params = np.array(a_bond_params).astype(np.float64).reshape((-1, 2))
        b_bond_idxs = np.array(b_bond_idxs).astype(np.int32).reshape((-1, 2))
        b_bond_params = np.array(b_bond_params).astype(np.float64).reshape((-1, 2))

        bond_idxs, bond_params = self.combine_bonds(a_bond_idxs, a_bond_params, b_bond_idxs, b_bond_params)

        return bond_idxs, bond_params

    def parameterize_bonded(self, params, handle, lamb):
        params_a, idxs_a = handle.partial_parameterize(params, self.mol_a)
        params_b, idxs_b = handle.partial_parameterize(params, self.mol_b)

        new_params_a = []
        for k, b in params_a:
            new_params_a.append([(1 - lamb) * k, b])

        new_params_b = []
        for k, b in params_b:
            new_params_b.append([lamb * k, b])

        new_params_a = np.array(new_params_a)
        new_params_b = np.array(new_params_b)

        combined_idxs, combined_params = self.combine_bonds(idxs_a, new_params_a, idxs_b, new_params_b)
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


def get_U_fn(ht, ff, lamb, k_core, k_dummy):

    new_bond_idxs, new_bond_params = ht.parameterize_bonded(ff.hb_handle.params, ff.hb_handle, lamb)
    new_angle_idxs, new_angle_params = ht.parameterize_bonded(ff.ha_handle.params, ff.ha_handle, lamb)
    new_dummy_idxs, new_dummy_params = ht.parameterize_dummy_restraints(lamb, k_dummy)
    new_core_idxs, new_core_params = ht.parameterize_core_restraints(k_core)

    box = np.eye(3) * 100

    harmonic_bond_U = functools.partial(
        bonded.harmonic_bond,
        bond_idxs=new_bond_idxs,
        box=box,
        params=new_bond_params,
        lamb=0.0,
    )

    harmonic_angle_U = functools.partial(
        bonded.harmonic_angle, angle_idxs=new_angle_idxs, box=box, params=new_angle_params, lamb=0.0
    )

    # core restraints are left on, and can be turned into constraints
    core_bond_U = functools.partial(
        bonded.harmonic_bond, bond_idxs=new_core_idxs, box=box, params=new_core_params, lamb=0.0
    )

    # dummy restraints are alchemically turned on/off, and cannot be made into constraints
    dummy_bond_U = functools.partial(
        bonded.harmonic_bond, bond_idxs=new_dummy_idxs, box=box, params=new_dummy_params, lamb=0.0
    )

    def U_fn(x):
        return harmonic_bond_U(x) + harmonic_angle_U(x) + core_bond_U(x) + dummy_bond_U(x)

    return U_fn


def run_simulation(mol_a, mol_b, core, k_core, k_dummy):

    ff_handlers = deserialize_handlers(open("ff/params/smirnoff_1_1_0_sc.py").read())
    ff = Forcefield(ff_handlers)

    ht = HybridTopology(mol_a, mol_b, ff, core)

    lambda_schedule = np.linspace(0.0, 1.0, 24)

    U_fns = []
    batch_U_fns = []

    for lamb in lambda_schedule:
        U_fn = get_U_fn(ht, ff, lamb, k_core, k_dummy)
        U_fns.append(U_fn)
        batch_U_fns.append(jax.vmap(U_fn))

    temperature = 300.0
    beta = 1 / (constants.BOLTZ * temperature)
    N_ks = []
    all_coords = []

    for idx, U_fn in tqdm(enumerate(U_fns)):

        x0 = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)])

        masses = np.concatenate([get_romol_masses(mol_a), get_romol_masses(mol_b)])

        num_batches = 1000
        coords = integrator.simulate(
            x0, U_fn, temperature, masses, steps_per_batch=500, num_batches=num_batches, num_workers=NUM_WORKERS
        )
        # coords = coords.squeeze(0)
        coords = coords.reshape(-1, x0.shape[0], 3)
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

        dG, dG_err = pymbar.BAR(fwd, rev)
        dG_errs.append(dG_err)
        dG_estimate += dG

        print(idx, "->", idx + 1, dG / beta, dG_err / beta)  # in kJ

    # plt.show()

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

    print("pbar", pbar_estimate, "+=", pbar_err, "kJ/mol | mbar", mbar_estimate, "kJ/mol")

    return dG_estimate / beta


from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def setup_relative_restraints_by_distance(mol_a: Chem.Mol, mol_b: Chem.Mol, cutoff: float = 0.1, terminal: bool = True):

    ligand_coords_a = get_romol_conf(mol_a)
    ligand_coords_b = get_romol_conf(mol_b)
    core_idxs_a = []
    core_idxs_b = []

    for idx, a in enumerate(mol_a.GetAtoms()):
        if not terminal and a.GetDegree() == 1:
            continue
        for b_idx, b in enumerate(mol_b.GetAtoms()):
            if not terminal and b.GetDegree() == 1:
                continue
            if np.linalg.norm(ligand_coords_a[idx] - ligand_coords_b[b_idx]) < cutoff:
                core_idxs_a.append(idx)
                core_idxs_b.append(b_idx)
    assert len(core_idxs_a) == len(core_idxs_b), "Core sizes were inconsistent"

    rij = cdist(ligand_coords_a[core_idxs_a], ligand_coords_b[core_idxs_b])

    row_idxs, col_idxs = linear_sum_assignment(rij)

    core_idxs = []

    for core_a, core_b in zip(row_idxs, col_idxs):
        core_idxs.append((core_idxs_a[core_a], core_idxs_b[core_b]))

    core_idxs = np.array(core_idxs, dtype=np.int32)

    return core_idxs


def test_hybrid_topology():

    suppl = Chem.SDMolSupplier("tests/data/ligands_40.sdf")
    mols = [mol for mol in suppl]
    for idx in range(len(mols)):
        for jdx in range(idx + 1, len(mols)):

            mol_a = mols[idx]
            mol_b = mols[jdx]
            # mol_a = hybrid_mols.A
            # mol_b = hybrid_mols.B

            try:
                core = setup_relative_restraints_by_distance(mol_a, mol_b)
                print(core)

                # core = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

                print("testing...", idx, jdx, "names", mol_a.GetProp("_Name"), mol_b.GetProp("_Name"))
                run_simulation(mol_a, mol_b, core, k_core=100000, k_dummy=100000)
                run_simulation(mol_a, mol_b, core, k_core=100000, k_dummy=100000)
                run_simulation(mol_a, mol_b, core, k_core=100000, k_dummy=100000)
                run_simulation(mol_a, mol_b, core, k_core=100000, k_dummy=100000)
                run_simulation(mol_a, mol_b, core, k_core=100000, k_dummy=100000)

            except Exception as e:
                print("failed", idx, jdx, e)
