import asciiplotlib

import multiprocessing
import time

# endpoint correction
import pymbar
import functools
import numpy as np
import jax
import jax.numpy as jnp
from timemachine.potentials import bonded
from timemachine.potentials.nonbonded import nonbonded_v3
from timemachine.integrator import langevin_coefficients
from timemachine import constants
from rdkit import Chem
from rdkit.Chem import AllChem
import asciiplotlib as apl

import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group
from ff.handlers.deserialize import deserialize_handlers
from ff.handlers.nonbonded import generate_exclusion_idxs
from ff import Forcefield

from testsystems import relative

def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm

def setup_system():

    pair = relative.hif2a_ligand_pair
    mol = pair.mol_a

    # load the molecule
    handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    forcefield = Forcefield(handlers)

    potentials = []

    # parameterize with bonds and angles
    bond_params, bond_idxs = forcefield.hb_handle.parameterize(mol)
    angle_params, angle_idxs = forcefield.ha_handle.parameterize(mol)
    lj_params = np.array(forcefield.lj_handle.parameterize(mol))
    q_params = np.zeros(mol.GetNumAtoms(), dtype=np.float64).reshape((-1, 1))

    qlj_params = np.concatenate([q_params, lj_params], axis=1)

    harmonic_bond_fn = functools.partial(
        bonded.harmonic_bond,
        bond_idxs=bond_idxs,
        params=bond_params,
        box=None,
    )

    harmonic_angle_fn = functools.partial(
        bonded.harmonic_angle,
        angle_idxs=angle_idxs,
        params=angle_params,
        box=None
    )

    _SCALE_12 = 1.0
    _SCALE_13 = 1.0
    _SCALE_14 = 0.5

    exclusion_idxs, scales = generate_exclusion_idxs(
        mol,
        _SCALE_12,
        _SCALE_13,
        _SCALE_14
    )

    scales = np.stack([scales, scales], axis=1)

    N = mol.GetNumAtoms()

    charge_rescale_mask = np.ones((N, N))
    for (i,j), exc in zip(exclusion_idxs, scales[:, 0]):
        charge_rescale_mask[i][j] = 1 - exc
        charge_rescale_mask[j][i] = 1 - exc

    lj_rescale_mask = np.ones((N, N))
    for (i,j), exc in zip(exclusion_idxs, scales[:, 1]):
        lj_rescale_mask[i][j] = 1 - exc
        lj_rescale_mask[j][i] = 1 - exc

    nonbonded_fn = functools.partial(
        nonbonded_v3,
        params=qlj_params,
        charge_rescale_mask=charge_rescale_mask,
        lj_rescale_mask=lj_rescale_mask,
        beta=2.0,
        cutoff=None,
        lambda_plane_idxs=None,
        lambda_offset_idxs=None
    )

    # inertia-based rotational restraints
    # potentials.append(rotation_restr)

    def u_fn(x, box, lamb):
        return harmonic_bond_fn(x, box=box, lamb=lamb) + harmonic_angle_fn(x, box=box, lamb=lamb) + nonbonded_fn(x, box=box, lamb=lamb)

    return u_fn, mol

def make_conformer(mol_a, mol_b, conf_c):

    mol_a = Chem.Mol(mol_a)
    mol_b = Chem.Mol(mol_b)

    """Remove all of mol's conformers, make a new mol containing two copies of mol,
    assign positions to each copy using conf_a and conf_b, respectively, assumed in nanometers"""
    assert conf_c.shape[0] == mol_a.GetNumAtoms() + mol_b.GetNumAtoms()
    mol_a.RemoveAllConformers()
    mol_b.RemoveAllConformers()
    mol = Chem.CombineMols(mol_a, mol_b)
    cc = Chem.Conformer(mol.GetNumAtoms())
    conf = np.copy(conf_c)
    conf *= 10  # TODO: label this unit conversion?
    for idx, pos in enumerate(np.asarray(conf)):
        cc.SetAtomPosition(idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(cc)

    return mol

def run(val):
    lamb_idx, lamb = val

    u_fn, mol = setup_system()

    mass = np.array([a.GetMass() for a in mol.GetAtoms()])
    conf = get_romol_conf(mol)

    # restrained potential
    du_dx_fn = jax.jit(jax.grad(u_fn, argnums=(0,)))
    du_dl_fn = jax.jit(jax.grad(u_fn, argnums=(2,)))

    temperature = 300.0
    dt = 1.5e-3
    friction = 1.0

    ca, cb, cc = langevin_coefficients(300.0, dt, friction, mass)
    cb = -1*np.expand_dims(cb, axis=-1)
    cc = np.expand_dims(cc, axis=-1)

    x_t = conf
    v_t = np.zeros_like(x_t)

    # NA = mol_a.GetNumAtoms()

    writer = Chem.SDWriter("multidimensional/trial_"+str(lamb_idx)+"_md.sdf")

    box = None

    n_steps = 50000
    equilibrium_steps = 10000
    sampling_frequency = 100

    du_dls = []

    start = time.time()

    for step in range(n_steps):
        du_dx = du_dx_fn(x_t, box, lamb)[0]

        v_t = ca*v_t + cb*du_dx + cc*np.random.normal(size=x_t.shape)
        x_t = x_t + v_t*dt
        if step % sampling_frequency == 0 and step > equilibrium_steps:
            # rmsds.append(np.linalg.norm(x_t[restr_group_idxs_a] - x_t[restr_group_idxs_b]))
            # lhs_du.append(delta_u_jit(x_t, box, lamb))
            du_dls.append(du_dl_fn(x_t, box, lamb)[0])
            print("lambda", lamb, step, "<du/dl>", np.mean(du_dls), "std(du/dl)", np.std(du_dls))
            # writer.write(make_conformer(mol_a, mol_b, x_t))
            # fig = asciiplotlib.figure()
            # fig.hist(*np.histogram(du_dls, bins=25), orientation="horizontal", force_ascii=False)
            # fig.show()

    return du_dls

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(24)

    args = []
    # (ytz): Be sure to sanity check the the region between 0 and 0.05
    # as it can hide a lot of singularities
    for lamb_idx, lamb in enumerate(np.linspace(0.0, 1.0, 200)):
        args.append((lamb_idx, lamb))

    results = pool.map(run, args)

    xs = []
    ys = []
    for du_dls, (lamb_idx, lamb) in zip(results, args):
        xs.append(lamb)
        ys.append(np.mean(du_dls))
        print(lamb, np.mean(du_dls), np.std(du_dls))

    fig = apl.figure()
    fig.plot(xs, ys, label="TI", width=50, height=15)
    fig.show()