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
from timemachine.integrator import langevin_coefficients
from timemachine import constants
from rdkit import Chem
from rdkit.Chem import AllChem
import asciiplotlib as apl

import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group
from ff.handlers.deserialize import deserialize_handlers
from ff import Forcefield

from testsystems import relative

def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm

def setup_system():

    # pair = relative.hif2a_ligand_pair
    # mol_a, mol_b = pair.mol_a, pair.mol_b

    mol_a = Chem.MolFromSmiles("C1CC1")
    mol_b = Chem.MolFromSmiles("C1CC1")

    AllChem.EmbedMolecule(mol_a)
    AllChem.EmbedMolecule(mol_b)

    core_idxs = np.array([
        [0,3],
        [1,4],
        [2,5]
    ])

    # load the molecule
    handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    forcefield = Forcefield(handlers)

    potentials = []

    # parameterize with bonds and angles
    bond_params_a, bond_idxs_a = forcefield.hb_handle.parameterize(mol_a)
    angle_params_a, angle_idxs_a = forcefield.ha_handle.parameterize(mol_a)

    bond_params_b, bond_idxs_b = forcefield.hb_handle.parameterize(mol_b)
    angle_params_b, angle_idxs_b = forcefield.ha_handle.parameterize(mol_b)

    harmonic_bond_fn = functools.partial(
        bonded.harmonic_bond,
        bond_idxs=np.concatenate([bond_idxs_a, bond_idxs_b+mol_a.GetNumAtoms()]),
        params=np.concatenate([bond_params_a, bond_params_b])
    )

    harmonic_angle_fn = functools.partial(
        bonded.harmonic_angle,
        angle_idxs=np.concatenate([angle_idxs_a, angle_idxs_b+mol_a.GetNumAtoms()]),
        params=np.concatenate([angle_params_a, angle_params_b])
    )

    # center of mass translational restraints
    restr_group_idxs_a = core_idxs[:, 0]
    restr_group_idxs_b = core_idxs[:, 1]

    translation_restr_kb = 200.0
    translation_restr_b0 = 0.0
    translation_restr = functools.partial(
        bonded.centroid_restraint,
        group_a_idxs=restr_group_idxs_a,
        group_b_idxs=restr_group_idxs_b,
        params=None,
        masses=None,
        kb=translation_restr_kb,
        b0=translation_restr_b0,
        lamb_offset=0.0,
        lamb_mult=1.0,
    )

    # (ytz):
    rotation_restr_kb = 25.0

    rotation_restr = functools.partial(
        bonded.rmsd_restraint,
        params=None,
        group_a_idxs=np.array(restr_group_idxs_a, dtype=np.int32),
        group_b_idxs=np.array(restr_group_idxs_b, dtype=np.int32),
        k=rotation_restr_kb,
        lamb_offset=0.0,
        lamb_mult=1.0
    )

    # inertia-based rotational restraints
    # potentials.append(rotation_restr)

    def u_fn(x, box, lamb):
        return harmonic_bond_fn(x, box=box, lamb=lamb) + harmonic_angle_fn(x, box=box, lamb=lamb) + translation_restr(x, box=box, lamb=lamb) + rotation_restr(x, box=box, lamb=lamb)

    return u_fn, translation_restr_kb, restr_group_idxs_a, restr_group_idxs_b, mol_a, mol_b

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

def run(trial, pool):

    u_fn, translation_restr_kb, restr_group_idxs_a, restr_group_idxs_b, mol_a, mol_b = setup_system()

    combined_mass = np.concatenate([
        [a.GetMass() for a in mol_a.GetAtoms()],
        [b.GetMass() for b in mol_b.GetAtoms()]
    ])

    combined_conf = np.concatenate([
        get_romol_conf(mol_a),
        get_romol_conf(mol_b) # just to deal with numerical jank
    ])


    print(combined_conf)


    # restrained potential
    du_dx_fn = jax.jit(jax.grad(u_fn, argnums=(0,)))
    du_dl_fn = jax.jit(jax.grad(u_fn, argnums=(2,)))
    # u_fn = jax.jit(u_fn)

    temperature = 300.0
    dt = 1.5e-3
    friction = 1.0

    ca, cb, cc = langevin_coefficients(300.0, dt, friction, combined_mass)
    cb = -1*np.expand_dims(cb, axis=-1)
    cc = np.expand_dims(cc, axis=-1)

    x_t = combined_conf
    v_t = np.zeros_like(x_t)

    NA = mol_a.GetNumAtoms()

    writer = Chem.SDWriter("overlap_data/trial_"+str(trial)+"_md.sdf")

    box = None

    n_steps = 100000001
    equilibrium_steps = 20000
    sampling_frequency = 1000

    du_dls = []
    lamb = 1.0

    start = time.time()

    for step in range(n_steps):
        du_dx = du_dx_fn(x_t, box, lamb)[0]
        # print(du_dx)
        # print(u_fn(x_t, box, lamb))
        # assert 0
        v_t = ca*v_t + cb*du_dx + cc*np.random.normal(size=x_t.shape)
        x_t = x_t + v_t*dt
        if step % sampling_frequency == 0 and step > equilibrium_steps:
            # lhs_du.append(delta_u_jit(x_t, box, lamb))
            du_dls.append(du_dl_fn(x_t, box, lamb)[0])
            print(step, np.mean(du_dls), np.std(du_dls))
            writer.write(make_conformer(mol_a, mol_b, x_t))
            fig = asciiplotlib.figure()
            fig.hist(*np.histogram(du_dls), orientation="horizontal", force_ascii=False)
            fig.show()

    assert 0

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(24)
    for trial in range(100):
        run(trial, pool)