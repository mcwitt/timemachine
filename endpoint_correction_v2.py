import multiprocessing
import time

import pymbar
import functools
import numpy as np
import jax
import jax.numpy as jnp
from timemachine.potentials import bonded, pmi
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


TEMPERATURE = 300.0
BETA = 1/(constants.BOLTZ*TEMPERATURE)


def get_romol_conf(mol):
    """Coordinates of mol's 0th conformer, in nanometers"""
    conformer = mol.GetConformer(0)
    guest_conf = np.array(conformer.GetPositions(), dtype=np.float64)
    return guest_conf/10 # from angstroms to nm

def setup_system():

    pair = relative.hif2a_ligand_pair
    mol_a, mol_b = pair.mol_a, pair.mol_b

    # load the molecule
    handlers = deserialize_handlers(open('ff/params/smirnoff_1_1_0_ccc.py').read())
    forcefield = Forcefield(handlers)

    potentials = []

    # parameterize with bonds and angles
    bond_params_a, bond_idxs_a = forcefield.hb_handle.parameterize(mol_a)
    angle_params_a, angle_idxs_a = forcefield.ha_handle.parameterize(mol_a)

    bond_params_b, bond_idxs_b = forcefield.hb_handle.parameterize(mol_b)
    angle_params_b, angle_idxs_b = forcefield.ha_handle.parameterize(mol_b)

    box = np.eye(3)*100.0

    bond_fn = functools.partial(
        bonded.harmonic_bond,
        bond_idxs=np.concatenate([bond_idxs_a, bond_idxs_b+mol_a.GetNumAtoms()]),
        params=np.concatenate([bond_params_a, bond_params_b]),
        box=box,
        lamb=None
    )

    angle_fn = functools.partial(
        bonded.harmonic_angle,
        angle_idxs=np.concatenate([angle_idxs_a, angle_idxs_b+mol_a.GetNumAtoms()]),
        params=np.concatenate([angle_params_a, angle_params_b]),
        box=box,
        lamb=None
    )

    # (ytz): pair.core is a singleton, do not modify in-place
    core_idxs = np.array(pair.core)
    core_idxs[:, 1] += mol_a.GetNumAtoms()

    # if I have a million bonds, each with a one kcal/mol restraint, what happens..?
    # maybe we can reduce strength of this by # of bonds present...
    core_params = np.array([
        [30.0, 0.0],
    ]*core_idxs.shape[0])

    core_restr = functools.partial(
        bonded.harmonic_bond,
        bond_idxs=core_idxs,
        params=core_params,
        box=box,
        lamb=None
    )

    potentials.append(core_restr)

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
        kb=translation_restr_kb,
        b0=translation_restr_b0,
        box=box,
        lamb=None
    )

    potentials.append(translation_restr)

    # (ytz):
    rotation_restr_kb = 100.0
    # never generate a force with this
    rotation_restr = functools.partial(
        bonded.rmsd_restraint,
        params=None,
        group_a_idxs=np.array(restr_group_idxs_a, dtype=np.int32),
        group_b_idxs=np.array(restr_group_idxs_b, dtype=np.int32),
        k=rotation_restr_kb,
        box=box,
        lamb=None
    )

    potentials.append(rotation_restr)

    def delta_u_fn(x_t):
        return translation_restr(x_t) + rotation_restr(x_t) - core_restr(x_t)

    # left hand state has intractable restraints turned on.
    def u_lhs_fn(x_t):
        return bond_fn(x_t) + angle_fn(x_t) + core_restr(x_t)

    # right hand state is post-processed from independent gas phase simulations
    def u_rhs_fn(x_t):
        return bond_fn(x_t) + angle_fn(x_t)

    return u_lhs_fn, u_rhs_fn, delta_u_fn, translation_restr_kb, rotation_restr_kb, restr_group_idxs_a, restr_group_idxs_b, mol_a, mol_b

# use for visualization in pymol
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

def exp_u(rotation, k):
    return jnp.exp(-BETA*bonded.psi(rotation, k))

exp_batch = jax.jit(jax.vmap(exp_u, (0, None)))

def sample_rotation(k):
    # generate rotations in batches for rejection sampling.
    num_batches = 500
    batch_size = 100

    for batch_attempt in range(num_batches):
        Rs = special_ortho_group.rvs(3, size=batch_size)
        tests = np.random.rand(batch_size)
        M = np.pi**2 # volume of SO(3)
        acceptance_prob = exp_batch(Rs, k)/M
        locations = np.argwhere(tests < acceptance_prob).reshape(-1)
        if len(locations) > 0:
            return Rs[locations[0]]

    raise Exception("Failed to Sample Rotation")

def run(trial):

    u_lhs_fn, u_rhs_fn, delta_u_fn, translation_restr_kb, rotation_restr_kb, restr_group_idxs_a, restr_group_idxs_b, mol_a, mol_b = setup_system()

    combined_mass = np.concatenate([
        [a.GetMass() for a in mol_a.GetAtoms()],
        [b.GetMass() for b in mol_b.GetAtoms()]
    ])

    combined_conf = np.concatenate([
        get_romol_conf(mol_a),
        get_romol_conf(mol_b)
    ])

    lhs_du_dx_fn = jax.jit(jax.grad(u_lhs_fn))
    delta_u_fn = jax.jit(delta_u_fn)

    dt = 1.5e-3
    friction = 1.0

    ca, cb, cc = langevin_coefficients(TEMPERATURE, dt, friction, combined_mass)
    cb = -1*np.expand_dims(cb, axis=-1)
    cc = np.expand_dims(cc, axis=-1)

    x_t = combined_conf
    v_t = np.zeros_like(x_t)

    NA = mol_a.GetNumAtoms()

    writer_core = Chem.SDWriter("overlap_data/trial_"+str(trial)+"_core.sdf")
    writer_mc = Chem.SDWriter("overlap_data/trial_"+str(trial)+"_monte_carlo.sdf")

    box = None

    n_steps = 500001
    # n_steps = 500
    equilibrium_steps = 20000
    # equilibrium_steps = 100
    sampling_frequency = 100

    lhs_du = []
    start = time.time()
    for step in range(n_steps):
        du_dx = lhs_du_dx_fn(x_t)
        v_t = ca*v_t + cb*du_dx + cc*np.random.normal(size=x_t.shape)
        x_t = x_t + v_t*dt
        if step % sampling_frequency == 0 and step > equilibrium_steps:
            lhs_du.append(delta_u_fn(x_t))
            writer_core.write(make_conformer(mol_a, mol_b, x_t))

    print("lhs time", time.time()-start)

    lhs_du = np.array(lhs_du)

    rhs_du = []

    # reset x and v
    x_t = combined_conf
    v_t = np.zeros_like(x_t)

    rhs_samples = []
    rmsds = []
    rhs_du_dx_fn = jax.jit(jax.grad(u_rhs_fn))

    start = time.time()
    for step in range(n_steps):
        du_dx = rhs_du_dx_fn(x_t)
        v_t = ca*v_t + cb*du_dx + cc*np.random.normal(size=x_t.shape)
        x_t = x_t + v_t*dt

        if step % sampling_frequency == 0 and step > equilibrium_steps:
            # align the two conformers
            x_a, x_b = bonded.rmsd_align(x_t, restr_group_idxs_a, restr_group_idxs_b, mol_a.GetNumAtoms())
            covariance = np.eye(3)/(2*BETA*translation_restr_kb)
            translation = np.random.multivariate_normal((0,0,0), covariance)
            # if sign is positive, then x_b is inverted
            rotation = sample_rotation(rotation_restr_kb)
            x_t_new = np.concatenate([x_a, x_b@rotation.T + translation])
            rmsds.append(np.mean(np.linalg.norm(x_t_new[restr_group_idxs_a] - x_b[restr_group_idxs_b], axis=1)))
            writer_mc.write(make_conformer(mol_a, mol_b, x_t_new))
            rhs_du.append(delta_u_fn(x_t_new))

    plt.clf()
    plt.title("BAR")
    plt.hist(np.array(lhs_du), alpha=0.5, label='forward', density=True)
    plt.hist(np.array(rhs_du), alpha=0.5, label='backward', density=True)
    plt.legend()
    plt.savefig("overlap_data/trial_"+str(trial)+"_over_lap.png")

    print("rhs time", time.time()-start)
    print("avg rmsd", np.mean(rmsds))
    dG = pymbar.BAR(BETA*lhs_du, -BETA*np.array(rhs_du))[0]/BETA
    print("trial", trial, "step", step, "dG estimate", dG, "msd", np.mean(rmsds))

    return

if __name__ == "__main__":
    # for trial in range(100):
        # run(trial)
    # assert 0
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(12)
    args = np.arange(100)
    pool.map(run, args, chunksize=1)