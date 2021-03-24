import multiprocessing
import time

# endpoint correction
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

    potentials.append(functools.partial(
        bonded.harmonic_bond,
        bond_idxs=np.concatenate([bond_idxs_a, bond_idxs_b+mol_a.GetNumAtoms()]),
        params=np.concatenate([bond_params_a, bond_params_b])
    ))

    potentials.append(functools.partial(
        bonded.harmonic_angle,
        angle_idxs=np.concatenate([angle_idxs_a, angle_idxs_b+mol_a.GetNumAtoms()]),
        params=np.concatenate([angle_params_a, angle_params_b])
    ))

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
        params=core_params
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
        masses=None,
        kb=translation_restr_kb,
        b0=translation_restr_b0
    )

    potentials.append(translation_restr)

    # rotation_restr = functools.partial(
    #     pmi.inertial_restraint,
    #     params=None,
    #     a_idxs=np.array(restr_group_idxs_a, dtype=np.int32),
    #     b_idxs=np.array(restr_group_idxs_b, dtype=np.int32),
    #     masses=None,
    #     k=restr_kb,
    #     lamb_offset=0.0,
    #     lamb_mult=1.0
    # )

    # (ytz):
    rotation_restr_kb = 25.0
    # never generate a force with this
    rotation_restr = functools.partial(
        bonded.rmsd_restraint,
        params=None,
        group_a_idxs=np.array(restr_group_idxs_a, dtype=np.int32),
        group_b_idxs=np.array(restr_group_idxs_b, dtype=np.int32),
        k=rotation_restr_kb
    )

    # inertia-based rotational restraints
    potentials.append(rotation_restr)

    return potentials, translation_restr_kb, restr_group_idxs_a, restr_group_idxs_b, mol_a, mol_b


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

def exp_U(conf, box, lamb, beta, u_fn):
    u = u_fn(conf, box=box, lamb=lamb)
    return jnp.exp(-beta*u)

def sample_rotation_batched(x, box, lamb, NA, b_idxs, exp_U_restr_batched):
    """
    Batched version of sample_rotation. See below for documentation.
    """
    offset = jnp.mean(x[b_idxs], axis=0)
    x_a = x[:NA]
    x_b = x[NA:]
    x_b = x[NA:] - offset

    num_batches = 500
    batch_size = 1000
    x_a_batch = np.tile(x_a, (batch_size, 1)).reshape(batch_size, NA, 3)

    for batch_attempt in range(num_batches):
        Rs = special_ortho_group.rvs(3, size=batch_size)
        tests = np.random.rand(batch_size)
        x_b_batch = x_b@np.transpose(Rs, axes=(0,2,1))
        M = np.pi**2 # volume of SO(3)
        acceptance_prob = exp_U_restr_batched(np.concatenate([x_a_batch, x_b_batch], axis=1), box, lamb)/M
        locations = np.argwhere(tests < acceptance_prob).reshape(-1)
        if len(locations) > 0:
            return x_b_batch[locations[0]]

    raise Exception("Failed to sample a rotation")

def sample_rotation(x, box, lamb, NA, b_idxs, exp_U_restr):
    """
    Sample a rotation from the boltzmann distribution of exp_U_restr. This function implements rejection
    sampling for f(x)<=Mg(x). f(x) is an unnormalized target distribution, M is a normalization constant
    needed to satsify the inequality, and g(x) is sampled uniformly from SO(3).
    """
    offset = jnp.mean(x[b_idxs], axis=0)
    x_a = x[:NA]
    for attempt in range(50000):
        R = special_ortho_group.rvs(3)
        x_b = x[NA:]
        x_b = x_b - offset
        x_b = x_b@R.T
        M = np.pi**2 # volume of SO(3)
        acceptance_prob = exp_U_restr(np.concatenate([x_a, x_b]), box, lamb)/M
        test = np.random.rand()
        if test < acceptance_prob:
            print("accepted on attempt", attempt)
            return x_b

    raise Exception("Failed to sample a rotation")

def process_sample(x_t, translation_restr_kb, NA, box, lamb, restr_group_idxs_a, restr_group_idxs_b, beta, exp_U, rotation_restr, delta_u_partial):
    exp_U_restr_rotation_batched = jax.jit(jax.vmap(functools.partial(exp_U, beta=beta, u_fn=rotation_restr), (0, None, None)))
    delta_u = jax.jit(delta_u_partial)
    x_a = x_t[:NA]
    com_a = np.mean(x_t[restr_group_idxs_a], axis=0)
    x_b_r = sample_rotation_batched(x_t, box, lamb, NA, restr_group_idxs_b, exp_U_restr_rotation_batched)
    covariance = np.eye(3)/(2*beta*translation_restr_kb)
    translation = np.random.multivariate_normal((0,0,0), covariance)
    x_b_n = x_b_r + com_a + translation
    x_new = np.concatenate([x_a, x_b_n])
    du = delta_u(x_new, box, lamb)
    return x_new, du


def delta_u(conf, box, lamb, translation_restr, rotation_restr, core_restr):
    """
    U_rot_res - U_core_restr
    """
    return translation_restr(conf=conf, box=box, lamb=lamb) + rotation_restr(conf=conf, box=box, lamb=lamb) - core_restr(conf=conf, box=box, lamb=lamb) 


def run(trial, pool):

    potentials, translation_restr_kb, restr_group_idxs_a, restr_group_idxs_b, mol_a, mol_b = setup_system()

    core_restr = potentials[-3]
    translation_restr = potentials[-2]
    rotation_restr = potentials[-1]

    # with core-restraints only
    def u_fn(conf, box, lamb):
        nrgs = []
        for p in potentials[:-2]:
            nrgs.append(p(conf=conf, box=box, lamb=lamb))
        return jnp.sum(jnp.array(nrgs))

    # no restraints period
    def u_nr_fn(conf, box, lamb):
        nrgs = []
        for p in potentials[:-3]:
            nrgs.append(p(conf=conf, box=box, lamb=lamb))
        return jnp.sum(jnp.array(nrgs))

    combined_mass = np.concatenate([
        [a.GetMass() for a in mol_a.GetAtoms()],
        [b.GetMass() for b in mol_b.GetAtoms()]
    ])

    combined_conf = np.concatenate([
        get_romol_conf(mol_a),
        get_romol_conf(mol_b)
    ])

    # restrained potential
    du_dx_fn = jax.jit(jax.grad(u_fn, argnums=(0,)))
    u_fn = jax.jit(u_fn)

    # unrestrained potential
    du_dx_nr_fn = jax.jit(jax.grad(u_nr_fn, argnums=(0,)))
    u_nr_fn = jax.jit(u_nr_fn)

    temperature = 300.0
    dt = 1.5e-3
    friction = 1.0

    ca, cb, cc = langevin_coefficients(300.0, dt, friction, combined_mass)
    cb = -1*np.expand_dims(cb, axis=-1)
    cc = np.expand_dims(cc, axis=-1)

    x_t = combined_conf
    v_t = np.zeros_like(x_t)

    NA = mol_a.GetNumAtoms()

    writer_core = Chem.SDWriter("overlap_data/trial_"+str(trial)+"_core.sdf")
    writer = Chem.SDWriter("overlap_data/trial_"+str(trial)+"_monte_carlo.sdf")
    writer_gas = Chem.SDWriter("overlap_data/trial_"+str(trial)+"_gas.sdf")

    box = None

    n_steps = 100001
    equilibrium_steps = 10000
    sampling_frequency = 50

    lhs_du = []
    lamb = 0.0

    delta_u_partial = functools.partial(delta_u,
        translation_restr=translation_restr,
        rotation_restr=rotation_restr,
        core_restr=core_restr
    )

    delta_u_jit = jax.jit(delta_u_partial)

    start = time.time()
    for step in range(n_steps):
        du_dx = du_dx_fn(x_t, box, lamb)[0]
        v_t = ca*v_t + cb*du_dx + cc*np.random.normal(size=x_t.shape)
        x_t = x_t + v_t*dt
        if step % sampling_frequency == 0 and step > equilibrium_steps:
            lhs_du.append(delta_u_jit(x_t, box, lamb))
            writer_core.write(make_conformer(mol_a, mol_b, x_t))
    print("lhs time", time.time()-start)

    lhs_du = np.array(lhs_du)

    beta = 1/(constants.BOLTZ*temperature)

    # no restraints variant
    lamb = 1.0
    rhs_du = []

    # reset x and v
    x_t = combined_conf
    v_t = np.zeros_like(x_t)

    exp_U_restr_rotation = jax.jit(functools.partial(exp_U, beta=beta, u_fn=rotation_restr))
    exp_U_restr_translation = jax.jit(functools.partial(exp_U, beta=beta, u_fn=translation_restr))

    rhs_samples = []

    start = time.time()
    for step in range(n_steps):
        du_dx = du_dx_nr_fn(x_t, box, lamb)[0]
        v_t = ca*v_t + cb*du_dx + cc*np.random.normal(size=x_t.shape)
        x_t = x_t + v_t*dt
        if step % sampling_frequency == 0 and step > equilibrium_steps:
            rhs_samples.append(x_t)

    print("rhs time", time.time()-start)

    process_fn = functools.partial(
        process_sample,
        translation_restr_kb=translation_restr_kb,
        NA=NA,
        box=box,
        lamb=lamb,
        restr_group_idxs_a=restr_group_idxs_a,
        restr_group_idxs_b=restr_group_idxs_b,
        beta=beta,
        exp_U=exp_U,
        rotation_restr=rotation_restr,
        delta_u_partial=delta_u_partial
        # exp_U_restr_rotation_batched=exp_U_restr_rotation_batched
    )

    start_time = time.time()
    results = pool.map(process_fn, rhs_samples)
    print("correction time", time.time() - start_time)

    start_time = time.time()
    for (x_new, du) in results:

        if False:
            # if step % sampling_frequency == 0 and step > equilibrium_steps:
            x_a = x_t[:NA]
            # (ytz): we can move this out of the MD loop if needed and batch post-process this.
            com_a = np.mean(x_t[restr_group_idxs_a], axis=0)
            # slow version is commented out
            x_b_r = sample_rotation_batched(x_t, box, lamb, NA, restr_group_idxs_b, exp_U_restr_rotation_batched)

            # we can sample the translational component directly from a multivariable gaussian
            # exp(-beta*U(x,y,z)) = exp(-beta*k*(dx^2 + dy^2 + dz^2)) = exp(-0.5*sig^2*(dx^2+dy^2+dz^2))
            # => beta*k = 0.5*sig^2
            # => sig^2 = 2*beta*k
            covariance = np.eye(3)/(2*beta*translation_restr_kb)
            translation = np.random.multivariate_normal((0,0,0), covariance)
            x_b_n = x_b_r + com_a + translation
            x_new = np.concatenate([x_a, x_b_n])


            writer_gas.write(make_conformer(mol_a, mol_b, x_t))

        writer.write(make_conformer(mol_a, mol_b, x_new))

        rhs_du.append(du)

    # plot the overlap
    plt.clf()
    plt.title("BAR")
    plt.hist(np.array(lhs_du), bins=np.arange(-150.0, 10.0, 2.0), alpha=0.5, label='forward', density=True)
    plt.hist(np.array(rhs_du), bins=np.arange(-150.0, 10.0, 2.0), alpha=0.5, label='backward', density=True)
    plt.legend()
    plt.savefig("overlap_data/trial_"+str(trial)+"_over_lap.png")
    dG = pymbar.BAR(beta*lhs_du, -beta*np.array(rhs_du))[0]/beta
    print("trial", trial, "step", step, "dG estimate", dG)

    # du = delta_u(x_new, box, lamb)
    if not np.isfinite(du):
        print(x)
        print("core U", core_restr(x))
        print("rotation U", rotation_restr(x))
        print("non-finite du found")
        assert 0

    # print("IO time", time.time() - start_time)
    rhs_du = np.array(rhs_du)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(24)
    for trial in range(100):
        run(trial, pool)
    # args = np.arange(100)
    # pool.map(run, args, chunksize=1)