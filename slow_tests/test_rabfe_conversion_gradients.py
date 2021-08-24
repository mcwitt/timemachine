# Test that we can adjust parameters to make the conversion dG correspond to
# a desired value

import numpy as np
from jax import numpy as jnp
from fe.loss import l1_loss
from optimize.step import truncated_step
from optimize.precondition import learning_rates_like_params, default_learning_rates
from optimize.utils import flatten_and_unflatten
from jax import value_and_grad
from datasets.hif2a import get_ligands, protein_pdb
from fe.model_rabfe import SolventConversion, ComplexConversion
from common import default_forcefield
from parallel.client import CUDAPoolClient
from ff.handlers.nonbonded import LennardJonesHandler, AM1CCCHandler

# how to interact with forcefield parameters
ordered_handles = default_forcefield.get_ordered_handles()
ordered_params = default_forcefield.get_ordered_params()
flatten, unflatten = flatten_and_unflatten(ordered_params)
initial_flat_params = flatten(ordered_params)

# define parameter-type-specific learning rates: adjust charges only, not LJ
from copy import deepcopy
learning_rate_dict = deepcopy(default_learning_rates)
learning_rate_dict[AM1CCCHandler] = 1.0
learning_rate_dict[LennardJonesHandler] = np.zeros(2)
ordered_learning_rates = learning_rates_like_params(ordered_handles, ordered_params, learning_rate_dict)
flat_learning_rates = flatten(ordered_learning_rates)

# get a pair of molecules to run tests on
mols = get_ligands()
mol, mol_ref = mols[:2]


def train(predict, x0, label, loss_fxn=l1_loss, n_epochs=10):
    # TODO: pick a naming convention to distinguish between:
    #   loss as a fxn of residuals vs. loss as a fxn of params

    def loss(params):
        residual = predict(params) - label
        return loss_fxn(residual)

    def update(x, v, g):
        # TODO: wrap up these next few lines into optimize.step...
        raw_search_direction = - g
        search_direction = raw_search_direction * flat_learning_rates

        x_increment = truncated_step(
            x, v, g, search_direction=search_direction,
            step_lower_bound=0.8 * v,
        )
        x_next = x + x_increment

        return x_next

    flat_param_traj = [x0]
    loss_traj = [loss(x0)]
    print(f'initial loss: {loss_traj[0]:.3f}')

    for t in range(n_epochs):
        x = flat_param_traj[-1]
        v, g = value_and_grad(loss)(x)
        x_next = update(x, v, g)

        print(x_next - x)

        print(f'epoch {t}: loss = {v:.3f}, gradient norm = {np.linalg.norm(g):.3f}')

        flat_param_traj.append(x_next)
        loss_traj.append(v)

    return flat_param_traj, loss_traj


def assert_trainable(predict, x0, initial_label_offset=-25, n_epochs=10):
    """test that the loss goes down
    predict: flat_params -> dG_of_interest
        dG_of_interest might be dG_solvent, dG_complex, or dG_solvent - dG_complex
    """

    initial_prediction = predict(x0)
    label = initial_prediction + initial_label_offset
    print(f'initial prediction = {initial_prediction:.3f}')
    print(f'label = initial_prediction + {initial_label_offset:.3f} = {label:.3f}')

    flat_param_traj, loss_traj = train(predict, x0, label, loss_fxn=l1_loss, n_epochs=n_epochs)

    window_size = min(5, n_epochs // 2)
    before = loss_traj[0]
    after = np.median(loss_traj[-window_size:])

    assert after < before, f"before: {before:.3f}, after: {after:.3f}"


def construct_vector_loss(predict_a_vec, labels, loss_on_residuals=l1_loss):
    def loss_fxn(params):
        predictions = predict_a_vec(params)
        residuals = predictions - labels
        return jnp.sum(loss_on_residuals(residuals))

    return loss_fxn

def check_vector_loss_differentiable(predict_a_vec, x, labels):
    loss_fxn = construct_vector_loss(predict_a_vec, labels)
    v, g = value_and_grad(loss_fxn)(x)


def assert_trainable_with_dG_solvent_pinned(predict_both, x0, initial_label_offset=np.array([0, -25]), n_epochs=10):
    """predict_both: flat_params -> [dG_solvent, dG_solvent - dG_complex]
    """

    initial_predictions = predict_both(x0)

    label = initial_predictions + initial_label_offset
    print(f'initial predictions = {initial_predictions}')
    print(f'label = initial_predictions + {initial_label_offset} = {label}')

    loss_fxn = lambda residuals : jnp.sum(l1_loss(residuals))

    flat_param_traj, loss_traj = train(predict_both, x0, label, loss_fxn=loss_fxn, n_epochs=n_epochs)

    window_size = min(5, n_epochs // 2)
    before = loss_traj[0]
    after = np.median(loss_traj[-window_size:])

    assert after < before, f"before: {before:.3f}, after: {after:.3f}"


def test_rabfe_solvent_conversion_trainable():
    """test that the loss for dG_solvent in isolation goes down"""
    client = CUDAPoolClient(10)
    solvent_conversion = SolventConversion(mol, mol_ref, default_forcefield, client)

    def predict(params):
        return solvent_conversion.predict(unflatten(params))

    assert_trainable(predict, initial_flat_params)


def test_rabfe_complex_conversion_trainable():
    """test that the loss for dG_complex in isolation goes down"""
    client = CUDAPoolClient(10)
    complex_conversion = ComplexConversion(
        mol, mol_ref, protein_pdb, default_forcefield, client,
        num_windows=10, num_equil_steps=int(1e4), num_prod_steps=int(1e5)
    )

    def predict(params):
        return complex_conversion.predict(unflatten(params))

    assert_trainable(predict, initial_flat_params)


def test_rabfe_combined_conversion_trainable():
    #"""test that the loss for dG_solvent - dG_complex goes down"""
    """test that the loss for [dG_solvent, dG_solvent - dG_complex] goes down"""
    client = CUDAPoolClient(10)
    shared_kwargs = dict(
        mol=mol,
        mol_ref=mol_ref,
        num_windows=20,
        num_equil_steps=10000,
        num_prod_steps=500001,
        initial_forcefield=default_forcefield,
        client=client,
    )
    solvent_conversion = SolventConversion(**shared_kwargs)
    complex_conversion = ComplexConversion(
        protein_pdb=protein_pdb,
        num_preequil_steps=int(1e4),  # if int(1e5) or int(8e5) then force magnitude exceeds threshold
        **shared_kwargs
    )

    def predict(params):
        dG_solvent = solvent_conversion.predict(unflatten(params))
        dG_complex = complex_conversion.predict(unflatten(params))

        return dG_solvent - dG_complex

    #assert_trainable(predict, initial_flat_params)

    def predict_both(params):
        dG_solvent = solvent_conversion.predict(unflatten(params))
        dG_complex = complex_conversion.predict(unflatten(params))

        return jnp.array([dG_solvent, dG_solvent - dG_complex])

    #check_vector_loss_differentiable(predict_both, initial_flat_params, np.zeros(2))
    assert_trainable_with_dG_solvent_pinned(predict_both, initial_flat_params)
