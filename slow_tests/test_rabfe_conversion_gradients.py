# Test that we can adjust parameters to make the conversion dG correspond to
# a desired value

import numpy as np
from fe.loss import l1_loss
from optimize.step import truncated_step
from optimize.precondition import learning_rates_like_params
from optimize.utils import flatten_and_unflatten
from jax import value_and_grad
from datasets.hif2a import get_ligands, protein_pdb
from fe.model_rabfe import SolventConversion, ComplexConversion
from common import default_forcefield

# how to interact with forcefield parameters
ordered_handles = default_forcefield.get_ordered_handles()
ordered_params = default_forcefield.get_ordered_params()
flatten, unfllatten = flatten_and_unflatten(ordered_params)

# get parameter-type-specific learning rates
ordered_learning_rates = learning_rates_like_params(ordered_handles, ordered_params)
learning_rates = flatten(ordered_learning_rates)

# get a pair of molecules to run tests on
mols = get_ligands()
mol, mol_ref = mols[:2]

initial_flat_params = flatten(ordered_params)


def train(predict, x0, label, loss_fxn=l1_loss, n_epochs=10):
    # TODO: pick a naming convention to distinguish between:
    #   loss as a fxn of residuals vs. loss as a fxn of params

    def loss(params):
        residual = predict(params) - label
        return loss_fxn(residual)

    def update(x, v, g):
        # TODO: wrap up these next few lines into optimize.step...
        raw_search_direction = - g
        search_direction = raw_search_direction * learning_rates

        x_increment = truncated_step(
            x, v, g, search_direction=search_direction,
            step_lower_bound=0.5 * v,
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


def assert_trainable(predict, x0, initial_label_offset=-100, n_epochs=10):
    """test that the loss goes down

    note: want to pull the training loop out into optimize probably...
    """

    initial_prediction = predict(x0)
    label = initial_prediction + initial_label_offset
    flat_param_traj, loss_traj = train(predict, x0, label, loss_fxn=l1_loss, n_epochs=n_epochs)

    window_size = min(5, n_epochs // 2)
    before = loss_traj[0]
    after = np.median(loss_traj[-window_size:])

    assert after < before, f"before: {before:.3f}, after: {after:.3f}"


def test_rabfe_solvent_conversion_trainable():
    """test that the loss for dG_solvent in isolation goes down"""

    solvent_conversion = SolventConversion(mol, mol_ref, default_forcefield)

    def predict(params):
        return solvent_conversion.predict(unfllatten(params))

    assert_trainable(predict, initial_flat_params, 10)


def test_rabfe_complex_conversion_trainable():
    """test that the loss for dG_complex in isolation goes down"""
    complex_conversion = ComplexConversion(mol, mol_ref, protein_pdb, default_forcefield)

    def predict(params):
        return complex_conversion.predict(unfllatten(params))

    assert_trainable(predict, initial_flat_params, 10)


def test_rabfe_combined_conversion_trainable():
    """test that the loss for dG_solvent - dG_complex goes down"""
    solvent_conversion = ComplexConversion(mol, mol_ref, protein_pdb, default_forcefield)
    complex_conversion = ComplexConversion(mol, mol_ref, protein_pdb, default_forcefield)

    def predict(params):
        dG_solvent = solvent_conversion.predict(unfllatten(params))
        dG_complex = complex_conversion.predict(unfllatten(params))

        return dG_solvent - dG_complex

    assert_trainable(predict, initial_flat_params, 10)
