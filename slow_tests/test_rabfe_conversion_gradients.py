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

# TODO: move this frequently repeated code fragment for fetching default
#  forcefield into ff module, but without circular imports?
from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
with open('ff/params/smirnoff_1_1_0_ccc.py') as f:
    ff_handlers = deserialize_handlers(f.read())
default_forcefield = Forcefield(ff_handlers)

ordered_handles = default_forcefield.get_ordered_handles()
ordered_params = default_forcefield.get_ordered_params()
ordered_learning_rates = learning_rates_like_params(ordered_handles, ordered_params)

flatten, unfllatten = flatten_and_unflatten(ordered_params)
learning_rates = flatten(ordered_learning_rates)

mols = get_ligands()
mol, mol_ref = mols[:2]


def assert_conversion_trainable(conversion, n_epochs=10):
    """test that the loss goes down

    note: want to pull the training loop out into optimize probably...
    """

    initial_flat_params = flatten(ordered_params)

    def predict(params):
        return conversion.predict(unfllatten(params))

    initial_prediction = predict(initial_flat_params)

    label = initial_prediction - 100

    def loss(params):
        residual = predict(params) - label
        return l1_loss(residual)

    def update(x, v, g):
        raw_search_direction = - g
        search_direction = raw_search_direction * learning_rates

        x_increment = truncated_step(x, v, g, search_direction=search_direction)
        x_next = x + x_increment

        return x_next

    flat_param_traj = [initial_flat_params]
    loss_traj = [l1_loss(initial_prediction - label)]
    print(f'initial loss: {loss_traj[-1]:.3f}')

    for t in range(n_epochs):
        x = flat_param_traj[-1]
        v, g = value_and_grad(loss)(x)
        x_next = update(x, v, g)

        print(x_next - x)

        print(f'epoch {t}: loss = {v:.3f}, gradient norm = {np.linalg.norm(g):.3f}')

        flat_param_traj.append(x_next)
        loss_traj.append(v)

    window_size = min(5, n_epochs // 2)
    before = loss_traj[0]
    after = np.median(loss_traj[-window_size:])

    assert after < before, f"before: {before:.3f}, after: {after:.3f}"


def test_rabfe_solvent_conversion_trainable():
    """test that the loss goes down"""
    solvent_conversion = SolventConversion(mol, mol_ref, default_forcefield)
    assert_conversion_trainable(solvent_conversion, 10)


def test_rabfe_complex_conversion_trainable():
    """test that the loss goes down"""
    complex_conversion = ComplexConversion(mol, mol_ref, protein_pdb, default_forcefield)
    assert_conversion_trainable(complex_conversion, 10)
