# Test that we can adjust parameters to make the conversion dG correspond to
# a desired value


import numpy as np

from fe.free_energy_rabfe import  construct_conversion_lambda_schedule
from fe import model_rabfe

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
from ff.handlers.serialize import serialize_handlers
from parallel.client import CUDAPoolClient

from rdkit import Chem

from md import builders, minimizer
from fe.utils import get_romol_conf
from fe.loss import l1_loss
from optimize.step import truncated_step
from optimize.precondition import learning_rates_like_params
from optimize.utils import flatten_and_unflatten
from jax import value_and_grad

with open('ff/params/smirnoff_1_1_0_ccc.py') as f:
    ff_handlers = deserialize_handlers(f.read())

default_forcefield = Forcefield(ff_handlers)
import timemachine
from pathlib import Path

root = Path(timemachine.__file__).parent.parent
path_to_hif2a = root.joinpath('datasets/fep-benchmark/hif2a')

def get_ligands(ligand_sdf):
    print(f'loading ligands from {ligand_sdf}')
    suppl = Chem.SDMolSupplier(ligand_sdf, removeHs=False)
    mols = [x for x in suppl]
    return mols


class SolventConversion():
    def __init__(self, mol, mol_ref,
                 temperature=300, pressure=1.0, dt=2.5*1e-3,
                 num_equil_steps=100, num_prod_steps=10001,
                 num_windows=10, client=CUDAPoolClient(2),
                 initial_forcefield=default_forcefield):

        self.mol = mol
        self.mol_ref = mol_ref
        self.temperature = temperature
        self.pressure = pressure
        self.dt = dt
        self.num_equil_steps = num_equil_steps
        self.num_prod_steps = num_prod_steps
        self.schedule = construct_conversion_lambda_schedule(num_windows)
        self.client = client
        self.initial_forcefield = initial_forcefield
        self.solvent_system, self.solvent_coords, self.solvent_box, self.solvent_topology = builders.build_water_system(
            4.0)

        self.conversion_model = model_rabfe.AbsoluteConversionModel(
            self.client,
            self.initial_forcefield,
            self.solvent_system,
            self.schedule,
            self.solvent_topology,
            self.temperature,
            self.pressure,
            self.dt,
            self.num_equil_steps,
            self.num_prod_steps
        )

        self.flatten, self.unflatten = flatten_and_unflatten(self.initial_forcefield.get_ordered_params())

    def predict(self, flat_params):
        ordered_params = self.unflatten(flat_params)
        mol_coords = get_romol_conf(self.mol)

        # TODO: double-check if this should use initial forcefield, or if I need
        #  to reconstruct a params-dependent forcefield and pass it here...
        forcefield_for_minimization = self.initial_forcefield
        min_solvent_coords = minimizer.minimize_host_4d([self.mol], self.solvent_system, self.solvent_coords,
                                                        forcefield_for_minimization, self.solvent_box)
        solvent_x0 = np.concatenate([min_solvent_coords, mol_coords])
        solvent_box0 = self.solvent_box
        dG_solvent_conversion, dG_solvent_conversion_error = self.conversion_model.predict(
            ordered_params,
            self.mol,
            solvent_x0,
            solvent_box0,
            prefix='solvent_conversion_test'
        )

        return dG_solvent_conversion

def test_rabfe_conversion_trainable(n_steps=10):
    """test that the loss goes down"""

    ligand_sdf = str(path_to_hif2a.joinpath('ligands.sdf').resolve())
    mols = get_ligands(ligand_sdf)
    mol, mol_ref = mols[:2]

    solvent_conversion = SolventConversion(mol, mol_ref)

    ordered_handles = default_forcefield.get_ordered_handles()
    ordered_params = default_forcefield.get_ordered_params()
    ordered_learning_rates = learning_rates_like_params(ordered_handles, ordered_params)

    initial_flat_params = solvent_conversion.flatten(ordered_params)
    learning_rates = solvent_conversion.flatten(ordered_learning_rates)

    initial_prediction = solvent_conversion.predict(initial_flat_params)

    label = initial_prediction - 100

    def loss(params):
        residual = solvent_conversion.predict(params) - label
        return l1_loss(residual)

    def step(x, v, g):
        raw_search_direction = - g
        search_direction = raw_search_direction * learning_rates

        x_increment = truncated_step(x, v, g, search_direction=search_direction)
        x_next = x + x_increment

        return x_next

    flat_param_traj = [initial_flat_params]
    loss_traj = [l1_loss(initial_prediction - label)]
    print(f'initial loss: {loss_traj[-1]:.3f}')

    for t in range(n_steps):
        x = flat_param_traj[-1]
        v, g = value_and_grad(loss)(x)
        x_next = step(x, v, g)

        print(x_next - x)

        print(f'epoch {t}: loss = {v:.3f}, gradient norm = {np.linalg.norm(g):.3f}')

        flat_param_traj.append(x_next)
        loss_traj.append(v)
    
    print(loss_traj)

    window_size = 5
    assert window_size * 2 < n_steps

    assert loss_traj[-window_size:] < loss_traj[:window_size]
