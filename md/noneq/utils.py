from timemachine.lib.potentials import Nonbonded
from copy import deepcopy
from md.ensembles import PotentialEnergyModel
import numpy as np


def make_particles_alchemical_in_place(nb: Nonbonded, particle_indices: np.array) -> None:
    """Set lambda offset idxs of these particles to 1, and assert that no other particles are alchemical"""

    offset_idxs = nb.get_lambda_offset_idxs()
    plane_idxs = nb.get_lambda_plane_idxs()

    all_particles = range(len(offset_idxs))
    _normal_particles = set(all_particles).difference(set(particle_indices))
    normal_particles = np.array(sorted(_normal_particles))

    # validate
    offsets_are_0 = set(offset_idxs[normal_particles]) == {0}
    planes_are_0 = set(plane_idxs[normal_particles]) == {0}
    if not (offsets_are_0 and planes_are_0):
        raise (RuntimeError('some particles already alchemical!'))

    modified_offset_indices = np.array(offset_idxs)
    modified_offset_indices[particle_indices] = 1

    nb.set_lambda_offset_idxs(modified_offset_indices)


def alchemify(potential_energy: PotentialEnergyModel, particle_indices: np.array) -> PotentialEnergyModel:
    """Construct a new potential energy model where the Nonbonded term has been replaced"""
    ubps = potential_energy.unbound_potentials
    nb = ubps[-1]
    assert type(nb) == Nonbonded
    alchemical_nb = deepcopy(nb)
    make_particles_alchemical_in_place(alchemical_nb, particle_indices)
    alchemical_unbound_potentials = ubps[:-1] + [alchemical_nb]

    # TODO: better way to copy all but one parameter?
    #   (if class attributes of PotentialEnergyModel change, this code will break)
    original_potential_energy_params = dict(
        sys_params=potential_energy.sys_params,
        precision=potential_energy.precision,
        guard_threshold=potential_energy.guard_threshold,
    )
    alchemical_model = PotentialEnergyModel(
        unbound_potentials=alchemical_unbound_potentials,
        **original_potential_energy_params
    )
    return alchemical_model
