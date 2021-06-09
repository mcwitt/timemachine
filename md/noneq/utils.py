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
        print(RuntimeWarning('some particles already alchemical!'))

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


def construct_water_deletion_lambda_schedule(num_windows=60):
    """Interpolate the results from a previous schedule optimization

    Notes
    -----
    * Adapted for deleting / inserting a single water in about ~10,000 MD steps
    * Reversing this gives an approximately good lambda schedule for water insertion

    TODO: rather than linearly interpolating raw data points this way, just precompute a few chebyshev coefficients
    TODO: optimize with more controllable dials

    See Also
    --------
    * fe.free_energy.construct_lambda_schedule -- manually optimized for relative calculations
    """

    # from applying https://github.com/proteneer/timemachine/blob/04d20878d1552efacac393171d39bb95f97430bf/scripts/noneq_adaptation/adapt_noneq.py#L98-L144
    yp = np.array([0., 0.01533059, 0.02229805, 0.02690808, 0.03101898,
                   0.03441388, 0.03791426, 0.04162575, 0.04488577, 0.04820164,
                   0.05065692, 0.05316312, 0.05590272, 0.05873761, 0.06141923,
                   0.06400197, 0.06637154, 0.06849308, 0.07079436, 0.07307307,
                   0.07536735, 0.07751866, 0.08011125, 0.08231346, 0.08437469,
                   0.08669951, 0.0889812, 0.09112025, 0.09335162, 0.09541216,
                   0.09742787, 0.09929203, 0.10117189, 0.10370826, 0.10606132,
                   0.10858983, 0.11071546, 0.11285919, 0.11467657, 0.11668331,
                   0.11894838, 0.12109086, 0.12322202, 0.12590958, 0.12799611,
                   0.12998178, 0.13209227, 0.13432605, 0.13611111, 0.13810413,
                   0.14046818, 0.14231534, 0.14440701, 0.14660081, 0.14867329,
                   0.15060446, 0.15302838, 0.15541909, 0.15768687, 0.16005014,
                   0.16192792, 0.16405529, 0.16653765, 0.16813328, 0.17017239,
                   0.17239472, 0.17520668, 0.17756246, 0.17940363, 0.1811484,
                   0.18326328, 0.18558914, 0.18782285, 0.18990331, 0.19238311,
                   0.19492217, 0.19782922, 0.20077991, 0.20373518, 0.20743844,
                   0.21053796, 0.21356965, 0.21657056, 0.22061602, 0.22332947,
                   0.22765496, 0.23361871, 0.23831932, 0.24281474, 0.24818963,
                   0.25544663, 0.26228598, 0.27096489, 0.27995752, 0.29032133,
                   0.3057806, 0.32154514, 0.33801365, 0.35913265, 0.38613041,
                   0.40806747, 0.44087817, 0.47039889, 0.49554231, 0.52291761,
                   1.])
    xp = np.linspace(0, 1, len(yp))

    x_interp = np.linspace(0, 1, num_windows)
    return np.interp(x_interp, xp, yp)
