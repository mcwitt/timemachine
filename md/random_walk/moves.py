from md.moves import MonteCarloMove
from md.states import CoordsVelBox
from md.random_walk.utils import swap_positions
from typing import Tuple, List, Callable
import numpy as np

IndexArray = np.array
ReducedPotentialFxn = Callable[[CoordsVelBox], float]


class RandomWalkMove(MonteCarloMove):
    def __init__(self):
        """Base class for moves with symmetric proposal probabilities,
        p(y|x) = p(x|y) for any x,y."""
        raise (NotImplementedError)


class SwapIndistinguishable(RandomWalkMove):
    def __init__(self, pivot_group: IndexArray, other_groups: List[IndexArray]):
        """Swap the indices of one particle group with a random other particle group.
        For example, we might swap the index of "water 0" with a random other water,
        and then make an alchemical move using water 0, to emulate making an alchemical move
        using a random water.

        Warnings
        --------
        * This assumes (without checking) that the particle groups are in fact indistinguishable!
        """

        # ensure these are original numpy arrays rather than Jax numpy arrays
        self.pivot_group = np.array(pivot_group)
        self.other_groups = list(map(np.array, other_groups))

        self._validate()

    def _validate(self):
        """Assert that all the groups have the same number of particles"""
        n = len(self.pivot_group)
        assert set(map(len, self.other_groups)) == {n}

    def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
        pivot = self.pivot_group
        i = np.random.randint(len(self.other_groups))
        other = self.other_groups[i]

        x_prime = swap_positions(x.coords, pivot, other)
        v_prime = swap_positions(x.velocities, pivot, other)

        return CoordsVelBox(x_prime, v_prime, x.box.copy()), 0.0


class GaussianTranslation(RandomWalkMove):
    def __init__(self, group: IndexArray, proposal_stddev: float, reduced_potential_fxn: ReducedPotentialFxn):
        """Rigidly translate a group of particles by a Gaussian perturbation

        Parameters
        ----------
        group: IndexArray
            particle indices to displace as a group
        proposal_stddev: float, assumed in nanometers
        reduced_potential_fxn: Callable[[CoordsVelBox], float]

        See Also
        --------
        * DecoupledGaussianTranslation, which *does not* check the potential energy
            before and after displacement.
        """
        self.group = np.array(group)
        self.proposal_stddev = proposal_stddev
        self.reduced_potential_fxn = reduced_potential_fxn

    def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
        x_prime = np.array(x.coords)
        translation = np.random.randn(3) * self.proposal_stddev
        x_prime[self.group] = x.coords[self.group] + translation

        # velocities, box unchanged
        v_prime, box_prime = x.velocities.copy(), x.box.copy()
        proposal = CoordsVelBox(x_prime, v_prime, box_prime)

        before = self.reduced_potential_fxn(x)
        after = self.reduced_potential_fxn(proposal)
        reduced_work = after - before

        return proposal, - reduced_work


class DecoupledGaussianTranslation(RandomWalkMove):
    def __init__(self, group: IndexArray, proposal_stddev: float):
        """Rigidly translate a group of particles by a Gaussian perturbation

        group: IndexArray
        proposal_stddev: float, assumed in nanometer

        Warnings
        --------
        * Assumes (without checking) that the potential energy before and after translating
            this group is the same! (I.e. that this group is fully decoupled from the
            rest of the system.)

        See Also
        --------
        * GaussianTranslation , which *does* check the potential energy before and after displacement
        """
        self.group = np.array(group)
        self.proposal_stddev = proposal_stddev

    def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
        x_prime = np.array(x.coords)
        translation = np.random.randn(3) * self.proposal_stddev
        x_prime[self.group] = x.coords[self.group] + translation

        # velocities, box unchanged
        v_prime, box_prime = x.velocities.copy(), x.box.copy()

        return CoordsVelBox(x_prime, v_prime, box_prime), 0.0
