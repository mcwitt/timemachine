from md.moves import MonteCarloMove
from md.states import CoordsVelBox
from md.random_walk.utils import swap_positions
from typing import Tuple, List
import numpy as np

IndexArray = np.array


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

        Warning
        -------
        * This assumes (without checking) that the particle groups are in fact indistinguishable!
        """

        # ensure these are original numpy arrays rather than Jax numpy arrays=
        self.pivot_group = np.array(pivot_group)
        self.other_groups = list(map(np.array, other_groups))

        self._validate()

    def _validate(self):
        """Assert that all the groups have the same number of particles"""
        n = len(self.pivot_group)
        assert set(map(len, self.other_groups)) == {n}

    def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
        pivot = self.pivot_group
        other = np.random.choice(self.other_groups)

        x_prime = swap_positions(x.coords, pivot, other)
        v_prime = swap_positions(x.velocities, pivot, other)

        return CoordsVelBox(x_prime, v_prime, x.box.copy()), 0.0
