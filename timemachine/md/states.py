# Sampler states

from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class CoordsVelBox:
    coords: NDArray
    velocities: NDArray
    box: NDArray
