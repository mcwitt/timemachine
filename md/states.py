# Sampler states

import numpy as np
from typing import NamedTuple

Coordinates = Velocities = Box = Array = np.array

class CoordsVelBox(NamedTuple):
    coords: Coordinates
    velocities: Velocities
    box: Box
