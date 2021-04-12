# classes that:
# * represent rigid-body orientations
# * project ligand coordinates to rigid-body orientations
# * compare rigid-body orientations

from jax import numpy as np
from typing import Optional

Coords = np.ndarray


# orientations are defined by a translation and optionally a rotation
class Translation:
    def __init__(self, translation_vector=np.zeros(3)):
        self.translation_vector = translation_vector


class Rotation:
    def __init__(self, rotation_matrix=np.eye(3)):
        """TODO: quaternions probably more suitable?"""
        self.rotation_matrix = rotation_matrix


class Orientation:
    def __init__(self, translation: Translation, rotation: Optional[Rotation] = None):
        self.translation = translation
        self.rotation = rotation


#
class Projection:
    """Projects a set of coordinates to an orientation"""

    def project(self, x: Coords) -> Orientation:
        raise (NotImplementedError)

    def __call__(self, x: Coords) -> Orientation:
        return self.project(x)


class CentroidProjection(Projection):
    def project(self, x: Coords):
        return Orientation(Translation(np.mean(x, axis=0)), rotation=None)


class RelativeOrientationProjection(Projection):
    def __init__(self, reference: Coords):
        self.reference = reference

    def project(self, x: Coords):
        displacement = Translation(np.mean(x, axis=0) - np.mean(self.reference, axis=0))

        # TODO: replace this line with a function that gets an orientation relative to reference
        rotation = Rotation(np.eye(3))

        return Orientation(displacement, rotation)


# functions that take orientations and return numbers
class OrientationBasedPotential:
    def compute_potential(self, a: Orientation, b: Orientation) -> float:
        raise (NotImplementedError)

    def __call__(self, a: Orientation, b: Orientation) -> float:
        return self.compute_potential(a, b)


class HarmonicDistancePotential(OrientationBasedPotential):
    def __init__(self, k=10.0):
        self.k = k

    def compute_potential(self, a: Orientation, b: Orientation) -> float:
        # TODO: oof: make this more direct
        distance = np.linalg.norm(a.translation.translation_vector - b.translation.translation_vector)

        return 0.5 * self.k * distance ** 2


class RotationTracePotential(OrientationBasedPotential):
    def __init__(self, k=10.0):
        self.k = k

    def compute_potential(self, a: Orientation, b: Orientation) -> float:
        r_a, r_b = a.rotation.rotation_matrix, b.rotation.rotation_matrix

        # http://www.boris-belousov.net/2016/12/01/quat-dist/
        difference_rotation = np.dot(r_a, r_b.T)

        return 0.5 * self.k * ((np.trace(difference_rotation) - 1) / 2 - 1) ** 2


class QuaternionDistancePotential(OrientationBasedPotential):
    def __init__(self, k=10.0):
        self.k = k

    def compute_potential(self, a: Orientation, b: Orientation) -> float:
        # I want to do something like
        # distance = quaternion_distance(a.rotation.as_quaternion() - b.rotation.as_quaternion())
        # return 0.5 * self.k * distance**2

        raise NotImplementedError


class OrientationRestraint:
    def __init__(self, projection: Projection, potential: OrientationBasedPotential):
        self.projection = projection
        self.potential = potential

    def compute_potential(self, x_a: Coords, x_b: Coords) -> float:
        return self.potential(self.projection(x_a), self.projection(x_b))
