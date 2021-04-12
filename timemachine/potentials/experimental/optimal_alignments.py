from jax import numpy as np


def svd_based_alignment(x_a, x_b):
    x_a = x_a - np.mean(x_a, axis=0)
    x_b = x_b - np.mean(x_b, axis=0)

    correlation_matrix = np.dot(x_b.T, x_a)
    U, S, V_tr = np.linalg.svd(correlation_matrix, full_matrices=False)

    ## may want to check for reflections
    # det_U, det_V = np.linalg.det(U), np.linalg.det(V_tr)
    # is_reflection = (det_U * det_V) < 0.0

    rotation = np.dot(U, V_tr)
    return rotation


# TODO: quaternion-based alignment?
#   see also https://github.com/charnley/rmsd
