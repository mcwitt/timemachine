import numpy as np

Array = np.array


def swap_positions(x: Array, indices_a: Array, indices_b: Array) -> Array:
    """x_new[indices_a] = x[indices_b], x_new[indices_b] = x[indices_a]"""

    # copy, and ensure original numpy rather than jax numpy
    x_new = np.array(x)

    x_new[indices_a] = x[indices_b]
    x_new[indices_b] = x[indices_a]
    return x_new
