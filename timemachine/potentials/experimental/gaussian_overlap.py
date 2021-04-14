from jax import config, grad, jit, numpy as np
from jax.numpy.linalg import inv, slogdet

from scipy.optimize import minimize

from collections import namedtuple

config.update("jax_enable_x64", True)

# types
Coords = np.ndarray
Gaussian = namedtuple('Gaussian', ['mean', 'cov'])


def best_fit_gaussian(x: Coords, eps: float = 1e-2) -> Gaussian:
    """Sample covariance of x, plus eps diagonal padding"""

    # sample mean
    mu = np.mean(x, 0)
    x_ = x - mu

    # sample covariance + some diagonal padding
    sample_cov = np.dot(x_.T, x_) / (len(x_) - 1)
    padding = eps * np.eye(x.shape[1])
    # if x is 'flat' in some direction, still have a little width in that direction
    cov = sample_cov + padding

    return Gaussian(mu, cov)


def log_kl_div(a: Gaussian, b: Gaussian) -> float:
    """ Expression for KL div between two Gaussians from eq 2 of
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.148.2502&rep=rep1&type=pdf

    Notes
    -----
    * This is log (KL) , rather than just KL, since we don't want the gradient of resulting restraining potentials to
        vanish when a and b are far away from each other.
    """

    dim = len(a.mean)

    # compute log determinant ratio
    sign_a, log_det_a = slogdet(a.cov)
    sign_b, log_det_b = slogdet(b.cov)

    assert (sign_a > 0)
    assert (sign_b > 0)

    log_det_ratio = log_det_b - log_det_a

    # compute trace (inv(b.cov) @ a.cov)
    inv_cov_b = inv(b.cov)
    trace_prod = np.trace(np.dot(inv_cov_b, a.cov))

    # compute quadratic form
    displacement = a.mean - b.mean
    quad_form = np.dot(np.dot(displacement.T, inv_cov_b), displacement)

    return 0.5 * (log_det_ratio + trace_prod - dim + quad_form)


def restraint_potential(x: Coords, y: Coords) -> float:
    """log ( kl(x, y) ) + log ( kl(y, x) )

    Notes
    -----
    * While intuitive, this is unsuitable for use as a rigid-body restraint, since it will be sensitive
        to global scaling of x and y.
        In other words, grad(restraint_potential)(x, y) will distort x by scaling x, not just
        rigidly translating / rotating it.
    """
    a, b = best_fit_gaussian(x), best_fit_gaussian(y)
    return log_kl_div(a, b) + log_kl_div(b, a)  # symmetrize


def volume(square_matrix: np.ndarray) -> float:
    return np.linalg.det(square_matrix)


def normalize_volume(g: Gaussian, target_vol: float) -> Gaussian:
    """ isotropically rescale g.cov so that det(g_prime.cov) = target_vol """

    dim = len(g.mean)

    current_vol = volume(g.cov)
    scaling_factor = np.power(target_vol / current_vol, 1 / dim)
    return Gaussian(g.mean, scaling_factor * g.cov)


def restraint_potential_with_volume(x: Coords, y: Coords, ref_vol_x: float, ref_vol_y: float) -> float:
    """ log ( kl(x_norm, y_norm) ) +  log (kl(y_norm, x_norm) )

    where x_norm is best fit Gaussian to x, but scaled to have ref_vol_x

    Notes
    -----
    * While this addresses the issue where grad(restraint_potential)(x, y) will distort x even by isotropic
        scaling, it is still not suitable for use as an effectively rigid-body restraint potential, since
        grad(restraint_potential_with_volume)(x, y) will still distort x.

        In restraint_potential_with_volume, the distortions can now be anisotropic (scaling up along one axis while
        compensating by scaling down along another axis).

        To address this, we need to further constrain the gaussian summaries
    """
    a, b = best_fit_gaussian(x), best_fit_gaussian(y)

    a_, b_ = normalize_volume(a, ref_vol_x), normalize_volume(b, ref_vol_y)

    return log_kl_div(a_, b_) + log_kl_div(b_, a_)  # symmetrize


def get_key_points(z: Coords) -> np.ndarray:
    """ Draw arrows from mean(z) to +/- eigval_i * eigvec_i

     for eigvals, eigvecs derived from best_fit_gaussian(z).cov
     """
    assert (z.shape[1] == 2)  # currently hardcoded for dim=2

    gaussian = best_fit_gaussian(z)
    vals, vecs = np.linalg.eigh(gaussian.cov)
    key_points = gaussian.mean + np.array([
        + vals[0] * vecs[0],
        - vals[0] * vecs[0],
        + vals[1] * vecs[1],
        - vals[1] * vecs[1],
    ])
    return key_points


def express_a_point_as_a_weighted_sum(points: Coords, target: np.ndarray, penalty=1.0) -> np.ndarray:
    """ find weights such that \sum_i w_i points_i = target, by optimizing
    loss(weights) = || (\sum_i w_i points_i) - target || + penalty ||weights||

    Notes
    -----
    TODO: jit overhead should be moved out of this function
    TODO: expose convergence criteria (tol, maxiter) to caller
    """
    weights_0 = np.ones(len(points)) / len(points)

    @jit
    def loss(weights):
        pred = np.dot(weights, points)

        return np.linalg.norm(pred - target) + penalty * np.linalg.norm(weights, ord=2)  # do I want ord = 1 or 2?

    def loss_and_grad(weights):
        return loss(weights), grad(loss)(weights)

    result = minimize(loss_and_grad, weights_0, jac=True, options=dict(maxiter=100))

    optimized_weights = result.x

    return optimized_weights


def get_key_point_weights(z: Coords) -> np.ndarray:
    """generate key points from z, and then express each key point as a weighted sum of z_i"""
    key_points = get_key_points(z)
    weights = []
    for key_point in key_points:
        weights.append(express_a_point_as_a_weighted_sum(z, key_point))
    weights = np.array(weights)

    return weights


def gaussian_from_key_points(z: np.ndarray, weights: np.ndarray) -> Gaussian:
    mean = np.mean(z, axis=0)
    reconstructed = np.array([np.dot(w, z) for w in weights])
    basis = 0.5 * (reconstructed[1::2] - reconstructed[::2])

    cov = np.zeros((2, 2))
    for vec in basis:
        cov += np.outer(vec, vec) / np.linalg.norm(vec)

    return Gaussian(mean, cov)


def restraint_using_key_points(x: Coords, y: Coords, x_weights: np.ndarray, y_weights: np.ndarray) -> float:
    """ Again, this has the x,y distortion problems from restraint_potential, restraint_potential_with volume"""

    a = gaussian_from_key_points(x, x_weights)
    b = gaussian_from_key_points(y, y_weights)

    return log_kl_div(a, b) + log_kl_div(b, a)


def gaussian_from_rigidified_key_points(z: Coords, weights: np.ndarray, eigvals: np.ndarray,
                                        eps: float = 1e-2) -> Gaussian:
    """construct a Gaussian from z by:
    * getting approximate PMI vectors from weighted combinations of z (using constant, precomputed weights)
    * normalizing each PMI vector, and then scaling its contribution (using constant, precomputed eigvals)

    Notes:
    ------
    * TODO: The vectors used in the reconstruction (cov = \sum_i val_i outer(vec_i, vec_i) won't be exactly orthogonal.
        This is okay from a stability perspective.
        (In fact, they could even be linearly dependent, as long as we keep eps > 0 !)
        However, I think deviations from orthogonality here might mean the reconstruction is still imperfectly rigid?
        Should probably orthogonalize these to be further on the safe side...
    """

    mean = np.mean(z, axis=0)
    dim = z.shape[1]

    reconstructed = np.array([np.dot(w, z) for w in weights])
    basis = 0.5 * (reconstructed[1::2] - reconstructed[::2])

    cov = eps * np.eye(dim)
    for val, vec in zip(eigvals, basis):
        vec_norm = np.linalg.norm(vec)
        normalized_vec = vec / vec_norm  # this should "rigidify"

        cov += val * np.outer(normalized_vec, normalized_vec)

    return Gaussian(mean, cov)


def rigid_restraint_with_rigid_key_points(x, y, x_weights, y_weights, x_vals, y_vals) -> float:
    """ log( kl(x_rigid, y_rigid) ) + log ( kl(y_rigid, x_rigid ) )

    where x_rigid is a Gaussian summary of x computed in such a way that it is mostly "rigid"

    Notes
    -----
    * Mostly addresses the x, y distortion issues with previous attempts
        (restraint_potential, restraint_potential_with_volume, and restraint_using_key_points)
    """

    a = gaussian_from_rigidified_key_points(x, x_weights, x_vals)
    b = gaussian_from_rigidified_key_points(y, y_weights, y_vals)

    return log_kl_div(a, b) + log_kl_div(b, a)


def gaussian_with_scale_hints(z: Coords, eigval_hints: np.ndarray, eps: float = 1e-2) -> Gaussian:
    """construct a Gaussian from z by:
    * computing g = best_fit_gaussian(z)
    * computing evals, evecs = np.linalg.eigh(g.cov)
    * reconstructing cov using eigval_hints rather than evals

    Notes
    -----
    * This is intended to give the reconstructed Gaussian a prescribed shape
    """

    mean = np.mean(z, axis=0)
    g = best_fit_gaussian(z, eps)
    evals, evecs = np.linalg.eigh(g.cov)
    assert min(evals) >= eps
    assert (eigval_hints[1:] >= eigval_hints[:-1]).all()  # assume sorted in same order as np.linalg.eigh

    rescaled_cov = np.sum(np.array([val * np.outer(vec, vec) for (val, vec) in zip(eigval_hints, evecs)]), axis=0)

    return Gaussian(mean, rescaled_cov)


def rigid_restraint_with_scale_hints(x: Coords, y: Coords, x_scales: np.ndarray, y_scales: np.ndarray) -> float:
    """ log( kl(x_rigid, y_rigid) ) + log ( kl(y_rigid, x_rigid ) )

    where x_rigid is a Gaussian summary of x computed in such a way that it is mostly "rigid"

    Notes
    -----
    * Similar goal as in rigid_restraint_with_rigid_key_points, but achieved by rescaling best-fit-Gaussian covariance matrix
        rather than by reconstructing a Gaussian using key points...
    """

    a = gaussian_with_scale_hints(x, x_scales)
    b = gaussian_with_scale_hints(y, y_scales)

    return log_kl_div(a, b) + log_kl_div(b, a)
