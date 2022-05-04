__all__ = [
    "one_sided_exp",
    "construct_endpoint_reweighting_estimator",
    "construct_mixture_reweighting_estimator",
    "interpret_as_mixture_potential",
]

from typing import Any, Callable, Collection

import numpy as np
from jax import numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp
from pymbar import BAR, MBAR

Samples = Collection
Params = Collection
Array = Any  # see https://github.com/google/jax/issues/943
Energies = Array

BatchedReducedPotentialFxn = Callable[[Samples, Params], Energies]


def log_mean(log_values: Array) -> float:
    """stable log(mean(values))

    log(mean(values))
    = log(sum(values / len(values)))
    = logsumexp(log(values) - log(len(values))
    """
    return logsumexp(log_values - jnp.log(len(log_values)))


def estimate_log_z_ratio(log_importance_weights: Array) -> float:
    """stable log(mean(importance_weights))"""
    return log_mean(log_importance_weights)


def one_sided_exp(delta_us: Array) -> float:
    """exponential averaging

    References
    ----------
    [1] pymbar implementation
        https://github.com/choderalab/pymbar/blob/15f932a271343e611ed4be2d468c77b1d11cf01f/pymbar/exp.py#L54
    """
    # delta_us = -log_importance_weights
    # delta_f  = -log_z_ratio
    return -estimate_log_z_ratio(-delta_us)


def interpret_as_mixture_potential(u_kn: Array, f_k: Array, N_k: Array) -> Array:
    r"""Interpret samples from multiple states k as if they originate from a single state
    defined as a weighted mixture:

    $p_{mix}(x) \propto \sum_k w_k q_k(x) / Z_k$

    where
    * $q_k(x) = exp(-u_k(x))$ is the Boltzmann weight function for state k
    * $f_k = - log Z_k$ is the assumed normalization for state k
    * $w_k \propto N_k$ is the mixture weight of state k

    (up to a single constant)

    Parameters
    ----------
    u_kn : [K, N] float array
        reduced potentials of all N samples evaluated in all K states
        u_kn[k, n] = u_k(x_n)
    f_k : [K,] float array
        reduced free energies of all K states
        (up to an additive constant)
    N_k : [K,] int array
        number of samples from each individual state
        (sum(N_k) must equal N)

    Returns
    -------
    mixture_u_n : [N,] float array
        mixture_u_n[n] = u_mix(x_n), where

        u_mix(x) = -log(q_mix(x))
        q_mix(x) = \sum_k w_k (q_k(x) / Z_k)

    Notes
    -----
    * This is nicely illustrated in ref [1] as a way to interpret MBAR [2]
    * Depends on the accuracy of the input free energy estimates (f_k - f_k[0]) for the source states.
    * This is not the only way to interpret samples from multiple distributions
        as if they come from a single mixture distribution.
        In ref [3], alternatives are systematically enumerated.
        Assuming f_k are exact, the current approach corresponds to N3 in ref [3].
        (N3 the most computationally expensive of the options in ref [3],
        since it requires evaluating all k energy functions on every sample,
        but it is also the lowest variance.)

    References
    ----------
    [1] [Shirts, 2017] Reweighting from the mixture distribution as a better way to describe
        the Multistate Bennett Acceptance Ratio
        https://arxiv.org/abs/1704.00891
    [2] [Shirts, Chodera, 2008] Statistically optimal analysis of samples from multiple equilibrium states.
        J. Chem. Phys. 129:124105, 2008.
        http://dx.doi.org/10.1063/1.2978177
    [3] [Elvira+, 2019] Generalized multiple importance sampling
        https://arxiv.org/abs/1511.03095
    """
    # one-liner: mixture_u_n = -logsumexp(f_k - u_kn.T, b=N_k, axis=1)

    # expanding steps:
    K, N = u_kn.shape
    assert f_k.shape == (K,)
    N_k = np.array(N_k)
    assert np.sum(N_k) == N

    # q_k(x_n) = exp(-u_k(x_n))
    log_q_kn = -u_kn

    # p_k(x_n) = c q_k(x_n) / Z_k
    # (up to a single undetermined constant c, shared across k)
    log_Z_k = -f_k
    normalized_log_q_kn = log_q_kn - jnp.expand_dims(log_Z_k, 1)

    # mixture weights from sampling proportions
    log_w_k = jnp.log(N_k) - jnp.log(jnp.sum(N_k))

    # q_mix(x_n) = \sum_k w_k p_k(x_n)
    weighted_log_q_kn = jnp.expand_dims(log_w_k, 1) + normalized_log_q_kn
    mixture_log_q_n = logsumexp(weighted_log_q_kn, axis=0)

    # q_mix(x_n) = exp(-u_mix(x_n))
    mixture_u_n = -mixture_log_q_n

    assert mixture_u_n.shape == (N,)

    return mixture_u_n


def construct_u_mix(u_k_fxn: Callable[[Any], Array], f_k: Array, N_k: Array) -> Callable[[Any], float]:
    """variant of interpret_as_mixture_potential where u_k_fxn: X -> R^K is not already precomputed on N samples"""
    # w_k \propto N_k, \sum_k w_k = 1
    log_w_k = jnp.log(N_k) - jnp.log(jnp.sum(N_k))

    # Z_k assumed normalization constants
    log_Z_k = -f_k

    def u_mix_fxn(x: Any) -> float:
        r"""
        p_k(x)   = exp(-u_k(x)) / Z_k
        p_mix(x) = \sum_k w_k p_k(x)
        u_mix(x) = -log(p_mix(x))
        """
        log_p_k = -u_k_fxn(x) - log_Z_k
        return -logsumexp(log_w_k + log_p_k)

    return u_mix_fxn


def construct_endpoint_reweighting_estimator(
    samples_0: Samples,
    samples_1: Samples,
    batched_u_0_fxn: BatchedReducedPotentialFxn,
    batched_u_1_fxn: BatchedReducedPotentialFxn,
    ref_params: Params,
    ref_delta_f: float,
) -> Callable[[Params], float]:
    """assuming
    * endpoint samples (samples_0, samples_1)
    * precise estimate of free energy difference at initial params
        ref_delta_f ~= f(ref_params, 1) - f(ref_params, 0)

    construct an estimator of f(params, 1) - f(params, 0)

    Parameters
    ----------
    samples_0: [N_0,] collection
        samples from endstate 0
    samples_1: [N_1,] collection
        samples from endstate 1
    batched_u_0_fxn
        function that computes batch of endstate 0 energies at specified params
        [u_0(x, params) for x in samples_0]
    batched_u_1_fxn
        function that computes batch of endstate 1 energies at specified params
        [u_1(x, params) for x in samples_1]
    ref_params
        assume samples_0 ~ exp(-u_0(., ref_params)) and
               samples_1 ~ exp(-u_1(., ref_params))
    ref_delta_f
        free energy difference between endstates 0, 1 at ref_params
        ref_delta_f ~= f(ref_params, 1) - f(ref_params, 0)

    Returns
    -------
    estimate_delta_f
        computes an estimate f(params, 1) - f(params, 0) for arbitrary params

        notes:
        * estimate_delta_f(ref_params) == ref_delta_f
        * estimate_delta_f(params) can become unreliable when
          params is very different from ref_params
    """
    ref_u_0 = batched_u_0_fxn(samples_0, ref_params)
    ref_u_1 = batched_u_1_fxn(samples_1, ref_params)

    def endpoint_correction_0(params) -> float:
        """estimate f(ref, 0) -> f(params, 0) by reweighting"""
        delta_us = batched_u_0_fxn(samples_0, params) - ref_u_0
        return one_sided_exp(delta_us)

    def endpoint_correction_1(params) -> float:
        """estimate f(ref, 1) -> f(params, 1) by reweighting"""
        delta_us = batched_u_1_fxn(samples_1, params) - ref_u_1
        return one_sided_exp(delta_us)

    def estimate_delta_f(params: Params) -> float:
        """estimate f(params, 1) - f(params, 0)

        using this thermodynamic cycle:

        f(params, 0)  --->  f(params, 1)

             ^                   ^
             |                   |
             |                   |
             |                   |

        f(ref, 0)     --->  f(ref, 1)


        where
        * "f(ref, 0) -> f(ref, 1)" is assumed precomputed (using any precise free energy method)
        * "f(ref, 0) -> f(params, 0)" is estimated by reweighting
        * "f(ref, 1) -> f(params, 1)" is estimated by reweighting
        """
        return ref_delta_f - endpoint_correction_0(params) + endpoint_correction_1(params)

    return estimate_delta_f


def construct_mixture_reweighting_estimator(
    samples_n: Samples,
    u_ref_n: Array,
    batched_u_0_fxn: BatchedReducedPotentialFxn,
    batched_u_1_fxn: BatchedReducedPotentialFxn,
) -> Callable[[Params], float]:
    r"""assuming
    * samples x_n from a distribution p_ref(x) \propto(exp(-u_ref(x))
      that has good overlap with BOTH p_0(params)(x) and p_1(params)(x),
    * evaluation (or estimates) of u_ref_n[n] = u_ref(x_n)

    construct an estimator for the free energy difference
    f_1(params) - f_0(params)

    Parameters
    ----------
    samples_n: [N,] collection
        samples[n] ~ p_ref
        p_ref(x) \propto exp(-u_ref(x))
    u_ref_n: [N,] array
        u_ref_n[n] = u_ref(samples[n])
    batched_u_0_fxn
        computes batch of endstate 0 energies at specified params
        [u_0(x, params) for x in samples_n]
    batched_u_1_fxn
        computes batch of endstate 1 energies at specified params
        [u_1(x, params) for x in samples_n]

    Returns
    -------
    estimate_delta_f
        computes an estimate f(params, 1) - f(params, 0) for arbitrary params

        notes:
        * estimate_delta_f(ref_params) == ref_delta_f
        * estimate_delta_f(params) can become unreliable when
          params is very different from ref_params

    Notes
    -----
    * This is essentially `computePerturbedFreeEnergies` in pymbar [1], but written in a slightly more generic way.
        (Allows the samples to come from sources other than an MBAR mixture, produces a function that can be
        differentiated w.r.t. params if batched_u_0_fxn, batched_u_0_fxn are differentiable w.r.t. params.)
    * Reweighting from a single reference state is used in works like ref [2] in the context of force field fitting
    * Forming a single reference state as a mixture of several states (i.e. a constant denominator "u_ref_n")
        and differentiating the numerator ("-u(samples_n, params)") w.r.t. params
        is used in works like ref [3] to differentiate free energy estimates w.r.t. params.
    * Non-requirement: u_ref does not have to be of the same functional form as u_0, u_1

    References
    ----------
    [1] pymbar implementation of computePerturbedFreeEnergies
        https://github.com/choderalab/pymbar/blob/3c4262c490261110a7595eec37df3e2b8caeab37/pymbar/mbar.py#L1163-L1237
    [2] Messerly RA, Razavi SM, and Shirts MR. Configuration-Sampling-Based Surrogate Models for Rapid
        parameterization of Non-Bonded Interactions.
        J. Chem. Theory Comput. 2018, 14, 6, 3144–3162 https://doi.org/10.1021/acs.jctc.8b00223
    [3] Wieder et al. PyTorch implementation of differentiable reweighting in neutromeratio
        https://github.com/choderalab/neutromeratio/blob/2abf29f03e5175a988503b5d6ceeee8ce5bfd4ad/neutromeratio/parameter_gradients.py#L246-L267
    """
    assert len(samples_n) == len(u_ref_n)

    def f_0(params):
        """estimate f(params, 0) - f(ref) by reweighting"""
        u_0_n = batched_u_0_fxn(samples_n, params)
        return one_sided_exp(u_0_n - u_ref_n)

    def f_1(params) -> float:
        """estimate f(params, 1) - f(ref) by reweighting"""
        u_1_n = batched_u_1_fxn(samples_n, params)
        return one_sided_exp(u_1_n - u_ref_n)

    def estimate_delta_f(params) -> float:
        r"""estimate f(params, 1) - f(params, 0)

        using this thermodynamic cycle:

        f(params, 0)  --->  f(params, 1)

                /\         /\
                 \         /
                  \       /
                   \     /

                   f(ref)
        where
        * "f(params, 0) - f(ref)" is estimated by reweighting
        * "f(params, 1) - f(ref)" is estimated by reweighting"""

        return f_1(params) - f_0(params)

    return estimate_delta_f


class SampledState:
    def __init__(self, samples, batched_u_fxn):
        self.samples = samples
        self.batched_u_fxn = batched_u_fxn

    def compute_u_n(self):
        u_n = self.batched_u_fxn(self.samples)
        assert u_n.shape == (len(self.samples),)
        return u_n


class Mixture:
    def __init__(self, component_samples, u_k_fxn):
        self.component_samples = component_samples
        self.u_k_fxn = u_k_fxn

    @property
    def N_k(self):
        return np.array([len(s) for s in self.component_samples])

    @property
    def samples(self):
        samples = np.concatenate(self.component_samples)
        assert len(samples) == sum(self.N_k)
        return samples

    def compute_u_kn(self):
        """TODO: incremental updates, rather than from scratch"""
        N, K = sum(self.N_k), len(self.N_k)

        u_kn = (vmap(self.u_k_fxn)(self.samples)).T
        assert u_kn.shape == (K, N)

        return u_kn

    def compute_f_k(self):
        u_kn = self.compute_u_kn()
        N_k = self.N_k
        mbar = MBAR(u_kn, N_k)

        # TODO: assert small estimation error
        return mbar.f_k

    def get_mixture_potential(self) -> Callable:
        f_k = np.array(self.compute_f_k())
        N_k = np.array(self.N_k)

        u_mix = construct_u_mix(self.u_k_fxn, f_k, N_k)
        return u_mix

    def collapse_to_single_state(self) -> SampledState:
        samples = np.array(self.samples)
        u_mix = self.get_mixture_potential()
        batched_u_fxn = vmap(u_mix)  # TODO: jit(vmap(u_mix)) ?

        return SampledState(samples, batched_u_fxn)


def estimate_delta_f_both_sampled(state_A: SampledState, state_B: SampledState):
    """f(B) - f(A) using BAR"""

    u_A, u_B = state_A.batched_u_fxn, state_B.batched_u_fxn
    xs_A, xs_B = state_A.samples, state_B.samples

    # forward: A -> B, reverse: B -> A
    w_F = u_B(xs_A) - u_A(xs_A)
    w_R = u_A(xs_B) - u_B(xs_B)

    delta_f, uncertainty = BAR(w_F, w_R)
    # TODO: assert small BAR uncertainty?

    return delta_f


def construct_endpoint_reweighting_estimator_with_proposals(
    proposal_0: SampledState,
    proposal_1: SampledState,
    ref_0: SampledState,
    ref_1: SampledState,
    ref_delta_f: float,
    batched_u_0_fxn: BatchedReducedPotentialFxn,
    batched_u_1_fxn: BatchedReducedPotentialFxn,
) -> Callable[[Params], float]:
    """Variant of construct_endpoint_reweighting_estimator that allows different proposal distributions to be used for
    reweighting than for reference calculation"""
    ref_to_proposal_0 = estimate_delta_f_both_sampled(ref_0, proposal_0)
    ref_to_proposal_1 = estimate_delta_f_both_sampled(ref_1, proposal_1)

    ref_0_to_1 = ref_delta_f

    u_0 = proposal_0.compute_u_n()
    u_1 = proposal_1.compute_u_n()

    def estimate_delta_f(params: Params) -> float:
        """
        (params, 0)  --->  (params, 1)

            ^                   ^
            |                   |
            |                   |
            |                   |

         proposal_0         proposal_1

            ^                   ^
            |                   |
            |                   |
            |                   |

        (ref, 0)     --->   (ref, 1)
        """
        proposal_to_query_0 = one_sided_exp(batched_u_0_fxn(proposal_0.samples, params) - u_0)
        proposal_to_query_1 = one_sided_exp(batched_u_1_fxn(proposal_1.samples, params) - u_1)

        return (-proposal_to_query_0) + (-ref_to_proposal_0) + ref_0_to_1 + ref_to_proposal_1 + proposal_to_query_1

    return estimate_delta_f
