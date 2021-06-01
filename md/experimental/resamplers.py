import numpy as np
from numpy import arange, cumsum, digitize
from numpy.random import rand
from scipy.special import logsumexp

from typing import Callable, Tuple

LogWeights = IndexArray = np.array
Resampler = Callable[[LogWeights], Tuple[IndexArray, LogWeights]]


def null_resample(log_weights: LogWeights) -> Tuple[IndexArray, LogWeights]:
    """No interaction"""
    indices = arange(len(log_weights))
    return indices, log_weights


def stratified_resample(log_weights: LogWeights) -> Tuple[IndexArray, LogWeights]:
    """Apply stratified resampling to a vector of unnormalized log weights

    Notes
    -----
    * Stratified resampling (unlike residual resampling or multinomial resampling),
        is sensitive to arbitrary particle ordering.
        TODO: before calling this function, should we periodically re-shuffle the particles?

    References
    ----------
    * [Douc, Cappé, Moulines, 2005] Comparison of Resampling Schemes for Particle Filtering
        https://arxiv.org/abs/cs/0507025
    * [Chopin, 2021] Implementation in nchopin/particles
        https://github.com/nchopin/particles/blob/8e5eb4c6886823598ba941d4f4eab551e8779509/particles/resampling.py#L528-L531
    * [Naesseth, 2019] Implementation in blei-lab/variational-smc
        https://github.com/blei-lab/variational-smc/blob/5487bf9666e6ae72ef8d34b71f7341768fec707a/variational_smc.py#L8-L19
    """

    # indices
    n = len(log_weights)
    w = np.exp(log_weights - logsumexp(log_weights))
    indices = digitize((arange(n) + rand(n)) / n, cumsum(w))

    # new weights
    avg_log_weights = logsumexp(log_weights - np.log(n)) * np.ones(n)

    return indices, avg_log_weights
