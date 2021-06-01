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
    """
    Notes
    -----
    * Stratified resampling (unlike residual resampling or multinomial resampling),
        is sensitive to arbitrary particle ordering.
        TODO: should we shuffle the weights first?


    References
    ----------
    * [Douc, Cappé, Moulines,
    * Implementation in variational SMC
    """

    # indices
    n = len(log_weights)
    w = np.exp(log_weights - logsumexp(log_weights))
    indices = digitize((arange(n) + rand(n)) / n, cumsum(w))

    # new weights
    avg_log_weights = logsumexp(log_weights - np.log(n)) * np.ones(n)

    return indices, avg_log_weights
