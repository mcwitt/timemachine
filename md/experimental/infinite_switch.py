"""Taking the "infinite-switch limit" of replica exchange, simulated tempering, etc.
leads to an averaged effective potential, or, equivalently, a mixture distribution.

Taking this limit is a good idea[1,2].

References
----------
[1] [Plattner et al., 2011] J. Chem. Phys. 135:134111
[2] [Martinsson et al., 2019] 10.1088/1742-5468/aaf323
"""

from jax import vmap, numpy as np
from jax.scipy.special import logsumexp
from typing import Callable

Lambda = Float = float
Conf = np.array

LogProbFxn = Callable[[Conf, Lambda], Float]
ReducedPotentialFxn = Callable[[Conf, Lambda], Float]


class WeightedMixture:
    def __init__(self, log_q: LogProbFxn):
        """q_mix(x) = sum([(q(x, lam) * weight) for (lam, weight) in mixture])

        Computed in log space using:
        log_q_mix(x) = logsumexp([log_q(x, lam) + log_weight for (lam, log_weight) in mixture])
        """
        self.log_q = log_q
        self.log_q_vec = vmap(self.log_q, (None, 0))

    def __call__(self, x, lambdas, log_weights):
        log_q_s = self.log_q_vec(x, lambdas)
        return logsumexp(log_q_s + log_weights)


class AveragedPotential:
    def __init__(self, u: ReducedPotentialFxn):
        """u_mix(x) = - logsumexp([-u(x, lam) + log_weight for (lam, log_weight) in mixture])"""
        self.u = u
        self.u_vec = vmap(self.u, (None, 0))

    def __call__(self, x, lambdas, log_weights):
        u_s = self.u_vec(x, lambdas)
        return - logsumexp(-u_s + log_weights)
