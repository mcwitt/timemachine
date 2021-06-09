import numpy as np
from typing import Callable, List, Tuple
from md.states import CoordsVelBox

Lambda = LogWeight = float
Schedule = LogWeights = IndexArray = np.array
Samples = List[CoordsVelBox]

Propagator = Callable[[CoordsVelBox, Lambda], CoordsVelBox]
VectorizedPropagator = Callable[[Samples, Lambda], Samples]

LogProb = Callable[[CoordsVelBox, Lambda], LogWeight]
VectorizedLogProb = Callable[[Samples, Lambda], LogWeights]
Resampler = Callable[[LogWeights], Tuple[IndexArray, LogWeights]]


def round_trip_smc(
        samples: Samples,
        lambdas: Schedule,
        propagate: VectorizedPropagator,
        log_prob: VectorizedLogProb,
        resample: Resampler,
) -> Tuple[Samples, LogWeights]:
    """Analogous to NCMC, but using collection of interacting proposal trajectories, rather
    than one proposal trajectory at a time.

    Notes
    -----
    One way to view NCMC is as "round-trip AIS."
        We anneal a single trajectory from p_0 through a sequence of T-1 intermediate distributions
         to complete a "round-trip" and end at p_0 again, accumulating a log importance weight for
         the trajectory as a whole.
        Applying this move to an initial sample x_0 ~ p_0, we produce a new properly weighted sample
            (x_T, log_weight_T) ~ p_0.
        We can then accept/reject x_0 -> x_T with probability min(1, exp(log_weight_T)) to form an
        "NCMC move."

    Rather than working with _a single proposal trajectory_ at a time, we could instead work with a
        _population of interacting trajectories_, in a method we might call "round-trip SMC."
     SMC is generally superior to AIS in terms of variance and opportunities for on-the-fly adaptation.
      (Although this comes at the cost of increased communication compared with the non-interacting trajectories.)

    Acronyms
    --------
    * NCMC - nonequilibrium candidate monte carlo
    * MCMC - markov chain monte carlo
    * SMC - sequential monte carlo
    * AIS - annealed importance sampling

    References
    ----------
    * [Nilmeier et al., 2011] Nonequilibrium Candidate Monte Carlo
        https://www.pnas.org/content/108/45/E1009
    * Arnaud Doucet's annotated bibliography of SMC
        https://www.stats.ox.ac.uk/~doucet/smc_resources.html
    * [Rousey, Dickson, 2020] Enhanced Jarzynski free energy calculations using weighted ensemble
        Demonstration of benefits of resampling in context of nonequilibrium free energy calculations
        https://aip.scitation.org/doi/abs/10.1063/5.0020600
    """
    n = len(samples)
    log_weights = np.zeros(n)

    for (lam_initial, lam_target) in zip(lambdas[:-2], lambdas[1:-1]):
        # update log weights
        log_weights += log_prob(samples, lam_target) - log_prob(samples, lam_initial)

        # resample
        indices, log_weights = resample(log_weights)
        resampled = [samples[i] for i in indices]

        # propagate
        samples = propagate(resampled, lam_target)

    # final result: a collection of samples, with associated log weights
    log_weights += log_prob(samples, lambdas[-1]) - log_prob(samples, lambdas[-2])

    return samples, log_weights
