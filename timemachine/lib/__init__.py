import numpy as np

from timemachine.lib import custom_ops
from timemachine.integrator import langevin_coefficients

# safe to pickle!

class LangevinIntegrator():

    def __init__(self, temperature, dt, friction, masses, seed):

        self.dt = dt
        self.seed = seed

        ca, cb, cc = langevin_coefficients(temperature, dt, friction, masses)
        cb *= -1
        self.ca = ca
        self.cbs = cb
        self.ccs = cc

    def impl(self):
        return custom_ops.LangevinIntegrator(self.dt, self.ca, self.cbs, self.ccs, self.seed)

class MonteCarloBarostat():

    def __init__(self, N, group_idxs, pressure, temperature, interval, seed):
        self.N = N
        self.group_idxs = group_idxs
        self.pressure = pressure
        self.temperature = temperature
        self.interval = interval
        self.seed = seed

    def impl(self, u_impls):
        return custom_ops.MonteCarloBarostat(
            self.N,
            self.pressure,
            self.temperature,
            self.group_idxs,
            self.interval,
            u_impls,
            self.seed)