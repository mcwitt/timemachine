import numpy as np

from jax.config import config;

config.update("jax_enable_x64", True)

from jax import value_and_grad, jit, numpy as jnp

from scipy.optimize import LinearConstraint
from md.barostat.utils import get_bond_list
import networkx as nx

from scipy.optimize import minimize, Bounds

from typing import Tuple, List, Callable

Graph = nx.Graph
Array = np.array
Float = float


def bond_vibration_periods(ks, bond_indices, masses):
    """2 * pi * sqrt(reduced_masses/ks)"""
    m1, m2 = masses[bond_indices[:, 0]], masses[bond_indices[:, 1]]
    reduced_masses = (m1 * m2) / (m1 + m2)
    return jnp.sqrt(reduced_masses / ks) * (2 * np.pi)


def arrays_to_graph(bond_indices: Array, ks: Array) -> Graph:
    assert len(ks) == len(bond_indices)

    g = nx.Graph()

    for i in range(len(ks)):
        g.add_edge(*bond_indices[i], k=ks[i])

    return g


def graph_to_arrays(g: Graph) -> Tuple[Array, Array]:
    edges = list(g.edges)
    bond_indices = np.array(edges, dtype=np.int32)
    ks = np.array([g.edges[e]['k'] for e in edges])
    return bond_indices, ks


def get_unique_subgraphs(g: Graph) -> List[Graph]:
    components = list(nx.connected_components(g))

    unique_components = []

    for component in components:
        already_there = False
        for unique in unique_components:
            if nx.is_isomorphic(nx.subgraph(g, component), nx.subgraph(g, unique)):
                already_there = True
        if not already_there:
            unique_components.append(component)
    return unique_components


def atom_indices_to_dict(atom_indices):
    """map atom_indices[i] -> i

    to create a contiguous, zero-indexed span of atom indices
    from a list of scattered atom indices
    """
    atom_map = dict()
    for i in range(len(atom_indices)):
        atom_map[atom_indices[i]] = i
    return atom_map


def apply_atom_map_to_bond_indices(bond_indices, atom_map):
    """apply elem -> atom_map[elem] for elem in bond_indices"""

    mapped_bond_indices = np.zeros_like(bond_indices)
    for i in range(len(bond_indices)):
        for j in range(len(bond_indices[i])):
            mapped_bond_indices[i, j] = atom_map[bond_indices[i, j]]

    return mapped_bond_indices


Masses = AtomIndices = Array
LossFxn = Callable[[Masses], Float]


def construct_loss(bond_indices, ks) -> Tuple[LossFxn, AtomIndices]:
    atom_indices = np.array(sorted(set(bond_indices.flatten())))
    atom_map = atom_indices_to_dict(atom_indices)
    mapped_bond_indices = apply_atom_map_to_bond_indices(bond_indices, atom_map)

    @jit
    def loss(masses: Array) -> Float:
        """Minimize this to maximize the shortest vibration period.

        Alternate losses considered but not used here:
        * np.sum(frequencies**2) where frequencies = 1 / periods
        """
        periods = bond_vibration_periods(ks, mapped_bond_indices, masses)
        return - np.min(periods)

    return loss, atom_indices


def maximize_shortest_bond_vibration(bond_indices, ks, total_mass):
    loss, atom_indices = construct_loss(bond_indices, ks)
    n = len(atom_indices)
    A = np.ones((1, n))
    lb = ub = np.ones(1) * total_mass
    sum_constraint = LinearConstraint(A, lb, ub, keep_feasible=False)

    def fun(masses):
        v, g = value_and_grad(loss)(masses)
        return float(v), np.array(g)

    # possible bounds we may want to enforce on the optimized masses
    # non_negative = Bounds(0, np.inf)
    # greater_than_1 = Bounds(1, np.inf)
    reasonable_range = Bounds(0.25, 100.0)

    result = minimize(
        fun, np.ones(n) * total_mass / n,
        jac=True, tol=0,
        bounds=reasonable_range, constraints=sum_constraint,
    )
    return result.x
