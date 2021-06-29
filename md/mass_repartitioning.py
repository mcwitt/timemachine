import numpy as np

from jax.config import config;

config.update("jax_enable_x64", True)

from jax import value_and_grad, jit, numpy as jnp
import networkx as nx

from scipy.optimize import minimize, Bounds

from typing import Tuple, List, Callable

Graph = nx.Graph
Array = np.array
Float = float
Int = int


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

NodeIndices = Array
Counts = List[Int]
def get_unique_subgraphs(g: Graph) -> Tuple[List[NodeIndices], Counts]:
    components = list(nx.connected_components(g))

    unique_components = []
    counts = []

    for component in components:
        already_there = False
        for i, unique in enumerate(unique_components):
            if nx.is_isomorphic(nx.subgraph(g, component), nx.subgraph(g, unique)):
                already_there = True
                counts[i] += 1
        if not already_there:
            unique_components.append(component)
            counts.append(1)
    return [np.array(component, dtype=int) for component in unique_components], counts


def get_water_indices(bond_list: List[Tuple[int, int]]) -> np.array:
    """Get array of indices of water atoms, each in the order HHO

    Notes
    -----
    * Assumes (without checking) that any bonded collection of 3 atoms is a water molecule
    """
    topology = nx.Graph(bond_list)
    components = [np.array(list(c)) for c in nx.algorithms.connected_components(topology)]
    is_water = lambda c : len(c) == 3
    waters = list(filter(is_water, components))

    def sort_atoms(unordered_water_atoms):
        """{HOH, OHH, HHO} -> HHO, without hardcoding an assumption on the masses"""
        return sorted(unordered_water_atoms, key=lambda a: len(list(topology.neighbors(a))))

    return np.array(list(map(sort_atoms, waters)))


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


def construct_loss(bond_indices, ks, total_mass) -> Tuple[LossFxn, AtomIndices]:
    atom_indices = np.array(sorted(set(bond_indices.flatten())))
    atom_map = atom_indices_to_dict(atom_indices)
    mapped_bond_indices = apply_atom_map_to_bond_indices(bond_indices, atom_map)

    @jit
    def loss(masses: Array) -> Float:
        """Minimize this to maximize the shortest vibration period.
        Encodes the mass sum constraint (sum(masses) = total_mass) as a penalty.

        Alternate losses considered but not used here:
        * np.sum(frequencies**2) where frequencies = 1 / periods
        """
        normalized_masses = masses / np.sum(masses) * total_mass
        periods = bond_vibration_periods(ks, mapped_bond_indices, normalized_masses)
        mass_sum_penalty = (total_mass - np.sum(masses))**2
        return - np.min(periods) + mass_sum_penalty

    return loss, atom_indices


def maximize_shortest_bond_vibration(bond_indices, ks, total_mass):
    loss, atom_indices = construct_loss(bond_indices, ks, total_mass)
    n = len(atom_indices)

    def fun(masses):
        v, g = value_and_grad(loss)(masses)
        return float(v), np.array(g)

    # bounds to enforce on the optimized masses
    reasonable_range = Bounds(0.25, 100.0)
    initial_masses = total_mass * (np.ones(n) / n)

    result = minimize(
        fun, initial_masses,
        jac=True, tol=0,# method='L-BFGS-B',
        bounds=reasonable_range,
    )
    optimized_masses = result.x
    return optimized_masses / np.sum(optimized_masses) * total_mass


if __name__ == '__main__':
    from testsystems.relative import hif2a_ligand_pair
    from md.builders import build_water_system, build_protein_system
    from fe.free_energy import AbsoluteFreeEnergy
    from md.barostat.utils import get_bond_list
    from time import time

    import matplotlib.pyplot as plt

    mol_a = hif2a_ligand_pair.mol_a
    ff = hif2a_ligand_pair.ff
    #complex_system, complex_coords, complex_box, complex_top = build_water_system(2.0)
    from pathlib import Path

    root = Path(__file__).parent.parent
    path_to_protein = str(root.joinpath('tests/data/hif2a_nowater_min.pdb'))
    complex_system, complex_coords, _, _, complex_box, _ = build_protein_system(path_to_protein)
    afe = AbsoluteFreeEnergy(mol_a, ff)

    unbound_potentials, sys_params, masses, coords = afe.prepare_host_edge(
        ff.get_ordered_params(), complex_system, complex_coords
    )
    harmonic_bond_potential = unbound_potentials[0]

    # get bond indices and ks from force object
    ks = np.array(sys_params[0][:, 0])
    bond_indices = np.array(get_bond_list(harmonic_bond_potential))

    # convert to a graph
    g = arrays_to_graph(bond_indices, ks)

    # extract unique components
    unique_components, counts = get_unique_subgraphs(g)
    # only expecting to see more than one copy of water
    for i, component in enumerate(unique_components):
        if len(component) != 3:
            assert counts[i] == 1

    # for each unique component, get bond_indices, ks, atom_indices

    n_components = len(unique_components)
    print('n_components', n_components)

    plot_index = 1
    plt.figure(figsize=(12, 9))

    whole_system_optimized_masses = np.array(masses)

    bond_list = [tuple(b) for b in bond_indices]
    water_indices = get_water_indices(bond_list)

    for i, component in enumerate(unique_components):
        subgraph = nx.subgraph(g, component)

        subgraph_bond_indices, subgraph_ks = graph_to_arrays(subgraph)

        atom_indices = np.array(sorted(set(subgraph_bond_indices.flatten())))
        n = len(atom_indices)
        atom_map = atom_indices_to_dict(atom_indices)
        mapped_bond_indices = apply_atom_map_to_bond_indices(subgraph_bond_indices, atom_map)

        original_masses = masses[atom_indices]
        total_mass = np.sum(original_masses)
        uniform_masses = np.ones(n) * total_mass / n

        t0 = time()
        optimized_masses = maximize_shortest_bond_vibration(subgraph_bond_indices, subgraph_ks, total_mass)
        t1 = time()
        print(f'optimized masses for {len(optimized_masses)}-atom component in {(t1 - t0):.3f} s')

        whole_system_optimized_masses[atom_indices] = optimized_masses
        # special case code for water. Other option: more generic (but probably slower) graph matcher?
        if n == 3:
            topology = nx.Graph(bond_list)
            sorted_local_atom_indices = np.array(sorted(np.arange(n), key=lambda a: len(list(topology.neighbors(a)))))
            for water in water_indices:
                whole_system_optimized_masses[water_indices] = optimized_masses[sorted_local_atom_indices]

        physical_periods = bond_vibration_periods(subgraph_ks, mapped_bond_indices, original_masses)
        initial_periods = bond_vibration_periods(subgraph_ks, mapped_bond_indices, uniform_masses)
        optimized_periods = bond_vibration_periods(subgraph_ks, mapped_bond_indices, optimized_masses)


        def add_labels():
            plt.ylabel('bond vibration period')
            plt.xlabel('bond')

        ylim = (0, 1.1 * np.max(list(map(max, [physical_periods, initial_periods, optimized_periods]))))

        plt.subplot(n_components, 3, plot_index)
        plot_index += 1
        plt.plot(physical_periods, '.')
        plt.title('physical masses')
        add_labels()
        plt.hlines(min(physical_periods), 0, len(physical_periods), color='grey')
        plt.ylim(*ylim)

        min_physical = np.min(physical_periods)

        plt.subplot(n_components, 3, plot_index)
        plot_index += 1
        plt.plot(initial_periods, '.')
        plt.title(f'uniform masses\n({np.min(initial_periods) / min_physical:.3f}x)')
        add_labels()
        plt.ylim(*ylim)
        plt.hlines(min(initial_periods), 0, len(initial_periods), color='grey')

        plt.subplot(n_components, 3, plot_index)
        plot_index += 1
        plt.plot(optimized_periods, '.')
        plt.title(f'optimized masses per-component\n({np.min(optimized_periods) / min_physical:.3f}x)')
        add_labels()
        plt.ylim(*ylim)
        plt.hlines(min(optimized_periods), 0, len(optimized_periods), color='grey')

        plt.tight_layout()

    plt.savefig('mass_repartitioning.png', dpi=300, bbox_inches='tight')
