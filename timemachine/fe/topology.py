from abc import ABC

import jax
import jax.numpy as jnp
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdmolops

from timemachine.fe import dummy, dummy_draw
from timemachine.fe.dummy import flag_factorizable_bonds, flag_stable_dummy_ixns
from timemachine.ff.handlers import nonbonded
from timemachine.lib import potentials

_SCALE_12 = 1.0
_SCALE_13 = 1.0
_SCALE_14 = 0.5
_BETA = 2.0
_CUTOFF = 1.2


def standard_qlj_typer(mol):
    """
    This function parameterizes the nonbonded terms of a molecule
    in a relatively simple and forcefield independent way. The
    parameters here roughly follow the Smirnoff 1.1.0 Lennard Jones types.

    These values are taken from timemachine/ff/params/smirnoff_1_1_0_cc.py, rounding down
    to two decimal places for sigma and one decimal place for epsilon.

    Note that charges are set to net_formal_charge(mol)/num_atoms.

    Parameters
    ----------
    mol: RDKit.ROMol
        RDKit molecule

    Returns
    -------
    [N,3] array containing (charge, sigma, epsilon)

    """

    standard_qlj = []

    # for charged ligands, we don't want to remove the charge fully as it will
    # introduce large variance in the resulting estimator
    standard_charge = float(rdmolops.GetFormalCharge(mol)) / mol.GetNumAtoms()

    for atom in mol.GetAtoms():
        a_num = atom.GetAtomicNum()
        if a_num == 1:
            assert len(atom.GetNeighbors()) == 1
            neighbor = atom.GetNeighbors()[0]
            b_num = neighbor.GetAtomicNum()
            if b_num == 6:
                val = (standard_charge, 0.25, 0.25)
            elif b_num == 7:
                val = (standard_charge, 0.10, 0.25)
            elif b_num == 8:
                val = (standard_charge, 0.05, 0.02)
            elif b_num == 16:
                val = (standard_charge, 0.10, 0.25)
            else:
                val = (standard_charge, 0.10, 0.25)
        elif a_num == 6:
            val = (standard_charge, 0.34, 0.6)
        elif a_num == 7:
            val = (standard_charge, 0.32, 0.8)
        elif a_num == 8:
            val = (standard_charge, 0.30, 0.9)
        elif a_num == 9:
            val = (standard_charge, 0.3, 0.5)
        elif a_num == 15:
            val = (standard_charge, 0.37, 0.9)
        elif a_num == 16:
            val = (standard_charge, 0.35, 1.0)
        elif a_num == 17:
            val = (standard_charge, 0.35, 1.0)
        elif a_num == 35:
            val = (standard_charge, 0.39, 1.1)
        elif a_num == 53:
            val = (standard_charge, 0.41, 1.2)
        else:
            # print("Unknown a_num", a_num)
            assert 0, "Unknown a_num " + str(a_num)

        # sigmas need to be halved
        standard_qlj.append((val[0], val[1] / 2, val[2]))

    standard_qlj = np.array(standard_qlj)

    return standard_qlj


class AtomMappingError(Exception):
    pass


class UnsupportedPotential(Exception):
    pass


class HostGuestTopology:
    def __init__(self, host_potentials, guest_topology):
        """
        Utility tool for combining host with a guest, in that order. host_potentials must be comprised
        exclusively of supported potentials (currently: bonds, angles, torsions, nonbonded).

        Parameters
        ----------
        host_potentials:
            Bound potentials for the host.

        guest_topology:
            Guest's Topology {Base, Dual, Single}Topology.

        """
        self.guest_topology = guest_topology

        self.host_nonbonded = None
        self.host_harmonic_bond = None
        self.host_harmonic_angle = None
        self.host_periodic_torsion = None

        # (ytz): extra assertions inside are to ensure we don't have duplicate terms
        for bp in host_potentials:
            if isinstance(bp, potentials.HarmonicBond):
                assert self.host_harmonic_bond is None
                self.host_harmonic_bond = bp
            elif isinstance(bp, potentials.HarmonicAngle):
                assert self.host_harmonic_angle is None
                self.host_harmonic_angle = bp
            elif isinstance(bp, potentials.PeriodicTorsion):
                assert self.host_periodic_torsion is None
                self.host_periodic_torsion = bp
            elif isinstance(bp, potentials.Nonbonded):
                assert self.host_nonbonded is None
                self.host_nonbonded = bp
            else:
                raise UnsupportedPotential("Unsupported host potential")

        self.num_host_atoms = len(self.host_nonbonded.get_lambda_plane_idxs())

    def get_num_atoms(self):
        return self.num_host_atoms + self.guest_topology.get_num_atoms()

    # tbd: just merge the hamiltonians here
    def _parameterize_bonded_term(self, guest_params, guest_potential, host_potential):

        if guest_potential is None:
            raise UnsupportedPotential("Mismatch in guest_potential")

        # (ytz): corner case exists if the guest_potential is None
        if host_potential is not None:
            assert type(host_potential) == type(guest_potential)

        guest_idxs = guest_potential.get_idxs() + self.num_host_atoms

        guest_lambda_mult = guest_potential.get_lambda_mult()
        guest_lambda_offset = guest_potential.get_lambda_offset()

        if guest_lambda_mult is None:
            guest_lambda_mult = np.zeros(len(guest_params))
        if guest_lambda_offset is None:
            guest_lambda_offset = np.ones(len(guest_params))

        if host_potential is not None:
            # the host is always on.
            host_params = host_potential.params
            host_idxs = host_potential.get_idxs()
            host_lambda_mult = jnp.zeros(len(host_idxs), dtype=np.int32)
            host_lambda_offset = jnp.ones(len(host_idxs), dtype=np.int32)
        else:
            # (ytz): this extra jank is to work around jnp.concatenate not supporting empty lists.
            host_params = np.array([], dtype=guest_params.dtype).reshape((-1, guest_params.shape[1]))
            host_idxs = np.array([], dtype=guest_idxs.dtype).reshape((-1, guest_idxs.shape[1]))
            host_lambda_mult = []
            host_lambda_offset = []

        combined_params = jnp.concatenate([host_params, guest_params])
        combined_idxs = np.concatenate([host_idxs, guest_idxs])
        combined_lambda_mult = np.concatenate([host_lambda_mult, guest_lambda_mult]).astype(np.int32)
        combined_lambda_offset = np.concatenate([host_lambda_offset, guest_lambda_offset]).astype(np.int32)

        ctor = type(guest_potential)

        return combined_params, ctor(combined_idxs, combined_lambda_mult, combined_lambda_offset)

    def parameterize_harmonic_bond(self, ff_params):
        guest_params, guest_potential = self.guest_topology.parameterize_harmonic_bond(ff_params)
        return self._parameterize_bonded_term(guest_params, guest_potential, self.host_harmonic_bond)

    def parameterize_harmonic_angle(self, ff_params):
        guest_params, guest_potential = self.guest_topology.parameterize_harmonic_angle(ff_params)
        return self._parameterize_bonded_term(guest_params, guest_potential, self.host_harmonic_angle)

    def parameterize_periodic_torsion(self, proper_params, improper_params):
        guest_params, guest_potential = self.guest_topology.parameterize_periodic_torsion(
            proper_params, improper_params
        )
        return self._parameterize_bonded_term(guest_params, guest_potential, self.host_periodic_torsion)

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        num_guest_atoms = self.guest_topology.get_num_atoms()
        guest_qlj, guest_p = self.guest_topology.parameterize_nonbonded(ff_q_params, ff_lj_params)

        if isinstance(guest_p, potentials.NonbondedInterpolated):
            assert guest_qlj.shape[0] == num_guest_atoms * 2
            is_interpolated = True
        else:
            assert guest_qlj.shape[0] == num_guest_atoms
            is_interpolated = False

        # see if we're doing parameter interpolation
        assert guest_qlj.shape[1] == 3
        assert guest_p.get_beta() == self.host_nonbonded.get_beta()
        assert guest_p.get_cutoff() == self.host_nonbonded.get_cutoff()

        hg_exclusion_idxs = np.concatenate(
            [self.host_nonbonded.get_exclusion_idxs(), guest_p.get_exclusion_idxs() + self.num_host_atoms]
        )
        hg_scale_factors = np.concatenate([self.host_nonbonded.get_scale_factors(), guest_p.get_scale_factors()])
        hg_lambda_offset_idxs = np.concatenate(
            [self.host_nonbonded.get_lambda_offset_idxs(), guest_p.get_lambda_offset_idxs()]
        )
        hg_lambda_plane_idxs = np.concatenate(
            [self.host_nonbonded.get_lambda_plane_idxs(), guest_p.get_lambda_plane_idxs()]
        )

        if is_interpolated:
            # with parameter interpolation
            hg_nb_params_src = jnp.concatenate([self.host_nonbonded.params, guest_qlj[:num_guest_atoms]])
            hg_nb_params_dst = jnp.concatenate([self.host_nonbonded.params, guest_qlj[num_guest_atoms:]])
            hg_nb_params = jnp.concatenate([hg_nb_params_src, hg_nb_params_dst])

            nb = potentials.NonbondedInterpolated(
                hg_exclusion_idxs,
                hg_scale_factors,
                hg_lambda_plane_idxs,
                hg_lambda_offset_idxs,
                guest_p.get_beta(),
                guest_p.get_cutoff(),
            )

            return hg_nb_params, nb
        else:
            # no parameter interpolation
            hg_nb_params = jnp.concatenate([self.host_nonbonded.params, guest_qlj])

            return hg_nb_params, potentials.Nonbonded(
                hg_exclusion_idxs,
                hg_scale_factors,
                hg_lambda_plane_idxs,
                hg_lambda_offset_idxs,
                guest_p.get_beta(),
                guest_p.get_cutoff(),
            )


class BaseTopology:
    def __init__(self, mol, forcefield):
        """
        Utility for working with a single ligand.

        Parameter
        ---------
        mol: ROMol
            Ligand to be parameterized

        forcefield: ff.Forcefield
            A convenience wrapper for forcefield lists.

        """
        self.mol = mol
        self.ff = forcefield

    def get_num_atoms(self):
        return self.mol.GetNumAtoms()

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        q_params = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol)
        lj_params = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol)

        exclusion_idxs, scale_factors = nonbonded.generate_exclusion_idxs(
            self.mol, scale12=_SCALE_12, scale13=_SCALE_13, scale14=_SCALE_14
        )

        scale_factors = np.stack([scale_factors, scale_factors], axis=1)

        N = len(q_params)

        lambda_plane_idxs = np.zeros(N, dtype=np.int32)
        lambda_offset_idxs = np.ones(N, dtype=np.int32)

        beta = _BETA
        cutoff = _CUTOFF  # solve for this analytically later

        nb = potentials.Nonbonded(exclusion_idxs, scale_factors, lambda_plane_idxs, lambda_offset_idxs, beta, cutoff)

        params = jnp.concatenate([jnp.reshape(q_params, (-1, 1)), jnp.reshape(lj_params, (-1, 2))], axis=1)

        return params, nb

    def parameterize_harmonic_bond(self, ff_params):
        params, idxs = self.ff.hb_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.HarmonicBond(idxs)

    def parameterize_harmonic_angle(self, ff_params):
        params, idxs = self.ff.ha_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.HarmonicAngle(idxs)

    def parameterize_proper_torsion(self, ff_params):
        params, idxs = self.ff.pt_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.PeriodicTorsion(idxs)

    def parameterize_improper_torsion(self, ff_params):
        params, idxs = self.ff.it_handle.partial_parameterize(ff_params, self.mol)
        return params, potentials.PeriodicTorsion(idxs)

    def parameterize_periodic_torsion(self, proper_params, improper_params):
        """
        Parameterize all periodic torsions in the system.
        """
        proper_params, proper_potential = self.parameterize_proper_torsion(proper_params)
        improper_params, improper_potential = self.parameterize_improper_torsion(improper_params)
        combined_params = jnp.concatenate([proper_params, improper_params])
        combined_idxs = np.concatenate([proper_potential.get_idxs(), improper_potential.get_idxs()])

        proper_lambda_mult = proper_potential.get_lambda_mult()
        proper_lambda_offset = proper_potential.get_lambda_offset()

        if proper_lambda_mult is None:
            proper_lambda_mult = np.zeros(len(proper_params))
        if proper_lambda_offset is None:
            proper_lambda_offset = np.ones(len(proper_params))

        improper_lambda_mult = improper_potential.get_lambda_mult()
        improper_lambda_offset = improper_potential.get_lambda_offset()

        if improper_lambda_mult is None:
            improper_lambda_mult = np.zeros(len(improper_params))
        if improper_lambda_offset is None:
            improper_lambda_offset = np.ones(len(improper_params))

        combined_lambda_mult = np.concatenate([proper_lambda_mult, improper_lambda_mult]).astype(np.int32)
        combined_lambda_offset = np.concatenate([proper_lambda_offset, improper_lambda_offset]).astype(np.int32)

        combined_potential = potentials.PeriodicTorsion(combined_idxs, combined_lambda_mult, combined_lambda_offset)
        return combined_params, combined_potential


class BaseTopologyConversion(BaseTopology):
    """
    Converts a single ligand into a standard, forcefield independent state. The ligand has its 4D
    coordinate set to zero at all times, so that it will be fully interacting with the host. The
    ligand's nonbonded parameters are interpolated such that the charges goto zero, and the lennard
    jones parameters goto a standard, forcefield independent state.
    """

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):

        qlj_params, nb_potential = super().parameterize_nonbonded(ff_q_params, ff_lj_params)
        src_qlj_params = qlj_params
        dst_qlj_params = standard_qlj_typer(self.mol)

        combined_qlj_params = jnp.concatenate([src_qlj_params, dst_qlj_params])
        lambda_plane_idxs = np.zeros(self.mol.GetNumAtoms(), dtype=np.int32)
        lambda_offset_idxs = np.zeros(self.mol.GetNumAtoms(), dtype=np.int32)

        interpolated_potential = nb_potential.interpolate()
        interpolated_potential.set_lambda_plane_idxs(lambda_plane_idxs)
        interpolated_potential.set_lambda_offset_idxs(lambda_offset_idxs)

        return combined_qlj_params, interpolated_potential


class BaseTopologyStandardDecoupling(BaseTopology):
    """
    Decouple a standard ligand from the environment.

    lambda=0 is the fully interacting state.
    lambda=1 is the non-interacting state.
    """

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):

        # mol is standardized into a forcefield independent state.
        _, nb_potential = super().parameterize_nonbonded(ff_q_params, ff_lj_params)
        qlj_params = standard_qlj_typer(self.mol)

        return qlj_params, nb_potential


class DualTopology(ABC):
    def __init__(self, mol_a, mol_b, forcefield):
        """
        Utility for working with two ligands via dual topology. Both copies of the ligand
        will be present after merging.

        Parameter
        ---------
        mol_a: ROMol
            First ligand to be parameterized

        mol_b: ROMol
            Second ligand to be parameterized

        forcefield: ff.Forcefield
            A convenience wrapper for forcefield lists.

        """
        self.mol_a = mol_a
        self.mol_b = mol_b
        self.ff = forcefield

    def get_num_atoms(self):
        return self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms()

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):

        # dummy is either "a or "b"
        q_params_a = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol_a)
        q_params_b = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol_b)
        lj_params_a = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol_a)
        lj_params_b = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol_b)

        q_params = jnp.concatenate([q_params_a, q_params_b])
        lj_params = jnp.concatenate([lj_params_a, lj_params_b])

        exclusion_idxs_a, scale_factors_a = nonbonded.generate_exclusion_idxs(
            self.mol_a, scale12=_SCALE_12, scale13=_SCALE_13, scale14=_SCALE_14
        )

        exclusion_idxs_b, scale_factors_b = nonbonded.generate_exclusion_idxs(
            self.mol_b, scale12=_SCALE_12, scale13=_SCALE_13, scale14=_SCALE_14
        )

        mutual_exclusions = []
        mutual_scale_factors = []

        NA = self.mol_a.GetNumAtoms()
        NB = self.mol_b.GetNumAtoms()

        for i in range(NA):
            for j in range(NB):
                mutual_exclusions.append([i, j + NA])
                mutual_scale_factors.append([1.0, 1.0])

        mutual_exclusions = np.array(mutual_exclusions)
        mutual_scale_factors = np.array(mutual_scale_factors)

        combined_exclusion_idxs = np.concatenate([exclusion_idxs_a, exclusion_idxs_b + NA, mutual_exclusions]).astype(
            np.int32
        )

        combined_scale_factors = np.concatenate(
            [
                np.stack([scale_factors_a, scale_factors_a], axis=1),
                np.stack([scale_factors_b, scale_factors_b], axis=1),
                mutual_scale_factors,
            ]
        ).astype(np.float64)

        combined_lambda_plane_idxs = None
        combined_lambda_offset_idxs = None

        beta = _BETA
        cutoff = _CUTOFF  # solve for this analytically later

        qlj_params = jnp.concatenate([jnp.reshape(q_params, (-1, 1)), jnp.reshape(lj_params, (-1, 2))], axis=1)

        return qlj_params, potentials.Nonbonded(
            combined_exclusion_idxs,
            combined_scale_factors,
            combined_lambda_plane_idxs,
            combined_lambda_offset_idxs,
            beta,
            cutoff,
        )

    def _parameterize_bonded_term(self, ff_params, bonded_handle, potential):
        offset = self.mol_a.GetNumAtoms()
        params_a, idxs_a = bonded_handle.partial_parameterize(ff_params, self.mol_a)
        params_b, idxs_b = bonded_handle.partial_parameterize(ff_params, self.mol_b)
        params_c = jnp.concatenate([params_a, params_b])
        idxs_c = np.concatenate([idxs_a, idxs_b + offset])
        return params_c, potential(idxs_c)

    def parameterize_harmonic_bond(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.hb_handle, potentials.HarmonicBond)

    def parameterize_harmonic_angle(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.ha_handle, potentials.HarmonicAngle)

    def parameterize_periodic_torsion(self, proper_params, improper_params):
        """
        Parameterize all periodic torsions in the system.
        """
        proper_params, proper_potential = self.parameterize_proper_torsion(proper_params)
        improper_params, improper_potential = self.parameterize_improper_torsion(improper_params)

        combined_params = jnp.concatenate([proper_params, improper_params])
        combined_idxs = np.concatenate([proper_potential.get_idxs(), improper_potential.get_idxs()])

        proper_lambda_mult = proper_potential.get_lambda_mult()
        proper_lambda_offset = proper_potential.get_lambda_offset()

        if proper_lambda_mult is None:
            proper_lambda_mult = np.zeros(len(proper_params))
        if proper_lambda_offset is None:
            proper_lambda_offset = np.ones(len(proper_params))

        improper_lambda_mult = improper_potential.get_lambda_mult()
        improper_lambda_offset = improper_potential.get_lambda_offset()

        if improper_lambda_mult is None:
            improper_lambda_mult = np.zeros(len(improper_params))
        if improper_lambda_offset is None:
            improper_lambda_offset = np.ones(len(improper_params))

        combined_lambda_mult = np.concatenate([proper_lambda_mult, improper_lambda_mult]).astype(np.int32)
        combined_lambda_offset = np.concatenate([proper_lambda_offset, improper_lambda_offset]).astype(np.int32)

        combined_potential = potentials.PeriodicTorsion(combined_idxs, combined_lambda_mult, combined_lambda_offset)
        return combined_params, combined_potential

    def parameterize_proper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.pt_handle, potentials.PeriodicTorsion)

    def parameterize_improper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.it_handle, potentials.PeriodicTorsion)


class BaseTopologyRHFE(BaseTopology):
    pass


# non-ring torsions are just always turned off at the end-states in the hydration
# free energy test
class DualTopologyRHFE(DualTopology):

    """
    Utility class used for relative hydration free energies. Ligand B is decoupled as lambda goes
    from 0 to 1, while ligand A is fully coupled. At the same time, at lambda=0, ligand B and ligand A
    have their charges and epsilons reduced by half.
    """

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):

        qlj_params, nb_potential = super().parameterize_nonbonded(ff_q_params, ff_lj_params)

        # halve the strength of the charge and the epsilon parameters
        src_qlj_params = jax.ops.index_update(qlj_params, jax.ops.index[:, 0], qlj_params[:, 0] * 0.5)
        src_qlj_params = jax.ops.index_update(src_qlj_params, jax.ops.index[:, 2], qlj_params[:, 2] * 0.5)
        dst_qlj_params = qlj_params
        combined_qlj_params = jnp.concatenate([src_qlj_params, dst_qlj_params])

        combined_lambda_plane_idxs = np.zeros(self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms(), dtype=np.int32)
        combined_lambda_offset_idxs = np.concatenate(
            [np.zeros(self.mol_a.GetNumAtoms(), dtype=np.int32), np.ones(self.mol_b.GetNumAtoms(), dtype=np.int32)]
        )

        nb_potential.set_lambda_plane_idxs(combined_lambda_plane_idxs)
        nb_potential.set_lambda_offset_idxs(combined_lambda_offset_idxs)

        return combined_qlj_params, nb_potential.interpolate()


class DualTopologyStandardDecoupling(DualTopology):
    """
    Standardized variant, where both ligands A and B have their charges, sigmas, and epsilons set
    to standard, forcefield-independent values. There is no parameter interpolation.

    lambda=0 has both ligand A and B fully in the pocket.
    lambda=1 has ligand B fully decoupled, while ligand A is fully interacting.

    """

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):

        # both mol_a and mol_b are standardized.
        _, nb_potential = super().parameterize_nonbonded(ff_q_params, ff_lj_params)

        qlj_params_a = standard_qlj_typer(self.mol_a)

        # src_qlj_params corresponds to the super state where both ligands are interacting with the environment
        # we scale down the charges and epsilons to half their values so as to roughly mimic net one molecule's
        # worth of nonbonded interactions.

        # dst_qlj_params corresponds to the end-state where only one of the molecule interacts with the binding pocket.
        src_qlj_params_a = jax.ops.index_update(qlj_params_a, jax.ops.index[:, 0], qlj_params_a[:, 0] * 0.5)
        src_qlj_params_a = jax.ops.index_update(src_qlj_params_a, jax.ops.index[:, 2], src_qlj_params_a[:, 2] * 0.5)
        dst_qlj_params_a = qlj_params_a

        qlj_params_b = standard_qlj_typer(self.mol_b)
        src_qlj_params_b = jax.ops.index_update(qlj_params_b, jax.ops.index[:, 0], qlj_params_b[:, 0] * 0.5)
        src_qlj_params_b = jax.ops.index_update(src_qlj_params_b, jax.ops.index[:, 2], src_qlj_params_b[:, 2] * 0.5)
        dst_qlj_params_b = qlj_params_b

        src_qlj_params = jnp.concatenate([src_qlj_params_a, src_qlj_params_b])
        dst_qlj_params = jnp.concatenate([dst_qlj_params_a, dst_qlj_params_b])

        combined_qlj_params = jnp.concatenate([src_qlj_params, dst_qlj_params])

        interpolated_potential = nb_potential.interpolate()
        combined_lambda_plane_idxs = np.zeros(self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms(), dtype=np.int32)
        combined_lambda_offset_idxs = np.concatenate(
            [np.zeros(self.mol_a.GetNumAtoms(), dtype=np.int32), np.ones(self.mol_b.GetNumAtoms(), dtype=np.int32)]
        )
        interpolated_potential.set_lambda_plane_idxs(combined_lambda_plane_idxs)
        interpolated_potential.set_lambda_offset_idxs(combined_lambda_offset_idxs)

        return combined_qlj_params, interpolated_potential


class DualTopologyMinimization(DualTopology):
    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):

        # both mol_a and mol_b are standardized.
        # we don't actually need derivatives for this stage.
        qlj_params, nb_potential = super().parameterize_nonbonded(ff_q_params, ff_lj_params)

        N_A, N_B = self.mol_a.GetNumAtoms(), self.mol_b.GetNumAtoms()
        combined_lambda_plane_idxs = np.zeros(N_A + N_B, dtype=np.int32)
        combined_lambda_offset_idxs = np.ones(N_A + N_B, dtype=np.int32)

        nb_potential.set_lambda_offset_idxs(combined_lambda_offset_idxs)
        nb_potential.set_lambda_plane_idxs(combined_lambda_plane_idxs)

        return qlj_params, nb_potential


def ordered_tuple(ixn):
    if ixn[0] > ixn[-1]:
        return tuple(ixn[::-1])
    else:
        return tuple(ixn)


class SingleTopologyV2:
    def __init__(self, mol_a, mol_b, core, ff):

        # test for uniqueness in core idxs for each mol
        assert len(set(tuple(core[:, 0]))) == len(core[:, 0])
        assert len(set(tuple(core[:, 1]))) == len(core[:, 1])

        self.mol_a = mol_a
        self.mol_b = mol_b
        self.ff = ff
        self.core = core

        self.a_to_c = np.arange(mol_a.GetNumAtoms(), dtype=np.int32)  # identity
        self.b_to_c = np.zeros(mol_b.GetNumAtoms(), dtype=np.int32) - 1

        self.NC = mol_a.GetNumAtoms() + mol_b.GetNumAtoms() - len(core)

        # mark membership in the combined atom:
        # 0: Core
        # 1: Dummy A
        # 2: Dummy B
        self.c_flags = np.ones(self.NC, dtype=np.int32)

        for a, b in core:
            self.c_flags[a] = 0
            self.b_to_c[b] = a

        # increment atoms in the second mol
        iota = self.mol_a.GetNumAtoms()
        for b_idx, c_idx in enumerate(self.b_to_c):
            if c_idx == -1:
                self.b_to_c[b_idx] = iota
                self.c_flags[iota] = 2
                iota += 1

    @staticmethod
    def _transform_indices(bond_idxs, a_to_c):

        updated_idxs = []
        for atom_idxs in bond_idxs:
            new_atom_idxs = []
            for a_idx in atom_idxs:
                new_atom_idxs.append(a_to_c[a_idx])
            updated_idxs.append(ordered_tuple(new_atom_idxs))

        return updated_idxs

    def _update_b(self, bond_idxs_b):
        """
        Update indices of b using b_to_c mapping. This also canonicalizes
        the bond indices in the new chimeric molecule.
        """
        updated_idxs = []
        for atom_idxs in bond_idxs_b:
            new_atom_idxs = []
            for a_idx in atom_idxs:
                new_atom_idxs.append(self.b_to_c[a_idx])
            updated_idxs.append(ordered_tuple(new_atom_idxs))

        return updated_idxs

    def interpolate_params(self, params_a, params_b):
        """
        Interpolate two sets of per-particle parameters.

        This can be used to interpolate masses, coordinates, etc.

        Parameters
        ----------
        params_a: np.ndarray, shape [N_A, ...]
            Parameters for the mol_a

        params_b: np.ndarray, shape [N_B, ...]
            Parameters for the mol_b

        Returns
        -------
        tuple: (src, dst)
            Two np.ndarrays each of shape [N_C, ...]

        """

        src_params = [None] * self.NC
        dst_params = [None] * self.NC

        for a_idx, c_idx in enumerate(self.a_to_c):
            src_params[c_idx] = params_a[a_idx]
            dst_params[c_idx] = params_a[a_idx]

        for b_idx, c_idx in enumerate(self.b_to_c):
            dst_params[c_idx] = params_b[b_idx]
            if src_params[c_idx] is None:
                src_params[c_idx] = params_b[b_idx]

        return np.array(src_params), np.array(dst_params)

    @staticmethod
    def _fuse_bond_tables(bond_idxs_a, bond_params_a, bond_idxs_b, bond_params_b):
        """
        bond_idxs_a and bond_idxs_b take transformed indices identities, and they should be
        sorted.

        fuse dummy atoms in b unto a
        """
        # bond_idxs may have duplicates:
        # eg: for 1-2 terms (harmonic bond and nonbonded)
        #     for 1-4 terms (periodic torsions with multiple components

        # sanity check ordering
        for atom_idxs in bond_idxs_a:
            assert atom_idxs[0] < atom_idxs[-1]
        for atom_idxs in bond_idxs_b:
            assert atom_idxs[0] < atom_idxs[-1]

        bond_idxs_c = []
        bond_params_c = []

        for atom_idxs, params in zip(bond_idxs_a, bond_params_a):
            assert atom_idxs[0] < atom_idxs[-1]
            bond_idxs_c.append(atom_idxs)
            bond_params_c.append(params)

        for atom_idxs, params in zip(bond_idxs_b, bond_params_b):
            assert atom_idxs[0] < atom_idxs[-1]
            if atom_idxs not in bond_idxs_a:
                bond_idxs_c.append(atom_idxs)
                bond_params_c.append(params)

        return bond_idxs_c, bond_params_c

    @staticmethod
    def _generate_hybrid_mol_impl(mol_a, a_to_c, mol_b, b_to_c):

        atom_kv = {}

        for atom in mol_b.GetAtoms():
            atom_kv[b_to_c[atom.GetIdx()]] = Chem.Atom(atom)

        for atom in mol_a.GetAtoms():
            atom_kv[a_to_c[atom.GetIdx()]] = Chem.Atom(atom)

        bond_idxs_a = [(x.GetBeginAtomIdx(), x.GetEndAtomIdx()) for x in mol_a.GetBonds()]
        bond_params_a = [x.GetBondType() for x in mol_a.GetBonds()]

        bond_idxs_b = [(x.GetBeginAtomIdx(), x.GetEndAtomIdx()) for x in mol_b.GetBonds()]
        bond_params_b = [x.GetBondType() for x in mol_b.GetBonds()]

        bond_idxs_a = SingleTopologyV2._transform_indices(bond_idxs_a, a_to_c)
        bond_idxs_b = SingleTopologyV2._transform_indices(bond_idxs_b, b_to_c)

        bond_idxs_c, bond_params_c = SingleTopologyV2._fuse_bond_tables(
            bond_idxs_a, bond_params_a, bond_idxs_b, bond_params_b
        )

        hybrid_mol = Chem.RWMol()
        for atom_idx in sorted(atom_kv):
            hybrid_mol.AddAtom(atom_kv[atom_idx])

        for (i, j), bond_type in zip(bond_idxs_c, bond_params_c):
            hybrid_mol.AddBond(int(i), int(j), bond_type)

        # make immutable
        return Chem.Mol(hybrid_mol)

    def generate_hybrid_mol_src(self):
        return self._generate_hybrid_mol_impl(self.mol_a, self.a_to_c, self.mol_b, self.b_to_c)

    def generate_hybrid_mol_dst(self):
        return self._generate_hybrid_mol_impl(self.mol_b, self.b_to_c, self.mol_a, self.a_to_c)

    def _draw_dummy_ixns(self):

        hal = [self.core[:, 0].tolist(), self.core[:, 1].tolist()]
        legends = ["mol_a", "mol_b"]
        res = Draw.MolsToGridImage([self.mol_a, self.mol_b], highlightAtomLists=hal, legends=legends, useSVG=True)
        with open("mols.svg", "w") as fh:
            fh.write(res)

        src, dst = self._parameterize_bonds()
        src_12, _, src_13, _, src_14, _ = src
        src_bond_idxs = [x for x in src_12] + [x for x in src_13] + [x for x in src_14]
        src_mol = self.generate_hybrid_mol_src()

        dst_12, _, dst_13, _, dst_14, _ = dst
        dst_bond_idxs = [x for x in dst_12] + [x for x in dst_13] + [x for x in dst_14]
        dst_mol = self.generate_hybrid_mol_dst()

        core_idxs_src = [x for x in range(self.NC) if (self.c_flags[x] == 1 or self.c_flags[x] == 0)]  # A + C dummy
        core_idxs_dst = [x for x in range(self.NC) if (self.c_flags[x] == 2 or self.c_flags[x] == 0)]  # B + C dummy

        hal = [core_idxs_src, core_idxs_dst]
        res = Draw.MolsToGridImage(
            [src_mol, dst_mol], highlightAtomLists=hal, legends=["src_mol", "dst_mol"], useSVG=True
        )
        with open("src_dst_mol.svg", "w") as fh:
            fh.write(res)

        dgs, ags, ag_ixns = dummy.generate_optimal_dg_ag_pairs(core_idxs_src, src_bond_idxs)

        for idx, (dummy_group, anchor_group, anchor_ixns) in enumerate(zip(dgs, ags, ag_ixns)):

            matched_ixns = []
            for idxs in src_bond_idxs:
                if tuple(idxs) in anchor_ixns:
                    if np.all([ii in dummy_group for ii in idxs]):
                        continue
                    elif np.all([ii in core_idxs_src for ii in idxs]):
                        continue
                    matched_ixns.append(idxs)

            res = dummy_draw.draw_dummy_core_ixns(src_mol, core_idxs_src, matched_ixns, dummy_group)

            with open("debug_src_" + str(idx) + ".svg", "w") as fh:
                fh.write(res)

        dgs, ags, ag_ixns = dummy.generate_optimal_dg_ag_pairs(core_idxs_dst, dst_bond_idxs)

        for idx, (dummy_group, anchor_group, anchor_ixns) in enumerate(zip(dgs, ags, ag_ixns)):

            matched_ixns = []
            for idxs in dst_bond_idxs:
                if tuple(idxs) in anchor_ixns:
                    if np.all([ii in dummy_group for ii in idxs]):
                        continue
                    elif np.all([ii in core_idxs_dst for ii in idxs]):
                        continue
                    matched_ixns.append(idxs)

            res = dummy_draw.draw_dummy_core_ixns(dst_mol, core_idxs_dst, matched_ixns, dummy_group)

            with open("debug_dst_" + str(idx) + ".svg", "w") as fh:
                fh.write(res)

    @staticmethod
    def _process_end_state_ixns(NC, core_idxs, hb_idxs, hb_params, ha_idxs, ha_params, pt_idxs, pt_params):

        keep_angle_flags, keep_torsion_flags = flag_stable_dummy_ixns(
            core_idxs, hb_idxs, hb_params, ha_idxs, ha_params, pt_idxs, pt_params
        )

        bond_idxs = [idxs for idxs in hb_idxs]
        bond_params = [params for params in hb_params]

        for idx, atom_idxs in enumerate(ha_idxs):
            if keep_angle_flags[idx]:
                bond_idxs.append(atom_idxs)
                bond_params.append(ha_params[idx])

        for idx, atom_idxs in enumerate(pt_idxs):
            if keep_torsion_flags[idx]:
                bond_idxs.append(atom_idxs)
                bond_params.append(pt_params[idx])

        keep_flags = flag_factorizable_bonds(core_idxs, bond_idxs)

        pruned_hb_idxs = []
        pruned_hb_params = []

        pruned_ha_idxs = []
        pruned_ha_params = []

        pruned_pt_idxs = []
        pruned_pt_params = []

        for keep, idxs, params in zip(keep_flags, bond_idxs, bond_params):
            if keep:
                if len(idxs) == 2:
                    pruned_hb_idxs.append(idxs)
                    pruned_hb_params.append(params)
                elif len(idxs) == 3:
                    pruned_ha_idxs.append(idxs)
                    pruned_ha_params.append(params)
                elif len(idxs) == 4:
                    pruned_pt_idxs.append(idxs)
                    pruned_pt_params.append(params)
                else:
                    assert 0

        return pruned_hb_idxs, pruned_hb_params, pruned_ha_idxs, pruned_ha_params, pruned_pt_idxs, pruned_pt_params

    @staticmethod
    def _parameterize_bonds_impl(NC, ff, core, mol_a, a_to_c, mol_b, b_to_c):
        hb_params_a, hb_idxs_a = ff.hb_handle.partial_parameterize(ff.hb_handle.params, mol_a)
        hb_params_b, hb_idxs_b = ff.hb_handle.partial_parameterize(ff.hb_handle.params, mol_b)
        hb_idxs_a = SingleTopologyV2._transform_indices(hb_idxs_a, a_to_c)
        hb_idxs_b = SingleTopologyV2._transform_indices(hb_idxs_b, b_to_c)

        ha_params_a, ha_idxs_a = ff.ha_handle.partial_parameterize(ff.ha_handle.params, mol_a)
        ha_params_b, ha_idxs_b = ff.ha_handle.partial_parameterize(ff.ha_handle.params, mol_b)
        ha_idxs_a = SingleTopologyV2._transform_indices(ha_idxs_a, a_to_c)
        ha_idxs_b = SingleTopologyV2._transform_indices(ha_idxs_b, b_to_c)

        pt_params_a, pt_idxs_a = ff.pt_handle.partial_parameterize(ff.pt_handle.params, mol_a)
        pt_params_b, pt_idxs_b = ff.pt_handle.partial_parameterize(ff.pt_handle.params, mol_b)
        pt_idxs_a = SingleTopologyV2._transform_indices(pt_idxs_a, a_to_c)
        pt_idxs_b = SingleTopologyV2._transform_indices(pt_idxs_b, b_to_c)

        # set up potentials to be interpolated on or off
        hb_idxs, hb_params = SingleTopologyV2._fuse_bond_tables(hb_idxs_a, hb_params_a, hb_idxs_b, hb_params_b)
        ha_idxs, ha_params = SingleTopologyV2._fuse_bond_tables(ha_idxs_a, ha_params_a, ha_idxs_b, ha_params_b)
        pt_idxs, pt_params = SingleTopologyV2._fuse_bond_tables(pt_idxs_a, pt_params_a, pt_idxs_b, pt_params_b)

        return SingleTopologyV2._process_end_state_ixns(
            NC, core, hb_idxs, hb_params, ha_idxs, ha_params, pt_idxs, pt_params
        )

    def _parameterize_bonds(self):
        print("PROCESSING SRC")
        core_idxs_src = [x for x in range(self.NC) if (self.c_flags[x] == 1 or self.c_flags[x] == 0)]  # A + C dummy
        src = self._parameterize_bonds_impl(
            self.NC, self.ff, core_idxs_src, self.mol_a, self.a_to_c, self.mol_b, self.b_to_c
        )

        print("PROCESSING DST")
        core_idxs_dst = [x for x in range(self.NC) if (self.c_flags[x] == 2 or self.c_flags[x] == 0)]  # B + C dummy
        dst = self._parameterize_bonds_impl(
            self.NC, self.ff, core_idxs_dst, self.mol_b, self.b_to_c, self.mol_a, self.a_to_c
        )

        return src, dst

    # def _parameterize_bonds(self):

    #     ff = self.ff

    #     hb_params_a, hb_idxs_a = ff.hb_handle.partial_parameterize(ff.hb_handle.params, self.mol_a)
    #     hb_params_b, hb_idxs_b = ff.hb_handle.partial_parameterize(ff.hb_handle.params, self.mol_b)
    #     hb_idxs_b = self._update_b(hb_idxs_b)  # increment indices in the chimeric molecule

    #     ha_params_a, ha_idxs_a = ff.ha_handle.partial_parameterize(ff.ha_handle.params, self.mol_a)
    #     ha_params_b, ha_idxs_b = ff.ha_handle.partial_parameterize(ff.ha_handle.params, self.mol_b)
    #     ha_idxs_b = self._update_b(ha_idxs_b)

    #     pt_params_a, pt_idxs_a = ff.pt_handle.partial_parameterize(ff.pt_handle.params, self.mol_a)
    #     pt_params_b, pt_idxs_b = ff.pt_handle.partial_parameterize(ff.pt_handle.params, self.mol_b)
    #     pt_idxs_b = self._update_b(pt_idxs_b)

    #     # set up potentials to be interpolated on or off
    #     hb_idxs_src, hb_params_src, hb_idxs_dst, hb_params_dst = self._combine_bonded_parameters(
    #         hb_idxs_a, hb_params_a, hb_idxs_b, hb_params_b
    #     )
    #     ha_idxs_src, ha_params_src, ha_idxs_dst, ha_params_dst = self._combine_bonded_parameters(
    #         ha_idxs_a, ha_params_a, ha_idxs_b, ha_params_b
    #     )
    #     pt_idxs_src, pt_params_src, pt_idxs_dst, pt_params_dst = self._combine_bonded_parameters(
    #         pt_idxs_a, pt_params_a, pt_idxs_b, pt_params_b
    #     )

    #     core_idxs_src = [x for x in range(self.NC) if (self.c_flags[x] == 1 or self.c_flags[x] == 0)]  # A + C dummy
    #     core_idxs_dst = [x for x in range(self.NC) if (self.c_flags[x] == 2 or self.c_flags[x] == 0)]  # B + C dummy

    #     (
    #         hb_idxs_src,
    #         hb_params_src,
    #         ha_idxs_src,
    #         ha_params_src,
    #         pt_idxs_src,
    #         pt_params_src,
    #     ) = self._process_end_state_ixns(
    #         core_idxs_src, hb_idxs_src, hb_params_src, ha_idxs_src, ha_params_src, pt_idxs_src, pt_params_src
    #     )

    #     (
    #         hb_idxs_dst,
    #         hb_params_dst,
    #         ha_idxs_dst,
    #         ha_params_dst,
    #         pt_idxs_dst,
    #         pt_params_dst,
    #     ) = self._process_end_state_ixns(
    #         core_idxs_dst, hb_idxs_dst, hb_params_dst, ha_idxs_dst, ha_params_dst, pt_idxs_dst, pt_params_dst
    #     )

    #     # return end-state parameters
    #     return (hb_idxs_src, hb_params_src, ha_idxs_src, ha_params_src, pt_idxs_src, pt_params_src), (
    #         hb_idxs_dst,
    #         hb_params_dst,
    #         ha_idxs_dst,
    #         ha_params_dst,
    #         pt_idxs_dst,
    #         pt_params_dst,
    #     )


class SingleTopology:
    def __init__(self, mol_a, mol_b, core, ff, minimize: bool = False):
        """
        SingleTopology combines two molecules through a common core. The combined mol has
        atom indices laid out such that mol_a is identically mapped to the combined mol indices.
        The atoms in the mol_b's R-group is then glued on to resulting molecule.

        Parameters
        ----------
        mol_a: ROMol
            First ligand

        mol_b: ROMol
            Second ligand

        core: np.array (C, 2)
            Atom mapping from mol_a to to mol_b

        ff: ff.Forcefield
            Forcefield to be used for parameterization.

        minimize: bool
            Whether both R groups should be interacting at lambda=0.5

        """
        self.mol_a = mol_a
        self.mol_b = mol_b
        self.ff = ff
        self.core = core
        self.minimize = minimize

        assert core.shape[1] == 2

        # map into idxs in the combined molecule
        self.a_to_c = np.arange(mol_a.GetNumAtoms(), dtype=np.int32)  # identity
        self.b_to_c = np.zeros(mol_b.GetNumAtoms(), dtype=np.int32) - 1

        self.NC = mol_a.GetNumAtoms() + mol_b.GetNumAtoms() - len(core)

        # mark membership:
        # 0: Core
        # 1: R_A (default)
        # 2: R_B
        self.c_flags = np.ones(self.get_num_atoms(), dtype=np.int32)

        for a, b in core:
            self.c_flags[a] = 0
            self.b_to_c[b] = a

        iota = self.mol_a.GetNumAtoms()
        for b_idx, c_idx in enumerate(self.b_to_c):
            if c_idx == -1:
                self.b_to_c[b_idx] = iota
                self.c_flags[iota] = 2
                iota += 1

        # test for uniqueness in core idxs for each mol
        assert len(set(tuple(core[:, 0]))) == len(core[:, 0])
        assert len(set(tuple(core[:, 1]))) == len(core[:, 1])

        self.assert_factorizability()

    def _identify_offending_core_indices(self):
        """Identifies atoms involved in violations of a factorizability assumption,
        but doesn't immediately raise an error.
        Later, could use this list to:
        * plot / debug
        * if in a "repair_mode", attempt to repair the mapping by removing offending atoms
        * otherwise, raise atom mapping error if any atoms were identified
        """

        # Test that R-groups can be properly factorized out in the proposed
        # mapping. The requirement is that R-groups must be branched from exactly
        # a single atom on the core.

        offending_core_indices = []

        # first convert to a dense graph
        N = self.get_num_atoms()
        dense_graph = np.zeros((N, N), dtype=np.int32)

        for bond in self.mol_a.GetBonds():
            i, j = self.a_to_c[bond.GetBeginAtomIdx()], self.a_to_c[bond.GetEndAtomIdx()]
            dense_graph[i, j] = 1
            dense_graph[j, i] = 1

        for bond in self.mol_b.GetBonds():
            i, j = self.b_to_c[bond.GetBeginAtomIdx()], self.b_to_c[bond.GetEndAtomIdx()]
            dense_graph[i, j] = 1
            dense_graph[j, i] = 1

            # sparsify to simplify and speed up traversal code
        sparse_graph = []
        for row in dense_graph:
            nbs = []
            for col_idx, col in enumerate(row):
                if col == 1:
                    nbs.append(col_idx)
            sparse_graph.append(nbs)

        def visit(i, visited):
            if i in visited:
                return
            else:
                visited.add(i)
                if self.c_flags[i] != 0:
                    for nb in sparse_graph[i]:
                        visit(nb, visited)
                else:
                    return

        for c_idx, group in enumerate(self.c_flags):
            # 0 core, 1 R_A, 2: R_B
            if group != 0:
                seen = set()
                visit(c_idx, seen)
                # (ytz): exactly one of seen should belong to core
                if np.sum(np.array([self.c_flags[x] for x in seen]) == 0) != 1:
                    offending_core_indices.append(c_idx)

        return offending_core_indices

    def assert_factorizability(self):
        """
        Number of atoms in the combined mol

        TODO: add a reference to Boresch paper describing the assumption being checked
        """
        offending_core_indices = self._identify_offending_core_indices()
        num_problems = len(offending_core_indices)
        if num_problems > 0:

            # TODO: revisit how to get atom pair indices -- this goes out of bounds
            # bad_pairs = [tuple(self.core[c_index]) for c_index in offending_core_indices]

            message = f"""Atom Mapping Error: the resulting map is non-factorizable!
            (The map contained  {num_problems} violations of the factorizability assumption.)
            """
            raise AtomMappingError(message)

    def get_num_atoms(self):
        return self.NC

    def interpolate_params(self, params_a, params_b):
        """
        Interpolate two sets of per-particle parameters.

        This can be used to interpolate masses, coordinates, etc.

        Parameters
        ----------
        params_a: np.ndarray, shape [N_A, ...]
            Parameters for the mol_a

        params_b: np.ndarray, shape [N_B, ...]
            Parameters for the mol_b

        Returns
        -------
        tuple: (src, dst)
            Two np.ndarrays each of shape [N_C, ...]

        """

        src_params = [None] * self.get_num_atoms()
        dst_params = [None] * self.get_num_atoms()

        for a_idx, c_idx in enumerate(self.a_to_c):
            src_params[c_idx] = params_a[a_idx]
            dst_params[c_idx] = params_a[a_idx]

        for b_idx, c_idx in enumerate(self.b_to_c):
            dst_params[c_idx] = params_b[b_idx]
            if src_params[c_idx] is None:
                src_params[c_idx] = params_b[b_idx]

        return jnp.array(src_params), jnp.array(dst_params)

    def interpolate_nonbonded_params(self, params_a, params_b):
        """
        Special interpolation method for nonbonded parameters. For R-group atoms,
        their charges and vdw eps parameters are scaled to zero. Vdw sigma
        remains unchanged. This method is needed in order to ensure that R-groups
        that branch from multiple distinct attachment points are fully non-interacting
        to allow for factorization of the partition function. In order words, this function
        implements essentially the non-softcore part of parameter interpolation.

        Parameters
        ----------
        params_a: np.ndarray, shape [N_A, 3]
            Nonbonded parameters for the mol_a

        params_b: np.ndarray, shape [N_B, 3]
            Nonbonded parameters for the mol_b

        Returns
        -------
        tuple: (src, dst)
            Two np.ndarrays each of shape [N_C, ...]

        """

        src_params = [None] * self.get_num_atoms()
        dst_params = [None] * self.get_num_atoms()

        # src -> dst is turning off the parameter
        for a_idx, c_idx in enumerate(self.a_to_c):
            params = params_a[a_idx]
            src_params[c_idx] = params
            if self.c_flags[c_idx] != 0:
                assert self.c_flags[c_idx] == 1
                dst_params[c_idx] = jnp.array([0, params[1], 0])  # q, sig, eps

        # b is initially decoupled
        for b_idx, c_idx in enumerate(self.b_to_c):
            params = params_b[b_idx]
            dst_params[c_idx] = params
            # this will already be processed when looping over a
            if self.c_flags[c_idx] == 0:
                assert src_params[c_idx] is not None
            else:
                assert self.c_flags[c_idx] == 2
                src_params[c_idx] = jnp.array([0, params[1], 0])  # q, sig, eps

        return jnp.array(src_params), jnp.array(dst_params)

    def parameterize_nonbonded(self, ff_q_params, ff_lj_params):
        # Nonbonded potentials combine through parameter interpolation, not energy interpolation.
        # They may or may not operate through 4D decoupling depending on the atom mapping. If an atom is
        # unique, it is kept at full strength and not switched off.

        q_params_a = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol_a)
        q_params_b = self.ff.q_handle.partial_parameterize(ff_q_params, self.mol_b)  # HARD TYPO
        lj_params_a = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol_a)
        lj_params_b = self.ff.lj_handle.partial_parameterize(ff_lj_params, self.mol_b)

        qlj_params_a = jnp.concatenate([jnp.reshape(q_params_a, (-1, 1)), jnp.reshape(lj_params_a, (-1, 2))], axis=1)
        qlj_params_b = jnp.concatenate([jnp.reshape(q_params_b, (-1, 1)), jnp.reshape(lj_params_b, (-1, 2))], axis=1)

        qlj_params_src, qlj_params_dst = self.interpolate_nonbonded_params(qlj_params_a, qlj_params_b)
        qlj_params = jnp.concatenate([qlj_params_src, qlj_params_dst])

        exclusion_idxs_a, scale_factors_a = nonbonded.generate_exclusion_idxs(
            self.mol_a, scale12=_SCALE_12, scale13=_SCALE_13, scale14=_SCALE_14
        )

        exclusion_idxs_b, scale_factors_b = nonbonded.generate_exclusion_idxs(
            self.mol_b, scale12=_SCALE_12, scale13=_SCALE_13, scale14=_SCALE_14
        )

        # (ytz): use the same scale factors of LJ & charges for now
        # this isn't quite correct as the LJ/Coluomb may be different in
        # different forcefields.
        scale_factors_a = np.stack([scale_factors_a, scale_factors_a], axis=1)
        scale_factors_b = np.stack([scale_factors_b, scale_factors_b], axis=1)

        combined_exclusion_dict = dict()

        for ij, scale in zip(exclusion_idxs_a, scale_factors_a):
            ij = tuple(sorted(self.a_to_c[ij]))
            if ij in combined_exclusion_dict:
                np.testing.assert_array_equal(combined_exclusion_dict[ij], scale)
            else:
                combined_exclusion_dict[ij] = scale

        for ij, scale in zip(exclusion_idxs_b, scale_factors_b):
            ij = tuple(sorted(self.b_to_c[ij]))
            if ij in combined_exclusion_dict:
                np.testing.assert_array_equal(combined_exclusion_dict[ij], scale)
            else:
                combined_exclusion_dict[ij] = scale

        combined_exclusion_idxs = []
        combined_scale_factors = []

        for e, s in combined_exclusion_dict.items():
            combined_exclusion_idxs.append(e)
            combined_scale_factors.append(s)

        combined_exclusion_idxs = np.array(combined_exclusion_idxs)
        combined_scale_factors = np.array(combined_scale_factors)

        # (ytz): we don't need exclusions between R_A and R_B will never see each other
        # under this decoupling scheme. They will always be at cutoff apart from each other.

        # plane_idxs: RA = Core = 0, RB = -1
        # offset_idxs: Core = 0, RA = RB = +1
        combined_lambda_plane_idxs = np.zeros(self.get_num_atoms(), dtype=np.int32)
        combined_lambda_offset_idxs = np.zeros(self.get_num_atoms(), dtype=np.int32)

        # w = cutoff * (lambda_plane_idxs + lambda_offset_idxs * lamb)
        for atom, group in enumerate(self.c_flags):
            if group == 0:
                # core atom
                combined_lambda_plane_idxs[atom] = 0
                combined_lambda_offset_idxs[atom] = 0
            elif group == 1:  # RA or RA and RB (if minimize) interact at lamb=0.0
                combined_lambda_plane_idxs[atom] = 0
                combined_lambda_offset_idxs[atom] = 1
            elif group == 2:  # R Groups of Mol B
                combined_lambda_plane_idxs[atom] = -1
                combined_lambda_offset_idxs[atom] = 1
            else:
                assert 0, f"Unknown group {group}"

        beta = _BETA
        cutoff = _CUTOFF  # solve for this analytically later

        nb = potentials.NonbondedInterpolated(
            combined_exclusion_idxs,
            combined_scale_factors,
            combined_lambda_plane_idxs,
            combined_lambda_offset_idxs,
            beta,
            cutoff,
        )

        return qlj_params, nb

    @staticmethod
    def _concatenate(arrs):
        non_empty = []
        for arr in arrs:
            if len(arr) != 0:
                non_empty.append(jnp.array(arr))
        return jnp.concatenate(non_empty)

    def _parameterize_bonded_term(self, ff_params, bonded_handle, potential):
        # Bonded terms are defined as follows:
        # If a bonded term is comprised exclusively of atoms in the core region, then
        # its energy its interpolated from the on-state to off-state.
        # Otherwise (i.e. it has one atom that is not in the core region), the bond term
        # is defined as unique, and is on at all times.
        # This means that the end state will contain dummy atoms that is not the true end-state,
        # but contains an analytical correction (through Boresch) that can be cancelled out.

        params_a, idxs_a = bonded_handle.partial_parameterize(ff_params, self.mol_a)
        params_b, idxs_b = bonded_handle.partial_parameterize(ff_params, self.mol_b)

        core_params_a = []
        core_params_b = []
        unique_params_r = []

        core_idxs_a = []
        core_idxs_b = []
        unique_idxs_r = []
        for p, old_atoms in zip(params_a, idxs_a):
            new_atoms = self.a_to_c[old_atoms]
            if np.all(self.c_flags[new_atoms] == 0):
                core_params_a.append(p)
                core_idxs_a.append(new_atoms)
            else:
                unique_params_r.append(p)
                unique_idxs_r.append(new_atoms)

        for p, old_atoms in zip(params_b, idxs_b):
            new_atoms = self.b_to_c[old_atoms]
            if np.all(self.c_flags[new_atoms] == 0):
                core_params_b.append(p)
                core_idxs_b.append(new_atoms)
            else:
                unique_params_r.append(p)
                unique_idxs_r.append(new_atoms)

        core_params_a = jnp.array(core_params_a)
        core_params_b = jnp.array(core_params_b)
        unique_params_r = jnp.array(unique_params_r)

        # number of parameters per term (2 for bonds, 2 for angles, 3 for torsions)
        # P = params_a.shape[-1]  # TODO: note P unused

        combined_params = self._concatenate([core_params_a, core_params_b, unique_params_r])

        # number of atoms involved in the bonded term
        K = idxs_a.shape[-1]

        core_idxs_a = np.array(core_idxs_a, dtype=np.int32).reshape((-1, K))
        core_idxs_b = np.array(core_idxs_b, dtype=np.int32).reshape((-1, K))
        unique_idxs_r = np.array(unique_idxs_r, dtype=np.int32).reshape((-1, K))  # always on

        # TODO: assert `len(core_idxs_a) == len(core_idxs_b)` in a more fine-grained way

        combined_idxs = np.concatenate([core_idxs_a, core_idxs_b, unique_idxs_r])

        lamb_mult = np.array(
            [-1] * len(core_idxs_a) + [1] * len(core_idxs_b) + [0] * len(unique_idxs_r), dtype=np.int32
        )
        lamb_offset = np.array(
            [1] * len(core_idxs_a) + [0] * len(core_idxs_b) + [1] * len(unique_idxs_r), dtype=np.int32
        )

        u_fn = potential(combined_idxs, lamb_mult, lamb_offset)
        return combined_params, u_fn

    def parameterize_harmonic_bond(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.hb_handle, potentials.HarmonicBond)

    def parameterize_harmonic_angle(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.ha_handle, potentials.HarmonicAngle)

    def parameterize_proper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.pt_handle, potentials.PeriodicTorsion)

    def parameterize_improper_torsion(self, ff_params):
        return self._parameterize_bonded_term(ff_params, self.ff.it_handle, potentials.PeriodicTorsion)

    def parameterize_periodic_torsion(self, proper_params, improper_params):
        """
        Parameterize all periodic torsions in the system.
        """
        proper_params, proper_potential = self.parameterize_proper_torsion(proper_params)
        improper_params, improper_potential = self.parameterize_improper_torsion(improper_params)
        combined_params = jnp.concatenate([proper_params, improper_params])
        combined_idxs = np.concatenate([proper_potential.get_idxs(), improper_potential.get_idxs()])
        combined_lambda_mult = np.concatenate(
            [proper_potential.get_lambda_mult(), improper_potential.get_lambda_mult()]
        )
        combined_lambda_offset = np.concatenate(
            [proper_potential.get_lambda_offset(), improper_potential.get_lambda_offset()]
        )
        combined_potential = potentials.PeriodicTorsion(combined_idxs, combined_lambda_mult, combined_lambda_offset)
        return combined_params, combined_potential
