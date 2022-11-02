from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import rdFMCS
from scipy.spatial.distance import cdist

from timemachine.constants import DEFAULT_FF
from timemachine.fe.topology import AtomMappingError
from timemachine.fe.utils import set_romol_conf
from timemachine.ff import Forcefield
from timemachine.md.align import align_mols_by_core


def mcs(
    a,
    b,
    threshold: float = 2.0,
    timeout: int = 5,
    smarts: Optional[str] = None,
    conformer_aware: bool = True,
    retry: bool = True,
    match_hydrogens: bool = True,
):
    """Find maximum common substructure between mols a and b
    using reasonable settings for single topology:
    * disallow partial ring matches
    * disregard element identity and valence

    if conformer_aware=True, only match atoms within distance threshold
        (assumes conformers are aligned)

    if retry=True, then reseed with result of easier MCS(RemoveHs(a), RemoveHs(b)) in case of failure

    if match_hydrogens=False, then do not match using hydrogens. Will not retry
    """
    params = rdFMCS.MCSParameters()

    # bonds
    params.BondCompareParameters.CompleteRingsOnly = 1
    params.BondCompareParameters.RingMatchesRingOnly = 1
    params.BondTyper = rdFMCS.BondCompare.CompareAny

    # atoms
    params.AtomCompareParameters.CompleteRingsOnly = 1
    params.AtomCompareParameters.RingMatchesRingOnly = 1
    params.AtomCompareParameters.MatchValences = 0
    params.AtomCompareParameters.MatchChiralTag = 0
    if conformer_aware:
        params.AtomCompareParameters.MaxDistance = threshold
    params.AtomTyper = rdFMCS.AtomCompare.CompareAny
    # globals
    params.Timeout = timeout
    if smarts is not None:
        if match_hydrogens:
            params.InitialSeed = smarts
        else:
            # need to remove Hs from the input smarts
            params.InitialSeed = Chem.MolToSmarts(Chem.RemoveHs(Chem.MolFromSmarts(smarts)))

    def strip_hydrogens(mol):
        """Strip hydrogens with deepcopy to be extra safe"""
        return Chem.RemoveHs(deepcopy(mol))

    if not match_hydrogens:
        # Setting CompareAnyHeavyAtom doesn't handle this correctly, strip hydrogens explicitly
        a = strip_hydrogens(a)
        b = strip_hydrogens(b)
        # Disable retrying, as it will compare original a and b
        retry = False

    # try on given mols
    result = rdFMCS.FindMCS([a, b], params)

    # optional fallback
    def is_trivial(mcs_result) -> bool:
        return mcs_result.numBonds < 2

    if retry and is_trivial(result) and smarts is None:
        # try again, but seed with MCS computed without explicit hydrogens
        a_without_hs = strip_hydrogens(a)
        b_without_hs = strip_hydrogens(b)

        heavy_atom_result = rdFMCS.FindMCS([a_without_hs, b_without_hs], params)
        params.InitialSeed = heavy_atom_result.smartsString

        result = rdFMCS.FindMCS([a, b], params)

    if is_trivial(result):
        message = f"""MCS result trivial!
            timed out: {result.canceled}
            # atoms in MCS: {result.numAtoms}
            # bonds in MCS: {result.numBonds}
        """
        raise AtomMappingError(message)

    return result


def _get_core_conf_oblivious(mol_a, mol_b, query_mol):
    core = np.array(
        [
            np.array(mol_a.GetSubstructMatch(query_mol)),
            np.array(mol_b.GetSubstructMatch(query_mol)),
        ]
    ).T
    return core


def get_core_by_mcs(
    mol_a,
    mol_b,
    query,
    threshold=0.5,
    conformer_aware: bool = True,
):
    """Return np integer array that can be passed to RelativeFreeEnergy constructor

    Parameters
    ----------
    mol_a, mol_b, query : RDKit molecules
    threshold : float, in angstroms
    conformer_aware: bool
        if True, only match atoms within distance threshold
        (assumes conformers are aligned)

    Returns
    -------
    core : np.ndarray of ints, shape (n_MCS, 2)

    Notes
    -----
    * Warning! Some atoms that intuitively should be mapped together are not,
        when threshold=0.5 Å in custom atom comparison, because conformers aren't
        quite aligned enough.
    * Warning! Because of the intermediate representation of a substructure query,
        the core indices can get flipped around,
        for example if the substructure match hits only part of an aromatic ring.

        In some cases, this can fail to find a mapping that satisfies the distance
        threshold, raising an AtomMappingError.
    """
    if not conformer_aware:
        return _get_core_conf_oblivious(mol_a, mol_b, query)

    # fetch conformer, assumed aligned
    conf_a = mol_a.GetConformer(0).GetPositions()
    conf_b = mol_b.GetConformer(0).GetPositions()

    # note that >1 match possible here -- must pick minimum-cost match
    # TODO: possibly break this into two stages
    #  following https://github.com/proteneer/timemachine/pull/819#discussion_r966130215
    max_matches = 10_000
    matches_a = mol_a.GetSubstructMatches(query, uniquify=False, maxMatches=max_matches)
    matches_b = mol_b.GetSubstructMatches(query, uniquify=False, maxMatches=max_matches)

    # warn if this search won't be exhaustive
    if len(matches_a) == max_matches or len(matches_b) == max_matches:
        print("Warning: max_matches exceeded -- cannot guarantee to find a feasible core")

    # once rather than in subsequent double for-loop
    all_distances = cdist(conf_a, conf_b)
    gt_threshold = all_distances > threshold

    matches_a = [np.array(a) for a in matches_a]
    matches_b = [np.array(b) for b in matches_b]

    cost = np.zeros((len(matches_a), len(matches_b)))

    for i, a in enumerate(matches_a):
        for j, b in enumerate(matches_b):
            if np.any(gt_threshold[a, b]):
                cost[i, j] = +np.inf
            else:
                dij = all_distances[a, b]
                cost[i, j] = np.sum(dij)

    # find (i,j) = argmin cost
    min_i, min_j = np.unravel_index(np.argmin(cost, axis=None), cost.shape)

    # concatenate into (n_atoms, 2) array
    inds_a, inds_b = matches_a[min_i], matches_b[min_j]
    core = np.array([inds_a, inds_b]).T

    if np.isinf(cost[min_i, min_j]):
        raise AtomMappingError(f"not all mapped atoms are within {threshold:.3f}Å of each other")

    return core


def get_core_with_alignment(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    threshold: float = 2.0,
    n_steps: int = 200,
    k: float = 10000,
    ff: Optional[Forcefield] = None,
    initial_smarts: Optional[str] = None,
) -> Tuple[NDArray, str]:
    """Selects a core between two molecules, by finding an initial core then aligning based on the core.

    Parameters
    ----------
    mol_a: RDKit Mol

    mol_b: RDKit Mol

    threshold: float
        Threshold between atoms in angstroms

    n_steps: float
        number of steps to run for alignment

    ff: Forcefield or None
        Forcefield to use for alignment, defaults to DEFAULT_FF forcefield if None

    initial_smarts: str or None
        If set uses smarts as the initial seed to MCS and as a fallback
        if mcs results in a trivial core.

    Returns
    -------
    core : np.ndarray of ints, shape (n_MCS, 2)
    smarts: SMARTS string used to find core

    Notes
    -----
    * Warning! The initial core can contain an incorrect mapping, in that case the
        core returned will be the same as running mcs followed by get_core_by_mcs.
    """
    # Copy mols so that when we change coordinates doesn't corrupt inputs
    a_copy = deepcopy(mol_a)
    b_copy = deepcopy(mol_b)

    if ff is None:
        ff = Forcefield.load_from_file(DEFAULT_FF)

    def setup_core(mol_a, mol_b, match_hydrogens, initial_smarts):
        result = mcs(mol_a, mol_b, threshold=threshold, match_hydrogens=match_hydrogens, smarts=initial_smarts)
        query_mol = Chem.MolFromSmarts(result.smartsString)
        core = get_core_by_mcs(mol_a, mol_b, query_mol, threshold=threshold)
        return core, result.smartsString

    try:
        heavy_atom_core, _ = setup_core(a_copy, b_copy, False, initial_smarts)

        conf_a, conf_b = align_mols_by_core(mol_a, mol_b, heavy_atom_core, ff, n_steps=n_steps, k=k)
        set_romol_conf(a_copy, conf_a)
        set_romol_conf(b_copy, conf_b)

        core, smarts = setup_core(a_copy, b_copy, True, initial_smarts)
        return core, smarts
    except AtomMappingError as err:
        # Fall back to user provided smarts
        if initial_smarts is not None:
            print(f"WARNING: Could not get atom mapping: {err}, falling back to user defined smarts: {initial_smarts}")
            query_mol = Chem.MolFromSmarts(initial_smarts)
            core = get_core_by_mcs(mol_a, mol_b, query_mol, threshold=threshold)
            return core, initial_smarts

        raise err
