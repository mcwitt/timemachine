import numpy as np

from ff.handlers.nonbonded import oe_assign_charges, AM1CCCHandler
from ff.handlers.deserialize import deserialize_handlers

import timemachine
from pathlib import Path
from ff import Forcefield
from timemachine import constants


def get_default_am1ccc_handler():
    # load ff
    root = Path(timemachine.__file__).parent.parent
    path_to_ff = str(root.joinpath('ff/params/smirnoff_1_1_0_ccc.py'))

    with open(path_to_ff) as f:
        ff_handlers = deserialize_handlers(f.read())
    ff = Forcefield(ff_handlers)

    # get AM1CCC handler
    ordered_handles = ff.get_ordered_handles()
    components = [handle.__class__ for handle in ordered_handles]
    handles = dict(zip(components, ordered_handles))
    am1ccc_handler = handles[AM1CCCHandler]

    return am1ccc_handler


def assert_am1ccc_am1bcc_consistency(mols, abs_tolerance=1e-3):
    """Assert that the partial charges assigned by ff/params/smirnoff_1_1_0_ccc.py
    are close to those assigned by AM1BCCELF10, for all atoms in a collection of mols"""
    # methods to compare
    am1ccc_handler = get_default_am1ccc_handler()

    def am1ccc_parameterize(mol):
        return am1ccc_handler.parameterize(mol)

    def am1bcc_parameterize(mol):
        return oe_assign_charges(mol, 'AM1BCCELF10')

    # run both methods on all mols
    inlined_constant = np.sqrt(timemachine.constants.ONE_4PI_EPS0)

    for mol in mols:
        ref = am1bcc_parameterize(mol) / inlined_constant
        test = am1ccc_parameterize(mol) / inlined_constant
        difference = np.max(np.abs(ref - test))
        assert difference < abs_tolerance, f'{difference:.3f} e > {abs_tolerance} e'