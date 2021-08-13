# TODO: move this frequently repeated code fragment for fetching default
#  forcefield into ff module or another place?

from ff import Forcefield
from ff.handlers.deserialize import deserialize_handlers
with open('ff/params/smirnoff_1_1_0_ccc.py') as f:
    ff_handlers = deserialize_handlers(f.read())
default_forcefield = Forcefield(ff_handlers)