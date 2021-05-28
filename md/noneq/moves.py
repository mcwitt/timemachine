import numpy as np

from md.states import CoordsVelBox
from md.ensembles import NVTEnsemble

from md.moves import MonteCarloMove
from md.noneq.utils import alchemify

from typing import Tuple

from timemachine.lib.custom_ops import Context, LangevinIntegrator

Array = np.array


class DeleteAndInsert(MonteCarloMove):
    def __init__(self,
                 deletion_protocol: Array, insertion_protocol: Array, alchemical_indices: Array,
                 integrator_impl: LangevinIntegrator, ensemble: NVTEnsemble):
        """Construct a move that alchemically deletes and reinserts the alchemical particles.

        Notes
        -----
        * Will need to compute the acceptance criterion differently if this move were to contain a barotat --
            currently asserts the move doesn't change the box dimensions
        * May want to make num_insertion_steps and num_deletion_steps adjustable on the fly?
        * TODO: validate that deletion and insertion order is correct? I imagine that will be an easy
            mistake to make
        """
        self.deletion_protocol = deletion_protocol
        self.insertion_protocol = insertion_protocol
        assert deletion_protocol[0] == insertion_protocol[-1]  # should start and end at same value of lambda, at least
        self.alchemical_indices = alchemical_indices
        self.integrator_impl = integrator_impl
        alchemical_model = alchemify(ensemble.potential_energy, alchemical_indices)
        self.ensemble = NVTEnsemble(alchemical_model, ensemble.temperature)

    def propose(self, x: CoordsVelBox) -> Tuple[CoordsVelBox, float]:
        """Simulate a deletion trajectory followed immediately by an insertion trajectory, accumulating work.
        Proposal: final snapshot of proposal trajectory
        Acceptance probability: min(1, exp(-reduced_work))

        Notes
        -----
        * For short protocols, computing the nonequilibrium work via trapz(du_dl) will become
            increasingly biased, compared with other ways to compute the work
        """

        ctxt = Context(x.coords, x.velocities, x.box, self.integrator_impl, self.ensemble.potential_energy.all_impls)

        # delete particles
        du_dl_traj_deletion, _ = ctxt.multiple_steps(self.deletion_protocol, 1, 0)

        # TODO: can add a random symmetric move here, such as a translation or re-orientation

        # insert insert particles
        du_dl_traj_insertion, _ = ctxt.multiple_steps(self.insertion_protocol, 1, 0)

        assert np.allclose(ctxt.get_box(), x.box)  # shouldn't have updated the box dimensions
        proposal = CoordsVelBox(ctxt.get_x_t(), ctxt.get_v_t(), ctxt.get_box())

        # work, in kJ/mol
        W_delete = np.trapz(du_dl_traj_deletion, self.deletion_protocol)
        W_insert = np.trapz(du_dl_traj_insertion, self.insertion_protocol)

        # reduced work, now unitless
        reduced_work = self.ensemble.reduce(W_delete + W_insert)
        log_acceptance_probability = min(0.0, - reduced_work)

        return proposal, log_acceptance_probability
