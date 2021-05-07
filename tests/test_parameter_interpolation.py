from jax.config import config; config.update("jax_enable_x64", True)
import copy
import functools
import numpy as np
import jax.numpy as jnp

from common import GradientTest
from timemachine.lib import potentials

from common import prepare_water_system

def interpolated_potential(conf, params, box, lamb, u_fn):
    assert params.size % 2 == 0

    CP = params.shape[0]//2
    new_params = (1-lamb)*params[:CP] + lamb*params[CP:]

    return u_fn(conf, new_params, box, lamb)


class TestInterpolatedPotential(GradientTest):

    def test_nonbonded(self):

        # we test nonbonded terms to ensure that we're doing the chain rule with du_dl correctly since
        # bonded terms do not depend on lambda.

        np.random.seed(4321)
        D = 3

        cutoff = 1.0
        size = 36

        water_coords = self.get_water_coords(D, sort=False)
        coords = water_coords[:size]
        padding = 0.2
        diag = np.amax(coords, axis=0) - np.amin(coords, axis=0) + padding
        box = np.eye(3)
        np.fill_diagonal(box, diag)

        N = coords.shape[0]

        lambda_plane_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)
        lambda_offset_idxs = np.random.randint(low=0, high=2, size=N, dtype=np.int32)

        for lamb in [0.0, 0.2, 1.0]:
            for precision, rtol in [(np.float64, 1e-8), (np.float32, 1e-4)]:

                    # E = 0 # DEBUG!
                    qlj_src, ref_potential, test_potential = prepare_water_system(
                        coords,
                        lambda_plane_idxs,
                        lambda_offset_idxs,
                        p_scale=1.0,
                        cutoff=cutoff
                    )

                    qlj_dst, _, _ = prepare_water_system(
                        coords,
                        lambda_plane_idxs,
                        lambda_offset_idxs,
                        p_scale=1.0,
                        cutoff=cutoff
                    )

                    qlj = np.concatenate([qlj_src, qlj_dst])

                    print("lambda", lamb, "cutoff", cutoff, "precision", precision, "xshape", coords.shape)

                    ref_interpolated_potential = functools.partial(
                        interpolated_potential,
                        u_fn=ref_potential)

                    test_interpolated_potential = potentials.NonbondedInterpolated(
                        *test_potential.args
                    )

                    self.compare_forces(
                        coords,
                        qlj,
                        box,
                        lamb,
                        ref_interpolated_potential,
                        test_interpolated_potential,
                        rtol,
                        precision=precision
                    )


    def test_nonbonded_advanced(self):

        # This test checks that we can supply arbitrary transformations of lambda to
        # the nonbonded potential, and that the resulting derivatives (both du/dp and du/dl)
        # are correct.

        np.random.seed(4321)
        D = 3

        cutoff = 1.0
        size = 36

        water_coords = self.get_water_coords(D, sort=False)
        coords = water_coords[:size]
        padding = 0.2
        diag = np.amax(coords, axis=0) - np.amin(coords, axis=0) + padding
        box = np.eye(3)
        np.fill_diagonal(box, diag)

        N = coords.shape[0]

        lambda_plane_idxs = np.random.randint(low=0, high=1, size=N, dtype=np.int32)
        lambda_offset_idxs = np.random.randint(low=1, high=2, size=N, dtype=np.int32)


        def transform_q(lamb):
            return lamb*lamb

        def transform_s(lamb):
            return jnp.sin(lamb*np.pi/2)

        def transform_e(lamb):
            return jnp.where(lamb < 0.5, jnp.sin(lamb*np.pi)*jnp.sin(lamb*np.pi), 1)

        offsets = np.arange(N)/N
        offsets = offsets*0.5

        def transform_w(lamb):
            w = jnp.sin((lamb - offsets)*np.pi)**2
            w = jnp.where(lamb < offsets, 0, w)
            w = jnp.where(lamb > 0.5+offsets, 1.0, w)
            return w

        # (ytz: leave this here in case the user wants to see what the
        # decoupling looks like)
        # all_ws = []
        # lambda_schedule = np.linspace(0, 1, 100)
        # for lamb in lambda_schedule:
        #     ws = transform_w(lamb)
        #     all_ws.append(ws)
        # all_ws = np.array(all_ws)
        # import matplotlib.pyplot as plt
        # for ws in all_ws.T:
        #     plt.plot(lambda_schedule, ws)
        # plt.show()

        # E = 0 # DEBUG!
        qlj_src, ref_potential, test_potential = prepare_water_system(
            coords,
            lambda_plane_idxs,
            lambda_offset_idxs,
            p_scale=1.0,
            cutoff=cutoff
        )

        qlj_dst, _, _ = prepare_water_system(
            coords,
            lambda_plane_idxs,
            lambda_offset_idxs,
            p_scale=1.0,
            cutoff=cutoff
        )

        def interpolate_params(lamb, qlj_src, qlj_dst):
            new_q = (1-transform_q(lamb))*qlj_src[:, 0] + transform_q(lamb)*qlj_dst[:, 0]
            new_s = (1-transform_s(lamb))*qlj_src[:, 1] + transform_s(lamb)*qlj_dst[:, 1]
            new_e = (1-transform_e(lamb))*qlj_src[:, 2] + transform_e(lamb)*qlj_dst[:, 2]
            return jnp.stack([new_q, new_s, new_e], axis=1)

        def u_reference(x, params, box, lamb):
            d4 = cutoff*(lambda_plane_idxs + lambda_offset_idxs*transform_w(lamb, ))
            d4 = jnp.expand_dims(d4, axis=-1)
            x = jnp.concatenate((x, d4), axis=1)

            qlj_src = params[:len(params)//2]
            qlj_dst = params[len(params)//2:]
            qlj = interpolate_params(lamb, qlj_src, qlj_dst)
            return ref_potential(x, qlj, box, lamb)

        for precision, rtol in [(np.float64, 1e-8), (np.float32, 1e-4)]:

            # for lamb in [0.0, 0.2, 0.6, 0.7, 0.8, 1.0]:
            for lamb in np.linspace(0, 1.0, 10):

                    # qlj = np.concatenate([qlj_src, qlj_dst])
                    qlj = np.concatenate([qlj_src, qlj_src])

                    print("lambda", lamb, "cutoff", cutoff, "precision", precision, "xshape", coords.shape)

                    args = copy.deepcopy(test_potential.args)
                    # args.append("return lambda*lambda") # transform q
                    # args.append("return sin(lambda*PI/2)") # transform sigma
                    # args.append("return lambda < 0.5 ? sin(lambda*PI)*sin(lambda*PI) : 1") # transform epsilon
                    args.append("return lambda;") # transform q
                    args.append("return lambda;") # transform sigma
                    args.append("return lambda;") # transform epsilon
                    args.append("""
                        double offset = atom_idx / (2.0*N);
                        if(lambda < offset) {
                            return 0.0;
                        } else if (lambda > (0.5 + offset)) {
                            return 1.0;
                        } else {
                            NumericType term = sin((lambda - offset)*PI);
                            return term*term;
                        }

                    """)

                    test_interpolated_potential = potentials.NonbondedInterpolated(
                        *args,
                    )

                    self.compare_forces(
                        coords,
                        qlj,
                        box,
                        lamb,
                        u_reference,
                        test_interpolated_potential,
                        rtol,
                        precision=precision
                    )