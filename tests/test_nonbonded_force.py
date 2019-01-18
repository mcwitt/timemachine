import unittest
import numpy as np
import tensorflow as tf
from timemachine.functionals.nonbonded import Electrostatic, LeonnardJones
from timemachine.constants import ONE_4PI_EPS0
from timemachine import derivatives

class TestNonbondedForce(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_lj612(self):
        """
        Testing non-periodic lj 612 forces.
        """
        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        x_ph = tf.placeholder(shape=(5, 3), dtype=np.float64)

        params_np = np.array([3.0, 2.0, 1.0, 1.4], dtype=np.float64)
        params = tf.convert_to_tensor(params_np)
        # Ai, Ci
        param_idxs = np.array([
            [0, 3],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2]], dtype=np.int32)

        exclusions = np.array([
            [0,0,1,0,0],
            [0,0,0,1,1],
            [1,0,0,0,0],
            [0,1,0,0,1],
            [0,1,0,1,0],
            ], dtype=np.bool)

        lj = LeonnardJones(params, param_idxs, exclusions)

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        num_atoms = 5
        ref_nrg = 0

        conf = x0

        for i in range(num_atoms):
            sig_i = params_np[param_idxs[i, 0]]
            eps_i = params_np[param_idxs[i, 1]]
            ri = conf[i]

            for j in range(i+1, num_atoms):
                if exclusions[i, j]:
                    continue

                sig_j = params_np[param_idxs[j, 0]]
                eps_j = params_np[param_idxs[j, 1]]
                rj = conf[j]

                r = np.linalg.norm(x0[i] - x0[j])

                sig = sig_i + sig_j
                sig2 = sig/r
                sig2 *= sig2
                sig6 = sig2*sig2*sig2
                eps = eps_i * eps_j

                vdwEnergy = eps*(sig6-1.0)*sig6

                ref_nrg += vdwEnergy

        # (ytz) TODO: add a test for forces, for now we rely on
        # autograd to be analytically correct.
        test_nrg_op = lj.energy(x_ph)
        test_nrg = sess.run(test_nrg_op, feed_dict={x_ph: x0})

        # 3707370.0568588967, 3707370.056858897
        np.testing.assert_almost_equal(ref_nrg, test_nrg, decimal=8)

        test_grads_op = tf.gradients(test_nrg_op, x_ph)[0]
        test_grads = sess.run(test_grads_op, feed_dict={x_ph: x0})

        net_force = np.sum(test_grads, axis=0)
        # this also checks that there are no NaNs
        np.testing.assert_almost_equal(net_force, [0,0,0], decimal=7)

        assert not np.any(np.isnan(sess.run(tf.hessians(test_nrg_op, x_ph), feed_dict={x_ph: x0})))
        mixed_partials = sess.run(derivatives.list_jacobian(test_grads_op, [lj.get_params()]), feed_dict={x_ph: x0})
        assert not np.any(np.isnan(mixed_partials))


    def test_electrostatics(self):
        """
        Testing non-periodic electrostatic forces.
        """
        x0 = np.array([
            [ 0.0637,   0.0126,   0.2203],
            [ 1.0573,  -0.2011,   1.2864],
            [ 2.3928,   1.2209,  -0.2230],
            [-0.6891,   1.6983,   0.0780],
            [-0.6312,  -1.6261,  -0.2601]
        ], dtype=np.float64)

        x_ph = tf.placeholder(shape=(5, 3), dtype=np.float64)

        params_np = np.array([1.3, 0.3], dtype=np.float64)
        params = tf.convert_to_tensor(params_np)
        param_idxs = np.array([0, 1, 1, 1, 1], dtype=np.int32)
        exclusions = np.array([
            [0,0,1,0,0],
            [0,0,0,1,1],
            [1,0,0,0,0],
            [0,1,0,0,1],
            [0,1,0,1,0],
            ], dtype=np.bool)

        ef = Electrostatic(params, param_idxs, exclusions)

        sess = tf.Session()
        sess.run(tf.initializers.global_variables())

        num_atoms = 5
        ref_nrg = 0

        for i in range(num_atoms):
            qi = params_np[param_idxs[i]]
            for j in range(i+1, num_atoms):
                if not exclusions[i, j]:
                    qj = params_np[param_idxs[j]]
                    qij = qi * qj
                    dij = np.linalg.norm(x0[i] - x0[j])
                    ref_nrg += qij/dij

        ref_nrg *= ONE_4PI_EPS0

        # (ytz) TODO: add a test for forces, for now we rely on
        # autograd to be analytically correct.
        test_nrg_op = ef.energy(x_ph)
        test_nrg = sess.run(test_nrg_op, feed_dict={x_ph: x0})
        np.testing.assert_almost_equal(ref_nrg, test_nrg, decimal=13)

        test_grads_op = tf.gradients(test_nrg_op, x_ph)[0]
        test_grads = sess.run(test_grads_op, feed_dict={x_ph: x0})

        net_force = np.sum(test_grads, axis=0)
        # this also checks that there are no NaNs
        np.testing.assert_almost_equal(net_force, [0,0,0], decimal=14)

        assert not np.any(np.isnan(sess.run(tf.hessians(test_nrg_op, x_ph), feed_dict={x_ph: x0})))
        mixed_partials = sess.run(derivatives.list_jacobian(test_grads_op, [ef.get_params()]), feed_dict={x_ph: x0})
        assert not np.any(np.isnan(mixed_partials))



if __name__ == "__main__":
    unittest.main()