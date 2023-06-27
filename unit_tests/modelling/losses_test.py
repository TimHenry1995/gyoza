import gyoza.modelling.losses as mls
import unittest
import tensorflow as tf

class TestSupervisedFactorLoss(unittest.TestCase):
    
    def test_init(self):
        """Tests whether SupervisedFactorLoss can be initialized."""

        # Initialize
        loss = mls.SupervisedFactorLoss(factor_channel_counts=[1,2,3])

        # Target
        x_target = tf.constant([[1,0,0,0,0,0],[0,1,1,0,0,0],[0,0,0,1,1,1]], dtype=tf.float32)

        # Observe
        x_observed = loss.__factor_masks__

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_call(self):
        """Tests whether SupervisedFactorLoss can be called."""

        # Initialize
        loss = mls.SupervisedFactorLoss(factor_channel_counts=[1,2,3])
        z_tilde_a = tf.constant([[3,6,8,4,6,2], [4,8,7,6,5,9], [6,8,5,6,3,2]], dtype=tf.float32)
        z_tilde_b = tf.constant([[3,6,5,7,8,4], [9,7,8,5,7,4], [9,7,2,5,3,6]], dtype=tf.float32)
        j_a = tf.constant([4,6,7], dtype=tf.float32)
        j_b = tf.constant([8,4,2], dtype=tf.float32)
        factor_indices = [0,2,1]

        # Target
        x_target = tf.constant(305.22574, dtype=tf.float32)

        # Observe
        x_observed = loss(y_true=tf.constant(factor_indices), y_pred=(z_tilde_a, z_tilde_b, j_a, j_b))

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

if __name__ == "__main__":
    #unittest.main()
    TestSupervisedFactorLoss().test_call()