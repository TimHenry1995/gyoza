import gyoza.modelling.losses as mls
import unittest
import tensorflow as tf

class TestSupervisedFactorLoss(unittest.TestCase):
    
    def test_init(self):
        """Tests whether SupervisedFactorLoss can be initialized."""

        # Initialize
        loss = mls.SupervisedFactorLoss(dimensions_per_factor=[1,2,3])

        # Target
        x_target = tf.constant([[1,0,0,0,0,0],[0,1,1,0,0,0],[0,0,0,1,1,1]], dtype=tf.keras.backend.floatx())

        # Observe
        x_observed = loss.__factor_masks__

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_standard_normal_minimal(self):
        """
        Tests whether the marginal of the standard normal results in minimal loss.
        """
        DIMENSIONS = [2, 2]
        D = sum(DIMENSIONS)
        K = len(DIMENSIONS)
        B = 512

        z_a = tf.random.normal([B, D])
        z_b = tf.random.normal([B, D])

        j_a = tf.zeros([B])
        j_b = tf.zeros([B])

        y_pred = tf.concat(
            [z_a, z_b, j_a[:, None], j_b[:, None]], axis=1
        )

        y_true = tf.zeros([B, K])

        loss_function = mls.SupervisedFactorLoss(dimensions_per_factor=[2,2])

        loss = loss_function(y_true, y_pred)

        # Shift z_a away from zero
        z_a_shifted = z_a + 2.0
        y_pred_shifted = tf.concat(
            [z_a_shifted, z_b, j_a[:, None], j_b[:, None]], axis=1
        )

        loss_shifted = loss_function(y_true, y_pred_shifted)

        self.assertGreater(a=loss_shifted, b=loss)

    def test_perfect_correlation(self):
        """Test whether perfect correlation reduces the loss."""
        
        DIMENSIONS = [2, 2]
        D = sum(DIMENSIONS)
        K = len(DIMENSIONS)
        B = 512
        
        z_a = tf.random.normal([B, D])
        z_b = z_a  # perfect correlation

        j_a = tf.zeros([B])
        j_b = tf.zeros([B])

        y_pred = tf.concat(
            [z_a, z_b, j_a[:, None], j_b[:, None]], axis=1
        )

        y_true = tf.ones([B, K]) * 0.99  # almost perfect

        loss_function = mls.SupervisedFactorLoss(dimensions_per_factor=[2,2])
        loss = loss_function(y_true, y_pred)

        # Break correlation
        z_b_shuffled = tf.random.shuffle(z_b)
        y_pred_bad = tf.concat(
            [z_a, z_b_shuffled, j_a[:, None], j_b[:, None]], axis=1
        )

        loss_bad = loss_function(y_true, y_pred_bad)

        self.assertGreater(a=loss_bad, b=loss)

    def test_zero_correlation_independence(self):
        """Tests whether a zero correlation matches independent Gaussians."""

        DIMENSIONS = [2, 2]
        D = sum(DIMENSIONS)
        K = len(DIMENSIONS)
        B = 512
        
        z_a = tf.random.normal([B, D])
        z_b = tf.random.normal([B, D])

        j = tf.zeros([B])

        y_pred = tf.concat(
            [z_a, z_b, j[:, None], j[:, None]], axis=1
        )

        y_true = tf.zeros([B, K])

        loss_function = mls.SupervisedFactorLoss(dimensions_per_factor=[2,2])
        loss = loss_function(y_true, y_pred)

        # If we force correlation that isn't present, loss increases
        y_true_wrong = tf.ones([B, K]) * 0.8
        loss_wrong = loss_function(y_true_wrong, y_pred)

        self.assertGreater(a=loss_wrong, b=loss)

    def test_jacobian_effect(self):
        """Tests whether the Jacobian term improves the likelihood."""

        DIMENSIONS = [2, 2]
        D = sum(DIMENSIONS)
        K = len(DIMENSIONS)
        B = 512
        
        z = tf.random.normal([B, D])
        j_small = tf.zeros([B])
        j_large = tf.ones([B]) * 5.0

        y_pred_small = tf.concat(
            [z, z, j_small[:, None], j_small[:, None]], axis=1
        )
        y_pred_large = tf.concat(
            [z, z, j_large[:, None], j_large[:, None]], axis=1
        )

        y_true = tf.ones([B, K]) * 0.9

        loss_function = mls.SupervisedFactorLoss(dimensions_per_factor=[2,2])
        loss_small = loss_function(y_true, y_pred_small)
        loss_large = loss_function(y_true, y_pred_large)

        self.assertLess(a=loss_large, b=loss_small)

    def test_variance_collapse(self):
        """Tests whether collapse of the variance is penalized."""

        DIMENSIONS = [2, 2]
        D = sum(DIMENSIONS)
        K = len(DIMENSIONS)
        B = 512
        
        z_a = tf.random.normal([B, D])
        z_b = tf.random.normal([B, D])  # NOT equal

        j = tf.zeros([B])

        y_pred = tf.concat(
            [z_a, z_b, j[:, None], j[:, None]], axis=1
        )

        y_true = tf.ones([B, K]) * 0.999

        loss_function = mls.SupervisedFactorLoss(dimensions_per_factor=DIMENSIONS)
        loss = loss_function(y_true, y_pred)

        self.assertTrue(tf.math.is_finite(loss))
        self.assertGreater(a=loss.numpy(), b= 100.0)  # should be large



    def test_compute(self):
        """Tests whether SupervisedFactorLoss can compute the same loss value as before."""

        # Initialize
        loss = mls.SupervisedFactorLoss(dimensions_per_factor=[2,1,3])
        z_tilde_a = tf.constant([[3,6,8,4,6,2], [4,8,7,6,5,9], [6,8,5,6,3,2]], dtype=tf.keras.backend.floatx())
        z_tilde_b = tf.constant([[3,6,5,7,8,4], [9,7,8,5,7,4], [9,7,2,5,3,6]], dtype=tf.keras.backend.floatx())
        j_a = tf.constant([[4],[6],[7]], dtype=tf.keras.backend.floatx())
        j_b = tf.constant([[8],[4],[2]], dtype=tf.keras.backend.floatx())
        y_true = tf.constant([[0,1,0],[0,0,1],[1,0,1]], dtype=tf.keras.backend.floatx())

        # Target
        x_target = tf.constant(5428047.5, dtype=tf.keras.backend.floatx())

        # Observe
        y_pred = tf.keras.layers.Concatenate(axis=-1)([z_tilde_a, z_tilde_b, j_a, j_b])
        x_observed = loss(y_true=tf.constant(y_true), y_pred=y_pred)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

if __name__ == "__main__":
    #unittest.main()
    TestSupervisedFactorLoss.test_compute(None)