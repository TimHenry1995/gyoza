import gyoza.modelling.semi_selectors as mss
import unittest
import tensorflow as tf

class TestHeaviside(unittest.TestCase):
    
    def test_mask_one_dimensional_even_length(self):
        """Tests whether the mask function of HeaviSide works on a 1 dimensional input of even length."""

        # Initialize
        x = tf.range(10, dtype=tf.float32)

        # Target
        x_target = tf.concat([tf.zeros([5]), tf.range(5,10, dtype=tf.float32)], axis=0)

        # Observe
        instance = mss.HeaviSide(axes=[0], shape=[10])
        x_observed = instance.mask(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum(x_target-x_observed).numpy(), second=0)

    def test_mask_one_dimensional_odd_length(self):
        """Tests whether the mask function of HeaviSide works on a 1 dimensional input of odd length."""

        # Initialize
        x = tf.range(11, dtype=tf.float32)

        # Target
        x_target = tf.concat([tf.zeros([5]), tf.range(5,11, dtype=tf.float32)], axis=0)

        # Observe
        instance = mss.HeaviSide(axes=[0], shape=[11])
        x_observed = instance.mask(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum(x_target-x_observed).numpy(), second=0)

    def test_mask_two_dimensional_axis_1(self):
        """Tests whether the mask function of HeaviSide works on a two dimensional input along axis 1."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.float32), shape=[3,5])

        # Target
        x_target = tf.constant([[0,0,2,3,4],[0,0,7,8,9],[0,0,12,13,14]], dtype=tf.float32)

        # Observe
        instance = mss.HeaviSide(axes=[1], shape=[5])
        x_observed = instance.mask(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum(x_target-x_observed).numpy(), second=0)

    def test_mask_two_dimensional_axis_0(self):
        """Tests whether the mask function of HeaviSide works on a two dimensional input along axis 0."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.float32), shape=[3,5])

        # Target
        x_target = tf.constant([[0,0,0,0,0],[5,6,7,8,9],[10,11,12,13,14]], dtype=tf.float32)

        # Observe
        instance = mss.HeaviSide(axes=[0], shape=[3])
        x_observed = instance.mask(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum(x_target-x_observed).numpy(), second=0)

    def test_mask_two_dimensional_axis_1_negative(self):
        """Tests whether the mask function of HeaviSide works on a two dimensional input along axis 1
        with a negative mask."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.float32), shape=[3,5])

        # Target
        x_target = tf.constant([[0,1,0,0,0],[5,6,0,0,0],[10,11,0,0,0]], dtype=tf.float32)

        # Observe
        instance = mss.HeaviSide(axes=[1], shape=[5], is_positive=False)
        x_observed = instance.mask(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum(x_target-x_observed).numpy(), second=0)




if __name__ == "__main__":
    unittest.main()