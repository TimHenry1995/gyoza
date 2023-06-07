import gyoza.utilities.tensors as utt
import unittest
import tensorflow as tf

class Test_Move_Axis(unittest.TestCase):

    def test_same_axis(self):
        """Test whether the move_axis function manages to move the axis to its current position.
        This is an identity operation."""

        # Initialize
        x = tf.reshape(tf.range(0,24), [2,3,4])

        # Target
        x_target = x

        # Observe
        x_observed = utt.move_axis(x=x, from_index=1, to_index=1)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

    def test_axes_1_3(self):
        """Test whether the move_axis function manages to move the axis from index 1 to 3."""

        # Initialize
        x = tf.reshape(tf.range(0,24), [1,2,3,4])

        # Target
        x_target = tf.transpose(x, [0,2,3,1])

        # Observe
        x_observed = utt.move_axis(x=x, from_index=1, to_index=3)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

    def test_axes_2_0(self):
        """Test whether the move_axis function manages to move the axis from index 2 to 0."""

        # Initialize
        x = tf.reshape(tf.range(0,24), [1,2,3,4])

        # Target
        x_target = tf.transpose(x, [2,0,1,3])

        # Observe
        x_observed = utt.move_axis(x=x, from_index=2, to_index=0)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

class Test_Expand_Axes(unittest.TestCase):

    def test_axes_0(self):
        """Test whether the expand_axes function manages to expand along one axis"""

        # Initialize
        x = tf.reshape(tf.range(0,24), [2,3,4])

        # Target
        x_target = tf.reshape(x, [1,2,3,4])

        # Observe
        x_observed = utt.expand_axes(x=x, axes=[0])

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

    def test_axes_0_4(self):
        """Test whether the expand_axes function manages to expand along axes 0 and 4"""

        # Initialize
        x = tf.reshape(tf.range(0,24), [2,3,4])

        # Target
        x_target = tf.reshape(x, [1,2,3,4,1])

        # Observe
        x_observed = utt.expand_axes(x=x, axes=[0,4])

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

    def test_axes_4_0(self):
        """Test whether the expand_axes function manages to expand along axes 4 and 0"""

        # Initialize
        x = tf.reshape(tf.range(0,24), [2,3,4])

        # Target
        x_target = tf.reshape(x, [1,2,3,4,1])

        # Observe
        x_observed = utt.expand_axes(x=x, axes=[4,0])

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

    def test_axes_3_0(self):
        """Test whether the expand_axes function manages to expand along axes 3 and 0"""

        # Initialize
        x = tf.reshape(tf.range(0,24), [2,3,4])

        # Target
        x_target = tf.reshape(x, [1,2,3,1,4])

        # Observe
        x_observed = utt.expand_axes(x=x, axes=[3,0])

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

    def test_axes_3_2_0(self):
        """Test whether the expand_axes function manages to expand along axes 3, 2 and 0"""

        # Initialize
        x = tf.reshape(tf.range(0,24), [2,3,4])

        # Target
        x_target = tf.reshape(x, [1,2,1,1,3,4])

        # Observe
        x_observed = utt.expand_axes(x=x, axes=[3,2,0])

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))


if __name__ == "__main__":
    unittest.main()