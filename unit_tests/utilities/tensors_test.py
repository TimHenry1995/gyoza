import gyoza.utilities.tensors as utt
import unittest
import tensorflow as tf

class TestMoveAxis(unittest.TestCase):

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

    def test_axes_1_minus_1(self):
        """Test whether the move_axis function manages to move the axis from index 2 to 0."""

        # Initialize
        x = tf.reshape(tf.range(0,24), [1,2,3,4])

        # Target
        x_target = tf.transpose(x, [0,2,3,1])

        # Observe
        x_observed = utt.move_axis(x=x, from_index=1, to_index=-1)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

class TestExpandAxes(unittest.TestCase):

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

class TestFlattenAlongAxes(unittest.TestCase):

    def test_3D_input_single_axis_empty(self):
        """Test whether the flatten_along_axes function manages to flatten a 3D input along an empty axis list.
        This is an identity operation."""

        # Initialize
        x = tf.reshape(tf.range(0,24), [2,3,4])

        # Target
        x_target = x

        # Observe
        x_observed = utt.flatten_along_axes(x=x, axes=[])

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))


    def test_3D_input_single_axis_0(self):
        """Test whether the flatten_along_axes function manages to flatten a 3D input along a single axis.
        This is an identity operation."""

        # Initialize
        x = tf.reshape(tf.range(0,24), [2,3,4])

        # Target
        x_target = x

        # Observe
        x_observed = utt.flatten_along_axes(x=x, axes=[0])

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

    def test_3D_input_single_axis_1(self):
        """Test whether the flatten_along_axes function manages to flatten a 3D input along a single axis.
        This is an identity operation."""

        # Initialize
        x = tf.reshape(tf.range(0,24), [2,3,4])

        # Target
        x_target = x

        # Observe
        x_observed = utt.flatten_along_axes(x=x, axes=[1])

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

    def test_3D_input_single_axis_2(self):
        """Test whether the flatten_along_axes function manages to flatten a 3D input along a single axis.
        This is an identity operation."""

        # Initialize
        x = tf.reshape(tf.range(0,24), [2,3,4])

        # Target
        x_target = x

        # Observe
        x_observed = utt.flatten_along_axes(x=x, axes=[2])

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

    def test_3D_input_axes_0_1(self):
        """Test whether the flatten_along_axes function manages to flatten a 3D input along axes 0 and 1."""

        # Initialize
        x = tf.reshape(tf.range(0,24), [2,3,4])

        # Target
        x_target = tf.reshape(x, [6,4])

        # Observe
        x_observed = utt.flatten_along_axes(x=x, axes=[0,1])

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

    def test_3D_input_axes_0_2(self):
        """Test whether the flatten_along_axes function manages to flatten a 3D input along axes 0 and 2."""

        # Initialize
        x = tf.reshape(tf.range(0,24), [2,3,4])

        # Target
        x_target = tf.transpose(x, perm=[0,2,1]) # Now has shape == [2,4,3]
        x_target = tf.reshape(x_target, [8,3])

        # Observe
        x_observed = utt.flatten_along_axes(x=x, axes=[0,2])

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

    def test_3D_input_axes_1_2(self):
        """Test whether the flatten_along_axes function manages to flatten a 3D input along axes 1 and 2."""

        # Initialize
        x = tf.reshape(tf.range(0,24), [2,3,4])

        # Target
        x_target = tf.reshape(x, [2,12])

        # Observe
        x_observed = utt.flatten_along_axes(x=x, axes=[1,2])

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

    def test_4D_input_axes_1_2(self):
        """Test whether the flatten_along_axes function manages to flatten a 4D input along axes 1 and 2."""

        # Initialize
        x = tf.reshape(tf.range(0,24*5), [2,3,4,5])

        # Target
        x_target = tf.transpose(x, perm=[0,3,1,2]) # Now has shape == [2,5,3,4]
        x_target = tf.reshape(x_target, [2,5,12])
        x_target = tf.transpose(x_target, perm=[0,2,1]) # Now has shape == [2,12,5]

        # Observe
        x_observed = utt.flatten_along_axes(x=x, axes=[1,2])

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

    def test_4D_input_axes_1_3(self):
        """Test whether the flatten_along_axes function manages to flatten a 4D input along axes 1 and 3."""

        # Initialize
        x = tf.reshape(tf.range(0,24*5), [2,3,4,5])

        # Target
        x_target = tf.transpose(x, perm=[0,2,1,3]) # Now has shape == [2,4,3,5]
        x_target = tf.reshape(x_target, [2,4,15])
        x_target = tf.transpose(x_target, perm=[0,2,1]) # Now has shape == [2,15,4]

        # Observe
        x_observed = utt.flatten_along_axes(x=x, axes=[1,3])

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))

if __name__ == "__main__":
    unittest.main()