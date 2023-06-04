import gyoza.utilities.tensors as utt
import unittest
import tensorflow as tf

class Test_Tensors_Move_Axis(unittest.TestCase):

    def test_same_axis(self):
        """Test whether the move_axis method manages to move the axis to its current position.
        This is an identity operation."""

        # Initialize
        x = tf.reshape(tf.range(0,24), [2,3,4])

        # Target
        x_target = x

        # Observe
        x_observed = utt.move_axis(x=x, from_index=1, to_index=1)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        


if __name__ == "__main__":
    unittest.main()