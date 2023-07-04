import gyoza.modelling.masks as mms
import unittest
import tensorflow as tf

class TestHeaviSide(unittest.TestCase):
    
    def test_mask_one_axis_even_length(self):
        """Tests whether the mask function of HeaviSide works on a 1 axis input of even length."""

        # Initialize
        x = tf.range(10, dtype=tf.keras.backend.floatx())

        # Target
        x_target = tf.concat([tf.zeros([5]), tf.range(5,10, dtype=tf.keras.backend.floatx())], axis=0)

        # Observe
        instance = mms.HeaviSide(axes=[0], shape=[10])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_one_axis_odd_length(self):
        """Tests whether the mask function of HeaviSide works on a 1 axis input of odd length."""

        # Initialize
        x = tf.range(11, dtype=tf.keras.backend.floatx())

        # Target
        x_target = tf.concat([tf.zeros([5]), tf.range(5,11, dtype=tf.keras.backend.floatx())], axis=0)

        # Observe
        instance = mms.HeaviSide(axes=[0], shape=[11])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_two_axes_axis_1(self):
        """Tests whether the mask function of HeaviSide works on a two axes input along axis 1."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.keras.backend.floatx()), shape=[3,5])

        # Target
        x_target = tf.constant([[0,0,2,3,4],[0,0,7,8,9],[0,0,12,13,14]], dtype=tf.keras.backend.floatx())

        # Observe
        instance = mms.HeaviSide(axes=[1], shape=[5])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_two_axes_axis_0(self):
        """Tests whether the mask function of HeaviSide works on a two axes input along axis 0."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.keras.backend.floatx()), shape=[3,5])

        # Target
        x_target = tf.constant([[0,0,0,0,0],[5,6,7,8,9],[10,11,12,13,14]], dtype=tf.keras.backend.floatx())

        # Observe
        instance = mms.HeaviSide(axes=[0], shape=[3])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_two_axes_axis_1_negative(self):
        """Tests whether the mask function of HeaviSide works on a two axes input along axis 1
        with a negative mask."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.keras.backend.floatx()), shape=[3,5])

        # Target
        x_target = tf.constant([[0,1,0,0,0],[5,6,0,0,0],[10,11,0,0,0]], dtype=tf.keras.backend.floatx())

        # Observe
        instance = mms.HeaviSide(axes=[1], shape=[5])
        x_observed = instance.call(x=x, is_positive=False)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_one_axis_odd(self):
        """Tests whether the arrange method of HeaviSide works on a 1 axis input of odd length."""

        # Initialize
        x = tf.range(11, dtype=tf.keras.backend.floatx())

        # Target
        x_target = tf.concat([x[5:],x[:5]],axis=0)

        # Observe
        instance = mms.HeaviSide(axes=[0], shape=[11])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_two_axes_axis_1(self):
        """Tests whether the arrange method of HeaviSide works on a two axes input along axis 1."""

        # Initialize
        x = tf.reshape(tf.range(21, dtype=tf.keras.backend.floatx()), shape=[3,7])

        # Target
        x_target = tf.concat([x[:,3:],x[:,:3]],axis=1)

        # Observe
        instance = mms.HeaviSide(axes=[1], shape=[7])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_one_axis_odd(self):
        """Tests whether the re_arrange method of HeaviSide works on a 1 axis input of odd length."""

        # Initialize
        x = tf.constant([5,6,7,8,9,10, 0,1,2,3,4], dtype=tf.keras.backend.floatx())

        # Target
        x_target = tf.range(0,11, dtype=tf.keras.backend.floatx())

        # Observe
        instance = mms.HeaviSide(axes=[0], shape=[11])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_two_axes_axis_1(self):
        """Tests whether the re_arrange method of HeaviSide works on a two axes input along axis 1."""

        # Initialize
        x = tf.constant([[3,4,5,6,0,1,2],[10,11,12,13,7,8,9]], dtype=tf.keras.backend.floatx())

        # Target
        x_target = tf.reshape(tf.range(0,14, dtype=tf.keras.backend.floatx()), [2,7])

        # Observe
        instance = mms.HeaviSide(axes=[1], shape=[7])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)


class TestSquareWaveSingleAxis(unittest.TestCase):
    def test_mask_one_axis_even_length(self):

        """Tests whether the mask method of SquareWaveSingleAxis works on a 1 axis input of even length."""

        # Initialize
        x = tf.range(10, dtype=tf.keras.backend.floatx())

        # Target
        x_target = x.numpy()
        x_target[::2] = 0
        x_target = tf.constant(x_target)

        # Observe
        instance = mms.SquareWaveSingleAxis(axes=[0], shape=[10])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_two_axes_axis_1(self):
        """Tests whether the mask method of SquareWaveSingleAxis works on a two axes input along axis 1."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.keras.backend.floatx()), shape=[3,5])

        # Target
        x_target = x.numpy()
        x_target[:,::2] = 0
        x_target = tf.constant(x_target)

        # Observe
        instance = mms.SquareWaveSingleAxis(axes=[1], shape=[5])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_two_axes_axis_0(self):
        """Tests whether the mask method of SquareWaveSingleAxis works on a two axes input along axis 0."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.keras.backend.floatx()), shape=[3,5])

        # Target
        x_target = x.numpy()
        x_target[::2,:] = 0
        x_target = tf.constant(x_target)

        # Observe
        instance = mms.SquareWaveSingleAxis(axes=[0], shape=[3])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_two_axes_axis_1_negative(self):
        """Tests whether the mask method of SquareWaveSingleAxis works on a two axes input along axis 1
        with a negative mask."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.keras.backend.floatx()), shape=[3,5])

        # Target
        x_target = x.numpy()
        x_target[:,1::2] = 0
        x_target = tf.constant(x_target)

        # Observe
        instance = mms.SquareWaveSingleAxis(axes=[1], shape=[5])
        x_observed = instance.call(x=x, is_positive=False)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_one_axis_odd(self):
        """Tests whether the arrange method of SquareWaveSingleAxis works on a 1 axis input of odd length."""

        # Initialize
        x = tf.constant([5,2,6,1,3], dtype=tf.keras.backend.floatx())

        # Target
        x_target = tf.constant([2,1,5,6,3], dtype=tf.keras.backend.floatx())

        # Observe
        instance = mms.SquareWaveSingleAxis(axes=[0], shape=[5])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_two_axes_axis_0(self):
        """Tests whether the arrange method of SquareWaveSingleAxis works on a two axes input along axis 0."""

        # Initialize
        x = tf.constant([[5,2,6,1,3],[8,6,2,4,0]], dtype=tf.keras.backend.floatx())

        # Target
        x_target = tf.constant([[8,6,2,4,0],[5,2,6,1,3]], dtype=tf.keras.backend.floatx())

        # Observe
        instance = mms.SquareWaveSingleAxis(axes=[0], shape=[2])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_two_axes_axis_1(self):
        """Tests whether the arrange method of SquareWaveSingleAxis works on a two axes input along axis 1."""

        # Initialize
        x = tf.constant([[5,2,6,1,3],[8,6,2,4,0]], dtype=tf.keras.backend.floatx())

        # Target
        x_target = tf.constant([[2,1,5,6,3],[6,4,8,2,0]], dtype=tf.keras.backend.floatx())

        # Observe
        instance = mms.SquareWaveSingleAxis(axes=[1], shape=[5])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_one_axis_odd(self):
        """Tests whether the re_arrange method of SquareWaveSingleAxis works on a 1 axis input of odd length."""

        # Initialize
        x = tf.constant([2,1,5,6,3], dtype=tf.keras.backend.floatx())

        # Target
        x_target = tf.constant([5,2,6,1,3], dtype=tf.keras.backend.floatx())

        # Observe
        instance = mms.SquareWaveSingleAxis(axes=[0], shape=[5])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_two_axes_axis_0(self):
        """Tests whether the re_arrange method of SquareWaveSingleAxis works on a two axes input along axis 0."""

        # Initialize
        x = tf.constant([[8,6,2,4,0],[5,2,6,1,3]], dtype=tf.keras.backend.floatx())

        # Target
        x_target = tf.constant([[5,2,6,1,3],[8,6,2,4,0]], dtype=tf.keras.backend.floatx())

        # Observe
        instance = mms.SquareWaveSingleAxis(axes=[0], shape=[2])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_two_axes_axis_1(self):
        """Tests whether the re_arrange method of SquareWaveSingleAxis works on a two axes input along axis 1."""

        # Initialize
        x = tf.constant([[2,1,5,6,3],[6,4,8,2,0]], dtype=tf.keras.backend.floatx())

        # Target
        x_target = tf.constant([[5,2,6,1,3],[8,6,2,4,0]], dtype=tf.keras.backend.floatx())

        # Observe
        instance = mms.SquareWaveSingleAxis(axes=[1], shape=[5])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

class TestSquareWaveTwoAxes(unittest.TestCase):

    def test_mask_two_axes_axes_0_1(self):
        """Tests whether the mask method of SquareWaveTwoAxes works on a two axes input along axes 0 and 1."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.keras.backend.floatx()), shape=[3,5])

        # Target
        x_target = x.numpy()
        x_target[0,::2] = 0
        x_target[1,1::2] = 0
        x_target[2,::2] = 0
        x_target = tf.constant(x_target)

        # Observe
        instance = mms.SquareWaveTwoAxes(axes=[0,1], shape=[3,5])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_three_axes_axes_1_2(self):
        """Tests whether the mask method of SquareWaveTwoAxes works on a three axes input along axes 1 and 2."""

        # Initialize
        x = tf.reshape(tf.range(30, dtype=tf.keras.backend.floatx()), shape=[2,3,5])

        # Target
        x_target = x.numpy()
        x_target[:,0,::2] = 0
        x_target[:,1,1::2] = 0
        x_target[:,2,::2] = 0
        x_target = tf.constant(x_target)

        # Observe
        instance = mms.SquareWaveTwoAxes(axes=[1,2], shape=[3,5])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_four_axes_axes_1_2(self):
        """Tests whether the mask method of SquareWaveTwoAxes works on a three axes input along axes 1 and 2."""

        # Initialize
        x = tf.reshape(tf.range(60, dtype=tf.keras.backend.floatx()), shape=[2,3,5,2])

        # Target
        x_target = x.numpy()
        x_target[:,0,::2,:] = 0
        x_target[:,1,1::2,:] = 0
        x_target[:,2,::2,:] = 0
        x_target = tf.constant(x_target)

        # Observe
        instance = mms.SquareWaveTwoAxes(axes=[1,2], shape=[3,5])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_two_axes_axis_0_1(self):
        """Tests whether the arrange method of SquareWaveTwoAxes works on a two axes input along axis 0 and 1."""

        # Initialize
        x = tf.constant([[5,2,6,1,3],
                         [8,6,2,4,0]], dtype=tf.keras.backend.floatx())

        # Target
        x_target = tf.constant([2,1,8,2,0,5,6,3,6,4], dtype=tf.keras.backend.floatx())

        # Observe
        instance = mms.SquareWaveTwoAxes(axes=[0,1], shape=[2,5])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_three_axes_axis_1_2(self):
        """Tests whether the arrange method of SquareWaveTwoAxes works on a three axes input along axis 1 and 2."""

        # Initialize
        x = tf.constant([[[5,2,6,1,3],
                         [8,6,2,4,0]],

                         [[1,4,8,3,5],
                         [3,5,2,8,4]]], dtype=tf.keras.backend.floatx())

        # Target
        x_target = tf.constant([[2,1,8,2,0,5,6,3,6,4],
                                [4,3,3,2,4,1,8,5,5,8]], dtype=tf.keras.backend.floatx())

        # Observe
        instance = mms.SquareWaveTwoAxes(axes=[1,2], shape=[2,5])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_four_axes_axis_1_2(self):
        """Tests whether the arrange method of SquareWaveTwoAxes works on a four axes input along axis 1 and 2."""

        # Initialize
        x_a = tf.constant([[[5,2,6,1,3], # First channel
                         [8,6,2,4,0]],

                         [[1,4,8,3,5],
                         [3,5,2,8,4]]], dtype=tf.keras.backend.floatx())
        
        x_b = tf.constant([[[4,6,8,1,3], # Second channel
                         [5,8,3,5,4]],

                         [[2,4,3,6,8],
                         [3,5,8,7,2]]], dtype=tf.keras.backend.floatx())
        x = tf.concat([x_a[:,:,:,tf.newaxis], x_b[:,:,:,tf.newaxis]], axis=-1)

        # Target
        x_target_a = tf.constant([[2,1,8,2,0,5,6,3,6,4], # First channel
                                  [4,3,3,2,4,1,8,5,5,8]], dtype=tf.keras.backend.floatx())

        x_target_b = tf.constant([[6,1,5,3,4,4,8,3,8,5], # Second channel
                                  [4,6,3,8,2,2,3,8,5,7]], dtype=tf.keras.backend.floatx())

        x_target = tf.concat([x_target_a[:,:,tf.newaxis], x_target_b[:,:,tf.newaxis]], axis=-1)

        # Observe
        instance = mms.SquareWaveTwoAxes(axes=[1,2], shape=[2,5])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_two_axes_axis_0_1(self):
        """Tests whether the re_arrange method of SquareWaveTwoAxes works on a two axes input along axis 0 and 1."""

        # Initialize
        x = tf.constant([2,1,8,2,0,5,6,3,6,4], dtype=tf.keras.backend.floatx())
        
        # Target
        x_target = tf.constant([[5,2,6,1,3],
                         [8,6,2,4,0]], dtype=tf.keras.backend.floatx())

        # Observe
        instance = mms.SquareWaveTwoAxes(axes=[0,1], shape=[2,5])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_three_axes_axis_1_2(self):
        """Tests whether the re_arrange method of SquareWaveTwoAxes works on a three axes input along axis 1 and 2."""

        # Initialize
        x = tf.constant([[2,1,8,2,0,5,6,3,6,4],
                         [4,3,3,2,4,1,8,5,5,8]], dtype=tf.keras.backend.floatx())

        # Target
        x_target = tf.constant([[[5,2,6,1,3],
                                 [8,6,2,4,0]],

                                [[1,4,8,3,5],
                                 [3,5,2,8,4]]], dtype=tf.keras.backend.floatx())

        # Observe
        instance = mms.SquareWaveTwoAxes(axes=[1,2], shape=[2,5])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_four_axes_axis_1_2(self):
        """Tests whether the re_arrange method of SquareWaveTwoAxes works on a four axes input along axis 1 and 2."""

        # Initialize
        x_a = tf.constant([[2,1,8,2,0,5,6,3,6,4], # First channel
                                  [4,3,3,2,4,1,8,5,5,8]], dtype=tf.keras.backend.floatx())

        x_b = tf.constant([[6,1,5,3,4,4,8,3,8,5], # Second channel
                                  [4,6,3,8,2,2,3,8,5,7]], dtype=tf.keras.backend.floatx())

        x = tf.concat([x_a[:,:,tf.newaxis], x_b[:,:,tf.newaxis]], axis=-1)

        # Target
        x_target_a = tf.constant([[[5,2,6,1,3], # First channel
                         [8,6,2,4,0]],

                         [[1,4,8,3,5],
                         [3,5,2,8,4]]], dtype=tf.keras.backend.floatx())
        
        x_target_b = tf.constant([[[4,6,8,1,3], # Second channel
                         [5,8,3,5,4]],

                         [[2,4,3,6,8],
                         [3,5,8,7,2]]], dtype=tf.keras.backend.floatx())
        x_target = tf.concat([x_target_a[:,:,:,tf.newaxis], x_target_b[:,:,:,tf.newaxis]], axis=-1)

        # Observe
        instance = mms.SquareWaveTwoAxes(axes=[1,2], shape=[2,5])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

if __name__ == "__main__":
    #unittest.main()
    TestSquareWaveTwoAxes.test_arrange_four_axes_axis_1_2(None)