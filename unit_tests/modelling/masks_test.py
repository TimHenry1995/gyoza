import gyoza.modelling.masks as mms
import unittest
import tensorflow as tf

class TestHeaviSide(unittest.TestCase):
    
    def test_mask_one_dimensional_even_length(self):
        """Tests whether the mask function of HeaviSide works on a 1 dimensional input of even length."""

        # Initialize
        x = tf.range(10, dtype=tf.float32)

        # Target
        x_target = tf.concat([tf.zeros([5]), tf.range(5,10, dtype=tf.float32)], axis=0)

        # Observe
        instance = mms.HeaviSide(axes=[0], shape=[10])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_one_dimensional_odd_length(self):
        """Tests whether the mask function of HeaviSide works on a 1 dimensional input of odd length."""

        # Initialize
        x = tf.range(11, dtype=tf.float32)

        # Target
        x_target = tf.concat([tf.zeros([5]), tf.range(5,11, dtype=tf.float32)], axis=0)

        # Observe
        instance = mms.HeaviSide(axes=[0], shape=[11])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_two_dimensional_axis_1(self):
        """Tests whether the mask function of HeaviSide works on a two dimensional input along axis 1."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.float32), shape=[3,5])

        # Target
        x_target = tf.constant([[0,0,2,3,4],[0,0,7,8,9],[0,0,12,13,14]], dtype=tf.float32)

        # Observe
        instance = mms.HeaviSide(axes=[1], shape=[5])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_two_dimensional_axis_0(self):
        """Tests whether the mask function of HeaviSide works on a two dimensional input along axis 0."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.float32), shape=[3,5])

        # Target
        x_target = tf.constant([[0,0,0,0,0],[5,6,7,8,9],[10,11,12,13,14]], dtype=tf.float32)

        # Observe
        instance = mms.HeaviSide(axes=[0], shape=[3])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_two_dimensional_axis_1_negative(self):
        """Tests whether the mask function of HeaviSide works on a two dimensional input along axis 1
        with a negative mask."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.float32), shape=[3,5])

        # Target
        x_target = tf.constant([[0,1,0,0,0],[5,6,0,0,0],[10,11,0,0,0]], dtype=tf.float32)

        # Observe
        instance = mms.HeaviSide(axes=[1], shape=[5])
        x_observed = instance.call(x=x, is_positive=False)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_one_dimensional_odd(self):
        """Tests whether the arrange method of HeaviSide works on a 1 dimensional input of odd length."""

        # Initialize
        x = tf.range(11, dtype=tf.float32)

        # Target
        x_target = tf.concat([x[5:],x[:5]],axis=0)

        # Observe
        instance = mms.HeaviSide(axes=[0], shape=[11])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_two_dimensional_axis_1(self):
        """Tests whether the arrange method of HeaviSide works on a two dimensional input along axis 1."""

        # Initialize
        x = tf.reshape(tf.range(21, dtype=tf.float32), shape=[3,7])

        # Target
        x_target = tf.concat([x[:,3:],x[:,:3]],axis=1)

        # Observe
        instance = mms.HeaviSide(axes=[1], shape=[7])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_one_dimensional_odd(self):
        """Tests whether the re_arrange method of HeaviSide works on a 1 dimensional input of odd length."""

        # Initialize
        x = tf.constant([5,6,7,8,9,10, 0,1,2,3,4], dtype=tf.float32)

        # Target
        x_target = tf.range(0,11, dtype=tf.float32)

        # Observe
        instance = mms.HeaviSide(axes=[0], shape=[11])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_two_dimensional_axis_1(self):
        """Tests whether the re_arrange method of HeaviSide works on a two dimensional input along axis 1."""

        # Initialize
        x = tf.constant([[3,4,5,6,0,1,2],[10,11,12,13,7,8,9]], dtype=tf.float32)

        # Target
        x_target = tf.reshape(tf.range(0,14, dtype=tf.float32), [2,7])

        # Observe
        instance = mms.HeaviSide(axes=[1], shape=[7])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)


class TestSquareWave1D(unittest.TestCase):

    def test_mask_one_dimensional_even_length(self):
        """Tests whether the mask method of SquareWave1D works on a 1 dimensional input of even length."""

        # Initialize
        x = tf.range(10, dtype=tf.float32)

        # Target
        x_target = x.numpy()
        x_target[::2] = 0
        x_target = tf.constant(x_target)

        # Observe
        instance = mms.SquareWave1D(axes=[0], shape=[10])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_two_dimensional_axis_1(self):
        """Tests whether the mask method of SquareWave1D works on a two dimensional input along axis 1."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.float32), shape=[3,5])

        # Target
        x_target = x.numpy()
        x_target[:,::2] = 0
        x_target = tf.constant(x_target)

        # Observe
        instance = mms.SquareWave1D(axes=[1], shape=[5])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_two_dimensional_axis_0(self):
        """Tests whether the mask method of SquareWave1D works on a two dimensional input along axis 0."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.float32), shape=[3,5])

        # Target
        x_target = x.numpy()
        x_target[::2,:] = 0
        x_target = tf.constant(x_target)

        # Observe
        instance = mms.SquareWave1D(axes=[0], shape=[3])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_two_dimensional_axis_1_negative(self):
        """Tests whether the mask method of SquareWave1D works on a two dimensional input along axis 1
        with a negative mask."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.float32), shape=[3,5])

        # Target
        x_target = x.numpy()
        x_target[:,1::2] = 0
        x_target = tf.constant(x_target)

        # Observe
        instance = mms.SquareWave1D(axes=[1], shape=[5])
        x_observed = instance.call(x=x, is_positive=False)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_one_dimensional_odd(self):
        """Tests whether the arrange method of SquareWave1D works on a 1 dimensional input of odd length."""

        # Initialize
        x = tf.constant([5,2,6,1,3], dtype=tf.float32)

        # Target
        x_target = tf.constant([2,1,5,6,3], dtype=tf.float32)

        # Observe
        instance = mms.SquareWave1D(axes=[0], shape=[5])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_two_dimensional_axis_0(self):
        """Tests whether the arrange method of SquareWave1D works on a two dimensional input along axis 0."""

        # Initialize
        x = tf.constant([[5,2,6,1,3],[8,6,2,4,0]], dtype=tf.float32)

        # Target
        x_target = tf.constant([[8,6,2,4,0],[5,2,6,1,3]], dtype=tf.float32)

        # Observe
        instance = mms.SquareWave1D(axes=[0], shape=[2])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_two_dimensional_axis_1(self):
        """Tests whether the arrange method of SquareWave1D works on a two dimensional input along axis 1."""

        # Initialize
        x = tf.constant([[5,2,6,1,3],[8,6,2,4,0]], dtype=tf.float32)

        # Target
        x_target = tf.constant([[2,1,5,6,3],[6,4,8,2,0]], dtype=tf.float32)

        # Observe
        instance = mms.SquareWave1D(axes=[1], shape=[5])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_one_dimensional_odd(self):
        """Tests whether the re_arrange method of SquareWave1D works on a 1 dimensional input of odd length."""

        # Initialize
        x = tf.constant([2,1,5,6,3], dtype=tf.float32)

        # Target
        x_target = tf.constant([5,2,6,1,3], dtype=tf.float32)

        # Observe
        instance = mms.SquareWave1D(axes=[0], shape=[5])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_two_dimensional_axis_0(self):
        """Tests whether the re_arrange method of SquareWave1D works on a two dimensional input along axis 0."""

        # Initialize
        x = tf.constant([[8,6,2,4,0],[5,2,6,1,3]], dtype=tf.float32)

        # Target
        x_target = tf.constant([[5,2,6,1,3],[8,6,2,4,0]], dtype=tf.float32)

        # Observe
        instance = mms.SquareWave1D(axes=[0], shape=[2])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_two_dimensional_axis_1(self):
        """Tests whether the re_arrange method of SquareWave1D works on a two dimensional input along axis 1."""

        # Initialize
        x = tf.constant([[2,1,5,6,3],[6,4,8,2,0]], dtype=tf.float32)

        # Target
        x_target = tf.constant([[5,2,6,1,3],[8,6,2,4,0]], dtype=tf.float32)

        # Observe
        instance = mms.SquareWave1D(axes=[1], shape=[5])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

class TestSquareWave2D(unittest.TestCase):

    def test_mask_two_dimensional_axes_0_1(self):
        """Tests whether the mask method of SquareWave2D works on a two dimensional input along axes 0 and 1."""

        # Initialize
        x = tf.reshape(tf.range(15, dtype=tf.float32), shape=[3,5])

        # Target
        x_target = x.numpy()
        x_target[0,::2] = 0
        x_target[1,1::2] = 0
        x_target[2,::2] = 0
        x_target = tf.constant(x_target)

        # Observe
        instance = mms.SquareWave2D(axes=[0,1], shape=[3,5])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_three_dimensional_axes_1_2(self):
        """Tests whether the mask method of SquareWave2D works on a three dimensional input along axes 1 and 2."""

        # Initialize
        x = tf.reshape(tf.range(30, dtype=tf.float32), shape=[2,3,5])

        # Target
        x_target = x.numpy()
        x_target[:,0,::2] = 0
        x_target[:,1,1::2] = 0
        x_target[:,2,::2] = 0
        x_target = tf.constant(x_target)

        # Observe
        instance = mms.SquareWave2D(axes=[1,2], shape=[3,5])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_mask_four_dimensional_axes_1_2(self):
        """Tests whether the mask method of SquareWave2D works on a three dimensional input along axes 1 and 2."""

        # Initialize
        x = tf.reshape(tf.range(60, dtype=tf.float32), shape=[2,3,5,2])

        # Target
        x_target = x.numpy()
        x_target[:,0,::2,:] = 0
        x_target[:,1,1::2,:] = 0
        x_target[:,2,::2,:] = 0
        x_target = tf.constant(x_target)

        # Observe
        instance = mms.SquareWave2D(axes=[1,2], shape=[3,5])
        x_observed = instance.call(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_two_dimensional_axis_0_1(self):
        """Tests whether the arrange method of SquareWave2D works on a two dimensional input along axis 0 and 1."""

        # Initialize
        x = tf.constant([[5,2,6,1,3],
                         [8,6,2,4,0]], dtype=tf.float32)

        # Target
        x_target = tf.constant([2,1,8,2,0,5,6,3,6,4], dtype=tf.float32)

        # Observe
        instance = mms.SquareWave2D(axes=[0,1], shape=[2,5])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_three_dimensional_axis_1_2(self):
        """Tests whether the arrange method of SquareWave2D works on a three dimensional input along axis 1 and 2."""

        # Initialize
        x = tf.constant([[[5,2,6,1,3],
                         [8,6,2,4,0]],

                         [[1,4,8,3,5],
                         [3,5,2,8,4]]], dtype=tf.float32)

        # Target
        x_target = tf.constant([[2,1,8,2,0,5,6,3,6,4],
                                [4,3,3,2,4,1,8,5,5,8]], dtype=tf.float32)

        # Observe
        instance = mms.SquareWave2D(axes=[1,2], shape=[2,5])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_arrange_four_dimensional_axis_1_2(self):
        """Tests whether the arrange method of SquareWave2D works on a four dimensional input along axis 1 and 2."""

        # Initialize
        x_a = tf.constant([[[5,2,6,1,3], # First channel
                         [8,6,2,4,0]],

                         [[1,4,8,3,5],
                         [3,5,2,8,4]]], dtype=tf.float32)
        
        x_b = tf.constant([[[4,6,8,1,3], # Second channel
                         [5,8,3,5,4]],

                         [[2,4,3,6,8],
                         [3,5,8,7,2]]], dtype=tf.float32)
        x = tf.concat([x_a[:,:,:,tf.newaxis], x_b[:,:,:,tf.newaxis]], axis=-1)

        # Target
        x_target_a = tf.constant([[2,1,8,2,0,5,6,3,6,4], # First channel
                                  [4,3,3,2,4,1,8,5,5,8]], dtype=tf.float32)

        x_target_b = tf.constant([[6,1,5,3,4,4,8,3,8,5], # Second channel
                                  [4,6,3,8,2,2,3,8,5,7]], dtype=tf.float32)

        x_target = tf.concat([x_target_a[:,:,tf.newaxis], x_target_b[:,:,tf.newaxis]], axis=-1)

        # Observe
        instance = mms.SquareWave2D(axes=[1,2], shape=[2,5])
        x_observed = instance.arrange(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_two_dimensional_axis_0_1(self):
        """Tests whether the re_arrange method of SquareWave2D works on a two dimensional input along axis 0 and 1."""

        # Initialize
        x = tf.constant([2,1,8,2,0,5,6,3,6,4], dtype=tf.float32)
        
        # Target
        x_target = tf.constant([[5,2,6,1,3],
                         [8,6,2,4,0]], dtype=tf.float32)

        # Observe
        instance = mms.SquareWave2D(axes=[0,1], shape=[2,5])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_three_dimensional_axis_1_2(self):
        """Tests whether the re_arrange method of SquareWave2D works on a three dimensional input along axis 1 and 2."""

        # Initialize
        x = tf.constant([[2,1,8,2,0,5,6,3,6,4],
                         [4,3,3,2,4,1,8,5,5,8]], dtype=tf.float32)

        # Target
        x_target = tf.constant([[[5,2,6,1,3],
                                 [8,6,2,4,0]],

                                [[1,4,8,3,5],
                                 [3,5,2,8,4]]], dtype=tf.float32)

        # Observe
        instance = mms.SquareWave2D(axes=[1,2], shape=[2,5])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_re_arrange_four_dimensional_axis_1_2(self):
        """Tests whether the re_arrange method of SquareWave2D works on a four dimensional input along axis 1 and 2."""

        # Initialize
        x_a = tf.constant([[2,1,8,2,0,5,6,3,6,4], # First channel
                                  [4,3,3,2,4,1,8,5,5,8]], dtype=tf.float32)

        x_b = tf.constant([[6,1,5,3,4,4,8,3,8,5], # Second channel
                                  [4,6,3,8,2,2,3,8,5,7]], dtype=tf.float32)

        x = tf.concat([x_a[:,:,tf.newaxis], x_b[:,:,tf.newaxis]], axis=-1)

        # Target
        x_target_a = tf.constant([[[5,2,6,1,3], # First channel
                         [8,6,2,4,0]],

                         [[1,4,8,3,5],
                         [3,5,2,8,4]]], dtype=tf.float32)
        
        x_target_b = tf.constant([[[4,6,8,1,3], # Second channel
                         [5,8,3,5,4]],

                         [[2,4,3,6,8],
                         [3,5,8,7,2]]], dtype=tf.float32)
        x_target = tf.concat([x_target_a[:,:,:,tf.newaxis], x_target_b[:,:,:,tf.newaxis]], axis=-1)

        # Observe
        instance = mms.SquareWave2D(axes=[1,2], shape=[2,5])
        x_observed = instance.re_arrange(x_new=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

if __name__ == "__main__":
    #unittest.main()
    TestSquareWave2D.test_arrange_four_dimensional_axis_1_2(None)