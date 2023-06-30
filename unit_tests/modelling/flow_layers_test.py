import gyoza.modelling.flow_layers as mfl
import unittest, copy as cp
import tensorflow as tf, numpy as np
import gyoza.modelling.standard_layers as msl
import gyoza.modelling.masks as mms
from tensorflow.python.ops.parallel_for.gradients import jacobian
import os
import shutil

class TestAdditiveCoupling(unittest.TestCase):
    
    def test_init_1_dimensional(self):
        """Tests whether an instance of AdditiveCoupling can be created for a 1-dimensional coupling."""

        # Initialize
        compute_coupling_parameters = tf.keras.models.Sequential([tf.keras.layers.Dense(units=5, activation='tanh')])
        mask = mms.HeaviSide(axes=[2], shape=[5]) # Heaviside mask
        mfl.AdditiveCoupling(axes=[2], shape=[5], compute_coupling_parameters=compute_coupling_parameters, mask=mask)

    def test_init_2_dimensional(self):
        """Tests whether an instance of AdditiveCoupling can be created for a 2-dimensional coupling."""

        # Initialize
        compute_coupling_parameters = msl.ChannelWiseConvolution2D(layer_count=1, conv2D_kwargs={'filters':1, 'kernel_size':2, 'padding':'same', 'activation':'tanh'}) 
        
        mask = mms.SquareWave2D(axes=[1,2], shape=[2,5]) 
        mfl.AdditiveCoupling(axes=[1,2], shape=[2,5],compute_coupling_parameters=compute_coupling_parameters, mask=mask)

    def test_call_1_dimensional(self):
        """Tests whether the call method of AdditiveCoupling can do 1-dimensional coupling."""

        # Initialize
        compute_coupling_parameters = lambda x: tf.ones(shape=x.shape, dtype=tf.float32) 
        mask = mms.HeaviSide(axes=[1],shape=[4])
        layer = mfl.AdditiveCoupling(shape=[4], axes=[1], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(0,24,dtype=tf.float32), [2,4,3])

        # Target
        x_target = x.numpy()
        x_target[:,:2,:] += 1
        x_target = tf.constant(x_target)

        # Observe
        x_observed = layer(x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_observed-x_target)**2).numpy(), second=0)

    def test_call_2_dimensional(self):
        """Tests whether the call method of AdditiveCoupling can do 2-dimensional coupling."""

        # Initialize
        compute_coupling_parameters = lambda x: tf.ones(shape=x.shape, dtype=tf.float32) 
        mask = mms.SquareWave2D(axes=[1,2], shape=[2,4])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[2,4], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(0,24,dtype=tf.float32), [1,2,4,3])

        # Target
        x_target = x.numpy()
        x_target[0,0,::2,:]+=1
        x_target[0,1,1::2,:]+=1
        x_target = tf.constant(x_target)

        # Observe
        x_observed = layer(x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_observed-x_target)**2).numpy(), second=0)

    def test_invert_1_dimensional(self):
        """Tests whether the inverse method of AdditiveCoupling can do 1-dimensional decoupling."""

        # Initialize
        compute_coupling_parameters = lambda x: tf.ones(shape=x.shape, dtype=tf.float32) 
        mask = mms.HeaviSide(axes=[1], shape=[4])
        layer = mfl.AdditiveCoupling(axes=[1], shape=[4], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(0,24,dtype=tf.float32), [2,4,3])
        y_hat = x.numpy()
        y_hat[:,:2,:] += 1
        y_hat = tf.constant(y_hat)

        # Target
        x_target = x

        # Observe
        x_observed = layer.invert(y_hat=y_hat)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_observed-x_target)**2).numpy(), second=0)

    def test_invert_2_dimensional(self):
        """Tests whether the inverse method of AdditiveCoupling can do 2-dimensional decoupling."""

        # Initialize
        compute_coupling_parameters = lambda x: tf.ones(shape=x.shape, dtype=tf.float64) 
        mask = mms.SquareWave2D(axes=[1,2], shape=[2,4])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[2,4], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.random.normal([1,2,4,3], dtype=tf.float64)
        y_hat = x.numpy()
        y_hat[0,0,::2,:] +=1
        y_hat[0,1,1::2,:] += 1
        y_hat = tf.constant(y_hat)

        # Target
        x_target = x

        # Observe
        x_observed = layer.invert(y_hat=y_hat)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_sum((x_observed-x_target)**2).numpy(), second=0)

    def test_call_triangular_jacobian_2_dimensional_input_heaviside_mask(self):
        """Tests whether the call method of AdditiveCoupling produces a triangular jacobian on 2-dimensional inputs 
        with Heaviside mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.HeaviSide(axes=[1], shape=[7])
        layer = mfl.AdditiveCoupling(axes=[1], shape=[7], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14,dtype=tf.float32), [2,7])

        # Compute jacobian
        x_new=tf.Variable(mask.arrange(x=x)) # For J, first arrange x such that entries selected by mask are leading
        with tf.GradientTape() as tape:
            tape.watch(x_new)
            x = mask.re_arrange(x_new=x_new) # Copuling expects default arrangement of x
            y = layer(x)
            y = mask.arrange(x=y) # For J, ensure that entries selected by mask are also leading in y
            J = tape.jacobian(y,x_new)
        J = tf.reduce_sum(J, axis=2) # This axis is redundant, see section on batch jacobians in https://www.tensorflow.org/guide/advanced_autodiff#jacobians
        
        # Observe
        x_observed = True
        for j in range(J.shape[0]):
            x_observed = x_observed and (np.allclose(J[j], np.tril(J[j])) or np.allclose(J[j], np.triu(J[j])))
        
        # Evaluate
        self.assertEqual(first=x_observed, second=True)

    def test_call_triangular_jacobian_2_dimensional_input_square_wave_1_d_mask(self):
        """Tests whether the call method of AdditiveCoupling produces a triangular jacobian on 2-dimensional inputs 
        with a square wave 1D mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.SquareWave1D(axes=[1], shape=[7])
        layer = mfl.AdditiveCoupling(axes=[1], shape=[7], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14,dtype=tf.float32), [2,7])

        # Compute jacobian
        x_new=tf.Variable(mask.arrange(x=x)) # For J, first arrange x such that entries selected by mask are leading
        with tf.GradientTape() as tape:
            tape.watch(x_new)
            x = mask.re_arrange(x_new=x_new) # Copuling expects default arrangement of x
            y = layer(x)
            y = mask.arrange(x=y) # For J, ensure that entries selected by mask are also leading in y
            J = tape.jacobian(y,x_new)
        J = tf.reduce_sum(J, axis=2) # This axis is redundant, see section on batch jacobians in https://www.tensorflow.org/guide/advanced_autodiff#jacobians
        
        # Observe
        x_observed = True
        for j in range(J.shape[0]):
            x_observed = x_observed and (np.allclose(J[j], np.tril(J[j])) or np.allclose(J[j], np.triu(J[j])))
        
        # Evaluate
        self.assertEqual(first=x_observed, second=True)

    def test_call_triangular_jacobian_3_dimensional_input_square_wave_2_d_mask(self):
        """Tests whether the call method of AdditiveCoupling produces a triangular jacobian on 3-dimensional inputs 
        with square wave 2D mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.SquareWave2D(axes=[1,2], shape=[2,7])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[2,7], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14*3,dtype=tf.float32), [3,2,7])

        # Compute jacobian
        x_new=tf.Variable(mask.arrange(x=x)) # For J, first arrange x such that entries selected by mask are leading
        with tf.GradientTape() as tape:
            tape.watch(x_new)
            x = mask.re_arrange(x_new=x_new) # Copuling expects default arrangement of x
            y = layer(x)
            y = mask.arrange(x=y) # For J, ensure that entries selected by mask are also leading in y
            J = tape.jacobian(y,x_new)
        J = tf.reduce_sum(J, axis=2)

        # Observe
        x_observed = np.allclose(J, np.tril(J)) or np.allclose(J, np.triu(J))
        
        # Evaluate
        self.assertEqual(first=x_observed, second=True)

    def test_call_triangular_jacobian_4_dimensional_input_square_wave_2_d_mask(self):
        """Tests whether the call method of AdditiveCoupling produces a triangular jacobian on 4-dimensional inputs 
        with square wave 2D mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=1, kernel_size=[2,2], padding='same')
        mask = mms.SquareWave2D(axes=[1,2], shape=[5,6])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[5,6], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(2*5*6,dtype=tf.float32), [2,5,6,1]) # Shape == [batch size, height, width, channel count]

        # Compute jacobian
        x_new=tf.Variable(mask.arrange(x=x)) # For J, first arrange x such that entries selected by mask are leading
        with tf.GradientTape() as tape:
            tape.watch(x_new)
            x = mask.re_arrange(x_new=x_new) # Copuling expects default arrangement of x
            y = layer(x)
            y = mask.arrange(x=y) # For J, ensure that entries selected by mask are also leading in y
            J = tape.jacobian(y,x_new)
        J = tf.reduce_sum(J, axis=5) # This axis is redundant, see section on batch jacobians in https://www.tensorflow.org/guide/advanced_autodiff#jacobians
        J = tf.reduce_sum(J, axis=3)
        J = tf.reduce_sum(J, axis=2)

        # Observe
        x_observed = True
        for j in range(J.shape[0]):
            x_observed = x_observed and (np.allclose(J, np.tril(J)) or np.allclose(J, np.triu(J)))
        
        # Evaluate
        self.assertEqual(first=x_observed, second=True)

    def test_compute_jacobian_determinant_2_dimensional_input_square_wave_1_d_mask(self):
        """Tests whether the compute_jacobian_determinant method of AdditiveCoupling correctly computes the determinant on 
        2-dimensional inputs with a square wave 1D mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.SquareWave1D(axes=[1], shape=[7])
        layer = mfl.AdditiveCoupling(axes=[1], shape=[7], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14,dtype=tf.float32), [2,7])

        # Observe
        x_observed = layer.compute_jacobian_determinant(x=x)

        # Target
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = layer(x)
            J = tape.jacobian(y,x)
        J = tf.reduce_sum(J, axis=2) # This axis is redundant, see section on batch jacobians in https://www.tensorflow.org/guide/advanced_autodiff#jacobians
        
        # Evaluate
        for j in range(J.shape[0]):
            self.assertEqual(first=x_observed[j], second=np.log(np.linalg.det(J[j].numpy())))  

    def test_compute_jacobian_determinant_3_dimensional_input_square_wave_2_d_mask(self):
        """Tests whether the compute_jacobian_determinant method of AdditiveCoupling correctly computes the determinant on 
        3-dimensional inputs with a square wave 2D mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.SquareWave2D(axes=[1,2], shape=[2,7])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[2,7], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14*3,dtype=tf.float32), [3,2,7])

        # Observe
        x_observed = layer.compute_jacobian_determinant(x=x)

        # Target
        x_new=tf.Variable(mask.arrange(x=x)) # For J, first arrange x such that entries selected by mask are leading
        with tf.GradientTape() as tape:
            tape.watch(x_new)
            x = mask.re_arrange(x_new=x_new) # Copuling expects default arrangement of x
            y = layer(x)
            y = mask.arrange(x=y) # For J, ensure that entries selected by mask are also leading in y
            J = tape.jacobian(y,x_new)
        J = tf.reduce_sum(J, axis=2) # This axis is redundant, see section on batch jacobians in https://www.tensorflow.org/guide/advanced_autodiff#jacobians
        
        # Evaluate
        for j in range(J.shape[0]):
            self.assertEqual(first=x_observed[j].numpy(), second=np.log(np.linalg.det(J[j].numpy())))  

    def test_compute_jacobian_determinant_4_dimensional_input_square_wave_2_d_mask(self):
        """Tests whether the call method of AdditiveCoupling produces a triangular jacobian on 4-dimensional inputs 
        with square wave 2D mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=1, kernel_size=[2,2], padding='same')
        mask = mms.SquareWave2D(axes=[1,2], shape=[5,6])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[5,6], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(2*5*6,dtype=tf.float32), [2,5,6,1]) # Shape == [batch size, height, width, channel count]

        # Observe
        x_observed = layer.compute_jacobian_determinant(x=x)

        # Compute jacobian
        x_new=tf.Variable(mask.arrange(x=x)) # For J, first arrange x such that entries selected by mask are leading
        with tf.GradientTape() as tape:
            tape.watch(x_new)
            x = mask.re_arrange(x_new=x_new) # Copuling expects default arrangement of x
            y = layer(x)
            y = mask.arrange(x=y) # For J, ensure that entries selected by mask are also leading in y
            J = tape.jacobian(y,x_new)
        J = tf.reduce_sum(J, axis=5) # This axis is redundant, see section on batch jacobians in https://www.tensorflow.org/guide/advanced_autodiff#jacobians
        J = tf.reduce_sum(J, axis=3)
        J = tf.reduce_sum(J, axis=2)

        # Evaluate
        for j in range(J.shape[0]):
             self.assertEqual(first=x_observed[j], second=np.log(np.linalg.det(J[j].numpy())))


    def test_load_and_save(self):

        """Tests whether the model provides the same shuffling after persistent storage."""
        # Initialize
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=1, kernel_size=[2,2], padding='same')
        mask = mms.SquareWave2D(axes=[1,2], shape=[5,6])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[5,6], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(2*5*6,dtype=tf.float32), [2,5,6,1]) # Shape == [batch size, height, width, channel count]

        # Observe first
        y_hat_1 = layer(x=x)
        
        # Save and delete
        path = os.path.join(os.getcwd(), "temporary_model_directory_for_additive_coupling_layer_unit_test.h5")
        layer.save_weights(path)
        del layer, mask, compute_coupling_parameters
        
        # Initialize again and load
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=1, kernel_size=[2,2], padding='same')
        mask = mms.SquareWave2D(axes=[1,2], shape=[5,6])
        loaded_layer = mfl.AdditiveCoupling(axes=[1,2], shape=[5,6], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        loaded_layer.build(input_shape=x.shape) # Warm-up phase to initialize all weights
        loaded_layer.load_weights(path)
        os.remove(path)
    
        # Observe second
        y_hat_2 = loaded_layer(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y_hat_1.shape), tuple2=tuple(y_hat_2.shape))
        self.assertEqual(first=tf.reduce_sum((y_hat_1-y_hat_2)**2).numpy(), second=0)

class TestShuffle(unittest.TestCase):

    def test_call_and_inverse_2D_input_along_1_axis(self):
        """Tests whether the inverse method is indeed providing the inverse of the call on a 2D input along 1 axis."""
        
        # Initialize
        channel_count = 100
        shuffling_layer = mfl.Shuffle(shape=[channel_count], axes=[1])
        x = tf.random.uniform(shape=[10, channel_count], dtype=tf.float32)
        
        # Observe
        y_hat = shuffling_layer(x=x)
        x_hat = shuffling_layer.invert(y_hat=y_hat)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x.shape), tuple2=tuple(x_hat.shape))
        self.assertEqual(first=tf.reduce_sum((x-x_hat)**2).numpy(), second=0)

    def test_call_and_inverse_3D_input_along_2_axes(self):
        """Tests whether the inverse method is indeed providing the inverse of the call on a 3D input along both axes."""
        
        # Initialize
        batch_size = 2; width = 4; height = 5
        shuffling_layer = mfl.Shuffle(shape=[width, height], axes=[1,2])
        x = tf.random.uniform(shape=[batch_size, width, height], dtype=tf.float32)
        
        # Observe
        y_hat = shuffling_layer(x=x)
        x_hat = shuffling_layer.invert(y_hat=y_hat)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x.shape), tuple2=tuple(x_hat.shape))
        self.assertEqual(first=tf.reduce_sum((x-x_hat)**2).numpy(), second=0)

    def test_call_reliability_2D_input_along_1_axis(self):
        """Tests whether call reproduces itself when called 2 times in a row on a 2D input along 1 axis."""
        
        # Initialize
        channel_count = 100
        shuffling_layer = mfl.Shuffle(shape=[channel_count], axes=[1])
        x = tf.random.uniform(shape=[10, channel_count], dtype=tf.float32)
        
        # Observe
        y_hat_1 = shuffling_layer(x=x)
        y_hat_2 = shuffling_layer(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y_hat_1.shape), tuple2=tuple(y_hat_2.shape))
        self.assertEqual(first=tf.reduce_sum((y_hat_1-y_hat_2)**2).numpy(), second=0)

    def test_call_reliability_3D_input_along_2_axes(self):
        """Tests whether call reproduces itself when called 2 times in a row on a 3D input along 2 axes."""
        
        # Initialize
        batch_size = 2; width = 4; height = 5
        shuffling_layer = mfl.Shuffle(shape=[width, height], axes=[1,2])
        x = tf.random.uniform(shape=[batch_size, width, height], dtype=tf.float32)
         
        # Observe
        y_hat_1 = shuffling_layer(x=x)
        y_hat_2 = shuffling_layer(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y_hat_1.shape), tuple2=tuple(y_hat_2.shape))
        self.assertEqual(first=tf.reduce_sum((y_hat_1-y_hat_2)**2).numpy(), second=0)

    def test_load_and_save_4D_input_along_2_axes(self):
        """Tests whether the model provides the same shuffling after persistent storage."""

        # Initialize
        width = 100; height = 200; channel_count = 3
        shuffling_layer = mfl.Shuffle(shape=[width, height], axes=[1,2])
        x = tf.random.uniform(shape=[10, width, height, channel_count], dtype=tf.float32)
        
        # Observe first
        y_hat_1 = shuffling_layer(x=x)
        
        # Save and load
        path = os.path.join(os.getcwd(), "temporary_model_directory_for_shuffle_model_unit_test.h5")
        shuffling_layer.save_weights(path)
        del shuffling_layer
        loaded_shuffling_layer = mfl.Shuffle(shape=[width, height], axes=[1,2])
        loaded_shuffling_layer.build(input_shape=x.shape)
        loaded_shuffling_layer.load_weights(path)
        os.remove(path)
    
        y_hat_2 = loaded_shuffling_layer(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y_hat_1.shape), tuple2=tuple(y_hat_2.shape))
        self.assertEqual(first=tf.reduce_sum((y_hat_1-y_hat_2)**2).numpy(), second=0)

class TestActivationNormalization(unittest.TestCase):
    def test_init_1_dimensional(self):
        """Tests whether an instance of ActivationNormalization can be created for a 1-dimensional input."""

        # Initialize
        mfl.ActivationNormalization(axes=[2], shape=[5])

    def test_init_2_dimensional(self):
        """Tests whether an instance of ActivationNormalization can be created for a 2-dimensional input."""

        # Initialize
        mfl.ActivationNormalization(axes=[1,2], shape=[2,5])
    
    def test_call_2D_input_along_axis_1(self):
        """Tests whether the call method of ActivatonNormalization can normalize 2D inputs along axis 1."""

        # Initialize
        layer = mfl.ActivationNormalization(shape=[3], axes=[1])
        x = tf.reshape(tf.range(0,24,dtype=tf.float32), [8,3])

        # Target
        l_target = tf.zeros([3])
        s_target = tf.ones([3])

        # Observe
        x_observed = layer(x)
        l_observed = tf.math.reduce_mean(x_observed, axis=0)
        s_observed = tf.math.reduce_std(x_observed, axis=0)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(l_target.shape), tuple2=tuple(l_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_mean((l_observed-l_target)**2).numpy(), second=0)

        self.assertTupleEqual(tuple1=tuple(s_target.shape), tuple2=tuple(s_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_mean((s_observed-s_target)**2).numpy(), second=0)

    def test_call_3D_input_along_axes_1_2(self):
        """Tests whether the call method of ActivatonNormalization can normalize 3D inputs along axes 1 and 2."""

        # Initialize
        layer = mfl.ActivationNormalization(shape=[3,4], axes=[1,2])
        x = tf.reshape(tf.range(0,24,dtype=tf.float32), [2,3,4])

        # Target
        l_target = tf.zeros([3,4])
        s_target = tf.ones([3,4])

        # Observe
        x_observed = layer(x)
        l_observed = tf.math.reduce_mean(x_observed, axis=0)
        s_observed = tf.math.reduce_std(x_observed, axis=0)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(l_target.shape), tuple2=tuple(l_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_mean((l_observed-l_target)**2).numpy(), second=0)

        self.assertTupleEqual(tuple1=tuple(s_target.shape), tuple2=tuple(s_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_mean((s_observed-s_target)**2).numpy(), second=0)

    def test_call_3D_input_along_axes_1(self):
        """Tests whether the call method of ActivatonNormalization can normalize 3D inputs along axis 1."""

        # Initialize
        layer = mfl.ActivationNormalization(shape=[3], axes=[1])
        x = tf.reshape(tf.range(0,24,dtype=tf.float32), [2,3,4])

        # Target
        l_target = tf.zeros([3])
        s_target = tf.ones([3])

        # Observe
        x_observed = layer(x)
        x_observed = tf.reshape(x_observed, [8,3])
        l_observed = tf.math.reduce_mean(x_observed, axis=0)
        s_observed = tf.math.reduce_std(x_observed, axis=0)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(l_target.shape), tuple2=tuple(l_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_mean((l_observed-l_target)**2).numpy(), second=0)

        self.assertTupleEqual(tuple1=tuple(s_target.shape), tuple2=tuple(s_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_mean((s_observed-s_target)**2).numpy(), second=0)

    def test_call_4D_input_along_axis_1(self):
        """Tests whether the call method of ActivatonNormalization can normalize 4D inputs along axis 1."""

        # Initialize
        layer = mfl.ActivationNormalization(shape=[3], axes=[1])
        x = tf.reshape(tf.range(0,24*5,dtype=tf.float32), [2,3,4,5])

        # Target
        l_target = tf.zeros([3])
        s_target = tf.ones([3])

        # Observe
        x_observed = layer(x)
        x_observed = tf.reshape(x_observed, [40,3])
        l_observed = tf.math.reduce_mean(x_observed, axis=0)
        s_observed = tf.math.reduce_std(x_observed, axis=0)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(l_target.shape), tuple2=tuple(l_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_mean((l_observed-l_target)**2).numpy(), second=0)

        self.assertTupleEqual(tuple1=tuple(s_target.shape), tuple2=tuple(s_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_mean((s_observed-s_target)**2).numpy(), second=0)

    def test_call_4D_input_along_axes_1_2(self):
        """Tests whether the call method of ActivatonNormalization can normalize 4D inputs along axes 1 and 2."""

        # Initialize
        layer = mfl.ActivationNormalization(shape=[3,4], axes=[1,2])
        x = tf.reshape(tf.range(0,24*5,dtype=tf.float32), [2,3,4,5])

        # Target
        l_target = tf.zeros([3,4])
        s_target = tf.ones([3,4])

        # Observe
        x_observed = layer(x)
        x_observed = tf.transpose(x_observed, perm=[0,3,1,2]) # Now has shape [2,5,3,4]
        x_observed = tf.reshape(x_observed, [10,3,4])
        l_observed = tf.math.reduce_mean(x_observed, axis=0)
        s_observed = tf.math.reduce_std(x_observed, axis=0)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(l_target.shape), tuple2=tuple(l_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_mean((l_observed-l_target)**2).numpy(), second=0)

        self.assertTupleEqual(tuple1=tuple(s_target.shape), tuple2=tuple(s_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_mean((s_observed-s_target)**2).numpy(), second=0)

    def test_call_4D_input_along_axes_1_3(self):
        """Tests whether the call method of ActivatonNormalization can normalize 4D inputs along axes 1 and 3."""

        # Initialize
        layer = mfl.ActivationNormalization(shape=[3,5], axes=[1,3])
        x = tf.reshape(tf.range(0,24*5,dtype=tf.float32), [2,3,4,5])

        # Target
        l_target = tf.zeros([3,5])
        s_target = tf.ones([3,5])

        # Observe
        x_observed = layer(x)
        x_observed = tf.transpose(x_observed, perm=[0,2,1,3]) # Now has shape [2,4,3,5]
        x_observed = tf.reshape(x_observed, [8,3,5])
        l_observed = tf.math.reduce_mean(x_observed, axis=0)
        s_observed = tf.math.reduce_std(x_observed, axis=0)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(l_target.shape), tuple2=tuple(l_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_mean((l_observed-l_target)**2).numpy(), second=0)

        self.assertTupleEqual(tuple1=tuple(s_target.shape), tuple2=tuple(s_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_mean((s_observed-s_target)**2).numpy(), second=0)

    def test_compute_jacobian_determinant_2_dimensional_axis_1(self):
        """Tests whether the activation normalization layer can compute the jacobian determinant on 2D inputs"""
                
        # Initialize
        layer = mfl.ActivationNormalization(shape=[3], axes=[1])
        x = tf.reshape(tf.range(0,1,delta=1/24,dtype=tf.float32), [8,3])

        # Compute jacobian
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = layer(x)
            J = tape.jacobian(y,x)
        J = tf.reduce_sum(J, axis=2) # This axis is redundant, see section on batch jacobians in https://www.tensorflow.org/guide/advanced_autodiff#jacobians
        
        # Verify triangularity
        is_triangular = True
        for j in range(J.shape[0]):
            is_triangular = is_triangular and (np.allclose(J[j], np.tril(J[j])) or np.allclose(J[j], np.triu(J[j])))
        
        self.assertEqual(first=is_triangular, second=True)
        
        # Verify determinants
        x_observed = layer.compute_jacobian_determinant(x=x)
        for j in range(J.shape[0]):
            self.assertEqual(first=np.log(np.prod(np.diagonal(J[j].numpy()))), second=x_observed[j].numpy())

    def test_compute_jacobian_determinant_3_dimensional_axis_1(self):
        """Tests whether the activation normalization layer can compute the jacobian determinant on 3D inputs"""
                
        # Initialize
        layer = mfl.ActivationNormalization(shape=[3], axes=[1])
        x = tf.reshape(tf.range(0,1,delta=1/60,dtype=tf.float32), [5,3,4])

        # Compute jacobian
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = layer(x)
            J = tape.jacobian(y,x)
        J = tf.reduce_sum(J, axis=3) # This axis is redundant, see section on batch jacobians in https://www.tensorflow.org/guide/advanced_autodiff#jacobians
        J = tf.reshape(J, [5,12,12])

        # Verify triangularity
        is_triangular = True
        for j in range(J.shape[0]):
            is_triangular = is_triangular and (np.allclose(J[j], np.tril(J[j])) or np.allclose(J[j], np.triu(J[j])))
        
        self.assertEqual(first=is_triangular, second=True)
        
        # Verify determinants
        x_observed = layer.compute_jacobian_determinant(x=x)
        for j in range(J.shape[0]):
            self.assertAlmostEqual(first=np.log(np.prod(np.diagonal(J[j].numpy()))), second=x_observed[j].numpy())

    def test_compute_jacobian_determinant_3_dimensional_axis_2(self):
        """Tests whether the activation normalization layer can compute the jacobian determinant on 3D inputs"""
                
        # Initialize
        layer = mfl.ActivationNormalization(shape=[4], axes=[2])
        x = tf.reshape(tf.range(0,1,delta=1/60,dtype=tf.float32), [5,3,4])

        # Compute jacobian
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = layer(x)
            J = tape.jacobian(y,x)
        J = tf.reduce_sum(J, axis=3) # This axis is redundant, see section on batch jacobians in https://www.tensorflow.org/guide/advanced_autodiff#jacobians
        J = tf.reshape(J, [5,12,12])

        # Verify triangularity
        is_triangular = True
        for j in range(J.shape[0]):
            is_triangular = is_triangular and (np.allclose(J[j], np.tril(J[j])) or np.allclose(J[j], np.triu(J[j])))
        
        self.assertEqual(first=is_triangular, second=True)
        
        # Verify determinants
        x_observed = layer.compute_jacobian_determinant(x=x)
        for j in range(J.shape[0]):
            self.assertAlmostEqual(first=np.log(np.prod(np.diagonal(J[j].numpy()))), second=x_observed[j].numpy(), places=5)

    def test_compute_jacobian_determinant_3_dimensional_axes_1_2(self):
        """Tests whether the activation normalization layer can compute the jacobian determinant on 3D inputs"""
                
        # Initialize
        layer = mfl.ActivationNormalization(shape=[3,4], axes=[1,2])
        x = tf.reshape(tf.range(0,1,delta=1/60,dtype=tf.float32), [5,3,4])

        # Compute jacobian
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = layer(x)
            J = tape.jacobian(y,x)
        J = tf.reduce_sum(J, axis=3) # This axis is redundant, see section on batch jacobians in https://www.tensorflow.org/guide/advanced_autodiff#jacobians
        J = tf.reshape(J, [5,12,12])

        # Verify triangularity
        is_triangular = True
        for j in range(J.shape[0]):
            is_triangular = is_triangular and (np.allclose(J[j], np.tril(J[j])) or np.allclose(J[j], np.triu(J[j])))
        
        self.assertEqual(first=is_triangular, second=True)
        
        # Verify determinants
        x_observed = layer.compute_jacobian_determinant(x=x)
        for j in range(J.shape[0]):
            self.assertAlmostEqual(first=np.log(np.prod(np.diagonal(J[j].numpy()))), second=x_observed[j].numpy())

    def test_compute_jacobian_determinant_4_dimensional_axes_1_2(self):
        """Tests whether the activation normalization layer can compute the jacobian determinant on 4D inputs"""
                
        # Initialize
        layer = mfl.ActivationNormalization(shape=[3,4], axes=[1,2])
        x = tf.reshape(tf.range(0,1,delta=1/120,dtype=tf.float32), [5,3,4,2])

        # Compute jacobian
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = layer(x)
            J = tape.jacobian(y,x)
        J = tf.reduce_sum(J, axis=4) # Second batch axis is redundant, see section on batch jacobians in https://www.tensorflow.org/guide/advanced_autodiff#jacobians
        J = tf.reduce_sum(J, axis=6) # Second channel axis
        J = tf.reduce_prod(J, axis=3) # 
        J = tf.reshape(J, [5,12,12])

        # Verify triangularity
        is_triangular = True
        for j in range(J.shape[0]):
            is_triangular = is_triangular and (np.allclose(J[j], np.tril(J[j])) or np.allclose(J[j], np.triu(J[j])))
        
        self.assertEqual(first=is_triangular, second=True)
        
        # Verify determinants
        x_observed = layer.compute_jacobian_determinant(x=x)
        for j in range(J.shape[0]):
            self.assertAlmostEqual(first=np.log(np.prod(np.diagonal(J[j].numpy()))), second=x_observed[j].numpy())

if __name__ == "__main__":
    #unittest.main()
    TestActivationNormalization.test_compute_jacobian_determinant_3_dimensional_axis_2(None)