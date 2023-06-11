import gyoza.modelling.flow_layers as mfl
import unittest, copy as cp
import tensorflow as tf, numpy as np
import gyoza.modelling.standard_layers as msl
import gyoza.modelling.masks as mms
from tensorflow.python.ops.parallel_for.gradients import jacobian

class TestAdditiveCoupling(unittest.TestCase):
    
    def test_init_1_dimensional(self):
        """Tests whether an instance of AdditiveCouplingLayer can be created for a 1-dimensional coupling."""

        # Initialize
        compute_coupling_parameters = tf.keras.models.Sequential([tf.keras.layers.Dense(units=5, activation='tanh')])
        mask = mms.HeaviSide(axes=[0], shape=[5]) # Heaviside mask
        mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask)

    def test_init_2_dimensional(self):
        """Tests whether an instance of AdditiveCouplingLayer can be created for a 2-dimensional coupling."""

        # Initialize
        compute_coupling_parameters = msl.ChannelWiseConvolution2D(layer_count=1, conv2D_kwargs={'filters':1, 'kernel_size':2, 'padding':'same', 'activation':'tanh'}) 
        
        mask = mms.SquareWave2D(axes=[0,1], shape=[2,5]) 
        mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask)

    def test_call_1_dimensional(self):
        """Tests whether the call method of AdditiveCouplingLayer can do 1-dimensional coupling."""

        # Initialize
        compute_coupling_parameters = lambda x: tf.ones(shape=x.shape, dtype=tf.float32) 
        mask = mms.HeaviSide(axes=[1],shape=[4])
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask)
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
        """Tests whether the call method of AdditiveCouplingLayer can do 2-dimensional coupling."""

        # Initialize
        compute_coupling_parameters = lambda x: tf.ones(shape=x.shape, dtype=tf.float32) 
        mask = mms.SquareWave2D(axes=[1,2], shape=[2,4])
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask)
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
        """Tests whether the inverse method of AdditiveCouplingLayer can do 1-dimensional decoupling."""

        # Initialize
        compute_coupling_parameters = lambda x: tf.ones(shape=x.shape, dtype=tf.float32) 
        mask = mms.HeaviSide(axes=[1], shape=[4])
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask)
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
        """Tests whether the inverse method of AdditiveCouplingLayer can do 2-dimensional decoupling."""

        # Initialize
        compute_coupling_parameters = lambda x: tf.ones(shape=x.shape, dtype=tf.float32) 
        mask = mms.SquareWave2D(axes=[1,2], shape=[2,4])
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(0,24,dtype=tf.float32), [1,2,4,3])
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
        self.assertEqual(first=tf.reduce_sum((x_observed-x_target)**2).numpy(), second=0)

    def test_call_triangular_jacobian_1_dimensional_input_heaviside_mask(self):
        """Tests whether the call method of AdditiveCouplingLayer produces a triangular jacobian on 1-dimensional inputs."""

        # Initialize 
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=5),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.HeaviSide(axes=[0], shape=[5])
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.range(0,5,dtype=tf.float32)

        # Compute jacobian
        x_new=tf.Variable(mask.arrange(x=x)) # For J, first arrange x such that entries selected by mask are leading
        with tf.GradientTape() as tape:
            tape.watch(x_new)
            x = mask.re_arrange(x_new=x_new) # Copuling expects default arrangement of x
            y = layer(x)
            y = mask.arrange(x=y) # For J, ensure that entries selected by mask are also leading in y
            J = tape.jacobian(y,x_new)
        
        # Observe
        x_observed = np.allclose(J, np.tril(J)) or np.allclose(J, np.triu(J))
        
        # Evaluate
        self.assertEqual(first=x_observed, second=True)

    def test_call_triangular_jacobian_2_dimensional_input_heaviside_mask(self):
        """Tests whether the call method of AdditiveCouplingLayer produces a triangular jacobian on 2-dimensional inputs 
        with Heaviside mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.HeaviSide(axes=[1], shape=[7])
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask)
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
        """Tests whether the call method of AdditiveCouplingLayer produces a triangular jacobian on 2-dimensional inputs 
        with a square wave 1D mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.SquareWave1D(axes=[1], shape=[7])
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask)
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

    def test_call_triangular_jacobian_2_dimensional_input_square_wave_2_d_mask(self):
        """Tests whether the call method of AdditiveCouplingLayer produces a triangular jacobian on 2-dimensional inputs 
        with square wave 2D mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.SquareWave2D(axes=[0,1], shape=[2,7])
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14,dtype=tf.float32), [2,7])

        # Compute jacobian
        x_new=tf.Variable(mask.arrange(x=x)) # For J, first arrange x such that entries selected by mask are leading
        with tf.GradientTape() as tape:
            tape.watch(x_new)
            x = mask.re_arrange(x_new=x_new) # Copuling expects default arrangement of x
            y = layer(x)
            y = mask.arrange(x=y) # For J, ensure that entries selected by mask are also leading in y
            J = tape.jacobian(y,x_new)
        
        # Observe
        x_observed = np.allclose(J, np.tril(J)) or np.allclose(J, np.triu(J))
        
        # Evaluate
        self.assertEqual(first=x_observed, second=True)

    def test_call_triangular_jacobian_4_dimensional_input_square_wave_2_d_mask(self):
        """Tests whether the call method of AdditiveCouplingLayer produces a triangular jacobian on 4-dimensional inputs 
        with square wave 2D mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=1, kernel_size=[2,2], padding='same')
        mask = mms.SquareWave2D(axes=[1,2], shape=[5,6])
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask)
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
        """Tests whether the compute_jacobian_determinant method of AdditiveCouplingLayer correctly computes the determinant on 
        2-dimensional inputs with a square wave 1D mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.SquareWave1D(axes=[1], shape=[7])
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask)
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

    def test_compute_jacobian_determinant_2_dimensional_input_square_wave_2_d_mask(self):
        """Tests whether the compute_jacobian_determinant method of AdditiveCouplingLayer correctly computes the determinant on 
        2-dimensional inputs with a square wave 2D mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.SquareWave2D(axes=[0,1], shape=[2,7])
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14,dtype=tf.float32), [2,7])

        # Observe
        x_observed = layer.compute_jacobian_determinant(x=x[tf.newaxis])[0] # newaxis because we need to have a batch of one element

        # Target
        x_new=tf.Variable(mask.arrange(x=x)) # For J, first arrange x such that entries selected by mask are leading
        with tf.GradientTape() as tape:
            tape.watch(x_new)
            x = mask.re_arrange(x_new=x_new) # Copuling expects default arrangement of x
            y = layer(x)
            y = mask.arrange(x=y) # For J, ensure that entries selected by mask are also leading in y
            J = tape.jacobian(y,x_new)
        
        # Evaluate
        self.assertEqual(first=x_observed, second=np.log(np.linalg.det(J.numpy())))  

    def test_compute_jacobian_determinant_4_dimensional_input_square_wave_2_d_mask(self):
        """Tests whether the call method of AdditiveCouplingLayer produces a triangular jacobian on 4-dimensional inputs 
        with square wave 2D mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=1, kernel_size=[2,2], padding='same')
        mask = mms.SquareWave2D(axes=[1,2], shape=[5,6])
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask)
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

if __name__ == "__main__":
    #unittest.main()
    TestAdditiveCoupling().test_call_triangular_jacobian_4_dimensional_input_square_wave_2_d_mask()