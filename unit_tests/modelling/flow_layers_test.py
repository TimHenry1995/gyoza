import gyoza.modelling.flow_layers as mfl
import unittest, copy as cp
import tensorflow as tf, numpy as np
import gyoza.modelling.standard_layers as msl

class TestAdditiveCoupling(unittest.TestCase):
    
    def test_init_1_dimensional(self):
        """Tests whether an instance of AdditiveCouplingLayer can be created for a 1-dimensional coupling."""

        # Initialize
        compute_coupling_parameters = tf.keras.models.Sequential([tf.keras.layers.Dense(units=5, activation='tanh')])
        mask = tf.constant([0,0,1,1,1]) # Heaviside mask
        mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask, axes=[1])

    def test_init_2_dimensional(self):
        """Tests whether an instance of AdditiveCouplingLayer can be created for a 2-dimensional coupling."""

        # Initialize
        compute_coupling_parameters = msl.ChannelWiseConvolution2D(layer_count=1, conv2D_kwargs={'filters':1, 'kernel_size':2, 'padding':'same', 'activation':'tanh'}) 
        
        mask = tf.constant([[0,1,0,1,0],[1,0,1,0,1]]) # Checkerboard mask
        mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask, axes=[1,2])

    def test_call_1_dimensional(self):
        """Tests whether the call method of AdditiveCouplingLayer can do 1-dimensional coupling."""

        # Initialize
        compute_coupling_parameters = lambda x: tf.ones(shape=x.shape, dtype=tf.float32) 
        mask = tf.constant([0,0,1,1]) # Heaviside mask
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask, axes=[1])
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
        mask = tf.constant([[0,1,0,1],[1,0,1,0]]) # Checkerboard mask
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask, axes=[1,2])
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
        mask = tf.constant([0,0,1,1]) # Heaviside mask
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask, axes=[1])
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
        mask = tf.constant([[0,1,0,1],[1,0,1,0]]) # Heaviside mask
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask, axes=[1,2])
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

    def test_invert_1_dimensional(self):
        """Tests whether the inverse method of AdditiveCouplingLayer can do 1-dimensional decoupling."""

        # Initialize
        compute_coupling_parameters = lambda x: tf.ones(shape=x.shape, dtype=tf.float32) 
        mask = tf.constant([0,0,1,1]) # Heaviside mask
        layer = mfl.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask, axes=[1])
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


if __name__ == "__main__":
    unittest.main()