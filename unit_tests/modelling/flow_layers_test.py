import gyoza.modelling.flow_layers as mfl
import unittest
from gyoza.tutorials import data_synthesis as gtds
import tensorflow as tf, numpy as np
import gyoza.modelling.masks as mms
from typing import Dict, Any
import os


class ChannelWiseConvolutionTwoAxes(tf.keras.Model):
    """This class provides a sequential convolutional neural network that applies the same spatial filters to each dimension.

    :param layer_count: The number of layers.
    :type layer_count: int
    :param conv2D_kwargs: The kew-word arguments for the :class:`tensorflow.keras.layers.Conv2D` layers that are used here.
        **Important**: The channel_axis is assumed to be the default, i.e. the last axis.
    """

    def __init__(self, layer_count:int = 3, conv2D_kwargs: Dict[str, Any] = {}):

        # Super
        super(ChannelWiseConvolutionTwoAxes, self).__init__()

        # Create layers
        layers = [None] * (layer_count + 2)
        layers[0] = tf.keras.layers.Lambda(lambda x: tf.transpose(x[:,tf.newaxis,:,:,:], [0,4,2,3,1]))
        for l in range(layer_count): layers[l+1] = tf.keras.layers.Conv2D(**conv2D_kwargs)
        layers[-1] = tf.keras.layers.Lambda(lambda x: tf.squeeze(tf.transpose(x, [0,4,2,3,1])))

        # Attributes
        self.sequential = tf.keras.models.Sequential(layers=layers)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        
        # Predict
        y_hat = self.sequential(x)

        # Outputs:
        return y_hat
    
class TestAdditiveCoupling(unittest.TestCase):
    
    def test_init_1_axis(self):
        """Tests whether an instance of AdditiveCoupling can be created for a 1-axis coupling."""

        # Initialize
        compute_coupling_parameters = tf.keras.models.Sequential([tf.keras.layers.Dense(units=5, activation='tanh')])
        mask = mms.Heaviside(axes=[2], shape=[5]) # Heaviside mask
        mfl.AdditiveCoupling(axes=[2], shape=[5], compute_coupling_parameters=compute_coupling_parameters, mask=mask)

    def test_init_2_axes(self):
        """Tests whether an instance of AdditiveCoupling can be created for a 2-axes coupling."""

        # Initialize
        compute_coupling_parameters = ChannelWiseConvolutionTwoAxes(layer_count=1, conv2D_kwargs={'filters':1, 'kernel_size':2, 'padding':'same', 'activation':'tanh'}) 
        
        mask = mms.CheckerBoard(axes=[1,2], shape=[2,5]) 
        mfl.AdditiveCoupling(axes=[1,2], shape=[2,5],compute_coupling_parameters=compute_coupling_parameters, mask=mask)

    def test_call_1_axis(self):
        """Tests whether the call method of AdditiveCoupling can do 1-axis coupling."""

        # Initialize
        compute_coupling_parameters = tf.keras.layers.Lambda(lambda x: tf.ones(shape=x.shape, dtype=tf.keras.backend.floatx()))
        mask = mms.Heaviside(axes=[1],shape=[4])
        layer = mfl.AdditiveCoupling(shape=[4], axes=[1], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(0,24,dtype=tf.keras.backend.floatx()), [2,4,3])

        # Target
        x_target = x.numpy()
        x_target[:,:2,:] += 1
        x_target = tf.constant(x_target)

        # Observe
        x_observed = layer(x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertEqual(first=tf.reduce_sum((x_observed-x_target)**2).numpy(), second=0)

    def test_call_2_axes(self):
        """Tests whether the call method of AdditiveCoupling can do 2-axes coupling."""

        # Initialize
        compute_coupling_parameters = tf.keras.layers.Lambda(lambda x: tf.ones(shape=x.shape, dtype=tf.keras.backend.floatx()))
        mask = mms.CheckerBoard(axes=[1,2], shape=[2,4])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[2,4], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(0,24,dtype=tf.keras.backend.floatx()), [1,2,4,3])

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

    def test_invert_1_axis(self):
        """Tests whether the inverse method of AdditiveCoupling can do 1-axis decoupling."""

        # Initialize
        compute_coupling_parameters = lambda x: tf.ones(shape=x.shape, dtype=tf.keras.backend.floatx()) 
        mask = mms.Heaviside(axes=[1], shape=[4])
        layer = mfl.AdditiveCoupling(axes=[1], shape=[4], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(0,24,dtype=tf.keras.backend.floatx()), [2,4,3])
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

    def test_invert_2_axes(self):
        """Tests whether the inverse method of AdditiveCoupling can do 2-axes decoupling."""

        # Initialize
        compute_coupling_parameters = lambda x: tf.ones(shape=x.shape, dtype=tf.keras.backend.floatx()) 
        mask = mms.CheckerBoard(axes=[1,2], shape=[2,4])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[2,4], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.random.normal([1,2,4,3], dtype=tf.keras.backend.floatx())
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

    def test_call_triangular_jacobian_2_axes_input_heaviside_mask(self):
        """Tests whether the call method of AdditiveCoupling produces a triangular jacobian on 2-axes inputs 
        with Heaviside mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.Heaviside(axes=[1], shape=[7])
        layer = mfl.AdditiveCoupling(axes=[1], shape=[7], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14,dtype=tf.keras.backend.floatx()), [2,7])

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

    def test_call_triangular_jacobian_2_axes_input_checker_board_1_axis_mask(self):
        """Tests whether the call method of AdditiveCoupling produces a triangular jacobian on 2-axes inputs 
        with a checker board 1 axis mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.CheckerBoard(axes=[1], shape=[7])
        layer = mfl.AdditiveCoupling(axes=[1], shape=[7], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14,dtype=tf.keras.backend.floatx()), [2,7])

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

    def test_call_triangular_jacobian_3_axes_input_checker_board_2_axes_mask(self):
        """Tests whether the call method of AdditiveCoupling produces a triangular jacobian on 3-axes inputs 
        with checker board 2_axes mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.CheckerBoard(axes=[1,2], shape=[2,7])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[2,7], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14*3,dtype=tf.keras.backend.floatx()), [3,2,7])

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

    def test_call_triangular_jacobian_4_axes_input_checker_board_2_axes_mask(self):
        """Tests whether the call method of AdditiveCoupling produces a triangular jacobian on 4-axes inputs 
        with checker board 2_axes mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=1, kernel_size=[2,2], padding='same')
        mask = mms.CheckerBoard(axes=[1,2], shape=[5,6])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[5,6], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(2*5*6,dtype=tf.keras.backend.floatx()), [2,5,6,1]) # Shape == [batch size, height, width, channel count]

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

    def test_compute_jacobian_determinant_2_axes_input_chcker_board_1_axis_mask(self):
        """Tests whether the compute_jacobian_determinant method of AdditiveCoupling correctly computes the determinant on 
        2-axes inputs with a checker board 1 axis mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.CheckerBoard(axes=[1], shape=[7])
        layer = mfl.AdditiveCoupling(axes=[1], shape=[7], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14,dtype=tf.keras.backend.floatx()), [2,7])

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

    def test_compute_jacobian_determinant_3_axes_input_checker_board_2_axes_mask(self):
        """Tests whether the compute_jacobian_determinant method of AdditiveCoupling correctly computes the determinant on 
        3-axes inputs with a checker board 2 axes mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.CheckerBoard(axes=[1,2], shape=[2,7])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[2,7], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14*3,dtype=tf.keras.backend.floatx()), [3,2,7])

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

    def test_compute_jacobian_determinant_4_axes_input_checker_board_2_axes_mask(self):
        """Tests whether the call method of AdditiveCoupling produces a triangular jacobian on 4-axes inputs 
        with checker board 2 axes mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=1, kernel_size=[2,2], padding='same')
        mask = mms.CheckerBoard(axes=[1,2], shape=[5,6])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[5,6], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(2*5*6,dtype=tf.keras.backend.floatx()), [2,5,6,1]) # Shape == [batch size, height, width, channel count]

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
        def create_flow_model(input_shape):
            """Function to create a fresh, unbuilt model architecture."""
            
            # Create coupling layer
            compute_coupling_parameters = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=[2,2], padding='same')
            mask = mms.CheckerBoard(axes=[1,2], shape=input_shape[-3:-1])
            
            # Create flow model
            model = mfl.FlowModel(flow_layers=[
                mfl.AdditiveCoupling(axes=[1,2], shape=input_shape[-3:-1], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
            ])

            # Build
            input = tf.keras.Input(shape=input_shape[-3:])    
            model.build(input_shape=tf.keras.ops.shape(input)) 

            # Output
            return model

        input_shape = [2,5,6,3] # Batch size 2, height 5, width 6, channels 3
        model = create_flow_model(input_shape=input_shape)
        x = tf.reshape(tf.range(2*5*6*3,dtype=tf.keras.backend.floatx()), input_shape) 

        # Observe 
        y_hat_1 = model(x=x)
        path = os.path.join(os.getcwd(), "temporary_model_directory_for_additive_coupling_layer_unit_test.weights.h5")
        model.save_weights(path)

        # Delete everything and clear session
        del model
        tf.keras.backend.clear_session() 

        # Initialize the second model 
        loaded_model = create_flow_model(input_shape=input_shape)

        # Load the weights (will now match the topology exactly)
        loaded_model.load_weights(path)
        os.remove(path)
            
        # Observe second - this should now match y_hat_1
        y_hat_2 = loaded_model(x=x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y_hat_1.shape), tuple2=tuple(y_hat_2.shape))
        self.assertEqual(first=tf.reduce_sum((y_hat_1-y_hat_2)**2).numpy(), second=0)

    def test_forward_inverse_roundtrip(self):
        """Tests whether a model can reconstruct its input"""

        # Setup
        input_shape = [16, 9, 9, 3] # Batch size, widht, height, channel-count

        # Create coupling layer
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=[2,2], padding='same')
        mask = mms.CheckerBoard(axes=[1,2], shape=input_shape[-3:-1])
        
        # Create layer
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=input_shape[-3:-1], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        
        # Build
        input = tf.keras.Input(shape=input_shape[-3:])    
        layer.build(input_shape=tf.keras.ops.shape(input)) 

        @tf.function
        def roundtrip(x):
            y = layer(x=x)
            return layer.invert(y)

        x = tf.random.normal(input_shape)
        x_rec = roundtrip(x)

        self.assertEqual(x.shape, x_rec.shape)
        self.assertLess(tf.reduce_max(tf.abs(x - x_rec)), 1e-5)

    def test_variable_batch_sizes(self):
        """Test variable batch-size tracing test for FlowModel with AdditiveCoupling layer."""
        
        # Setup
        input_shape = [None, 9, 9, 3] # Batch size, widht, height, channel-count

        # Create coupling layer
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=[2,2], padding='same')
        mask = mms.CheckerBoard(axes=[1,2], shape=input_shape[-3:-1])
        
        # Create flow model
        model = mfl.FlowModel(flow_layers=[
            mfl.AdditiveCoupling(axes=[1,2], shape=input_shape[-3:-1], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        ])

        # Build
        input = tf.keras.Input(shape=input_shape[-3:])    
        model.build(input_shape=tf.keras.ops.shape(input)) 

        @tf.function
        def traced_forward(x):
            return model(x)

        for batch_size in [1, 5, 17, 64]:
            x = tf.random.normal([batch_size] + input_shape[-3:])
            y = traced_forward(x)
            self.assertEqual(y.shape, x.shape)

    def test_earger_vs_graph_call_equivalence(self):
        """Tests whether the call method of AdditiveCoupling is equivalent in graphs mode and in regular eager mode."""

        # Setup
        input_shape = [16, 9, 9, 3] # Batch size, widht, height, channel-count

        # Create coupling layer
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=[2,2], padding='same')
        mask = mms.CheckerBoard(axes=[1,2], shape=input_shape[-3:-1])
        
        # Create flow model
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=input_shape[-3:-1], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.random.uniform(shape=input_shape, dtype=tf.keras.backend.floatx())
        
        # Observe
        x_target = layer(x=x)

        # Graph mode
        @tf.function
        def graph_call(x): return layer(x=x)
        x_observed = graph_call(x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_max((x_observed-x_target)**2).numpy(), second=0)

    def test_earger_vs_graph_determinant_equivalence(self):
        """Tests whether the call method of AdditiveCoupling is equivalent in graphs mode and in regular eager mode."""

        # Setup
        input_shape = [16, 9, 9, 3] # Batch size, widht, height, channel-count

        # Create coupling layer
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=[2,2], padding='same')
        mask = mms.CheckerBoard(axes=[1,2], shape=input_shape[-3:-1])
        
        # Create flow model
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=input_shape[-3:-1], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.random.uniform(shape=input_shape, dtype=tf.keras.backend.floatx())
        
        # Observe
        det_target = layer.compute_jacobian_determinant(x=x)
        
        # Graph mode
        @tf.function
        def graph_determinant(x): return layer.compute_jacobian_determinant(x=x)

        det_observed = graph_determinant(x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(det_target.shape), tuple2=tuple(det_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_max((det_observed-det_target)**2).numpy(), second=0)

    def test_earger_vs_graph_round_trip_equivalence(self):
        """Tests whether the call method of AdditiveCoupling is equivalent in graphs mode and in regular eager mode."""

        # Setup
        input_shape = [16, 9, 9, 3] # Batch size, widht, height, channel-count

        # Create coupling layer
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=[2,2], padding='same')
        mask = mms.CheckerBoard(axes=[1,2], shape=input_shape[-3:-1])
        
        # Create flow model
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=input_shape[-3:-1], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.random.uniform(shape=input_shape, dtype=tf.keras.backend.floatx())
        
        # Observe
        x_target = layer.invert(y_hat=layer(x))
        
        # Graph mode
        @tf.function
        def graph_round_trip(x): return layer.invert(y_hat=layer(x=x))

        x_observed = graph_round_trip(x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_max((x_observed-x_target)**2).numpy(), second=0)

    def test_build_layer(self):
        """Tests whether the build method of AdditiveCoupling works as intended."""

        # Initialize
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=1, kernel_size=[2,2], padding='same')
        mask = mms.CheckerBoard(axes=[1,2], shape=[5,6])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[5,6], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        input_shape = (10, 5, 6, 3) # Batch size 10, height 5, width 6, channels 3

        # Build
        layer.build(input_shape=input_shape)

        # Evaluate
        self.assertEqual(first=layer.built, second=True)
        self.assertEqual(first=layer._compute_coupling_parameters_.built, second=True)

    def test_build_model(self):
        """Tests whether the build method of AdditiveCoupling works as intended inside a FlowModel."""

        # Initialize
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=1, kernel_size=[2,2], padding='same')
        mask = mms.CheckerBoard(axes=[1,2], shape=[5,6])
        model = mfl.FlowModel(flow_layers=[
            mfl.AdditiveCoupling(axes=[1,2], shape=[5,6], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        ])
        input_shape = (10, 5, 6, 3) # Batch size 10, height 5, width 6, channels 3

        # Build
        model.build(input_shape=input_shape)

        # Evaluate
        self.assertEqual(first=model.built, second=True)
        self.assertEqual(first=model.flow_layers[0].built, second=True)
        self.assertEqual(first=model.flow_layers[0]._compute_coupling_parameters_.built, second=True)

class TestShuffle(unittest.TestCase):

    def test_call_and_inverse_2_axes_input_along_1_axis(self):
        """Tests whether the inverse method is indeed providing the inverse of the call on a 2_axes input along 1 axis."""
        
        # Initialize
        dimension_count = 100
        shuffling_layer = mfl.Shuffle(shape=[dimension_count], axes=[1])
        x = tf.random.uniform(shape=[10, dimension_count], dtype=tf.keras.backend.floatx())
        
        # Observe
        y_hat = shuffling_layer(x=x)
        x_hat = shuffling_layer.invert(y_hat=y_hat)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x.shape), tuple2=tuple(x_hat.shape))
        self.assertEqual(first=tf.reduce_sum((x-x_hat)**2).numpy(), second=0)

    def test_call_and_inverse_3_axes_input_along_2_axes(self):
        """Tests whether the inverse method is indeed providing the inverse of the call on a 3_axes input along both axes."""
        
        # Initialize
        batch_size = 2; width = 4; height = 5
        shuffling_layer = mfl.Shuffle(shape=[width, height], axes=[1,2])
        x = tf.random.uniform(shape=[batch_size, width, height], dtype=tf.keras.backend.floatx())
        
        # Observe
        y_hat = shuffling_layer(x=x)
        x_hat = shuffling_layer.invert(y_hat=y_hat)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x.shape), tuple2=tuple(x_hat.shape))
        self.assertEqual(first=tf.reduce_sum((x-x_hat)**2).numpy(), second=0)

    def test_call_reliability_2_axes_input_along_1_axis(self):
        """Tests whether call reproduces itself when called 2 times in a row on a 2_axes input along 1 axis."""
        
        # Initialize
        dimension_count = 100
        shuffling_layer = mfl.Shuffle(shape=[dimension_count], axes=[1])
        x = tf.random.uniform(shape=[10, dimension_count], dtype=tf.keras.backend.floatx())
        
        # Observe
        y_hat_1 = shuffling_layer(x=x)
        y_hat_2 = shuffling_layer(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y_hat_1.shape), tuple2=tuple(y_hat_2.shape))
        self.assertEqual(first=tf.reduce_sum((y_hat_1-y_hat_2)**2).numpy(), second=0)

    def test_call_reliability_3_axes_input_along_2_axes(self):
        """Tests whether call reproduces itself when called 2 times in a row on a 3_axes input along 2 axes."""
        
        # Initialize
        batch_size = 2; width = 4; height = 5
        shuffling_layer = mfl.Shuffle(shape=[width, height], axes=[1,2])
        x = tf.random.uniform(shape=[batch_size, width, height], dtype=tf.keras.backend.floatx())
         
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
        model = mfl.FlowModel([shuffling_layer])
        x = tf.random.uniform(shape=[10, width, height, channel_count], dtype=tf.keras.backend.floatx())
        
        # Observe first
        y_hat_1 = model(x=x)
        
        # Save and load
        path = os.path.join(os.getcwd(), "temporary_model_directory_for_shuffle_model_unit_test.weights.h5")
        model.save_weights(path)
        del model
        loaded_shuffling_layer = mfl.Shuffle(shape=[width, height], axes=[1,2])
        loaded_model = mfl.FlowModel([loaded_shuffling_layer])
        loaded_model.build(input_shape=x.shape)
        loaded_model.load_weights(path)
        os.remove(path)
    
        y_hat_2 = loaded_model(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y_hat_1.shape), tuple2=tuple(y_hat_2.shape))
        self.assertEqual(first=tf.reduce_sum((y_hat_1-y_hat_2)**2).numpy(), second=0)

    def test_earger_vs_graph_call_equivalence(self):
        """Tests whether the call method of Shuffle is equivalent in graphs mode and in regular eager mode."""

        # Initialize
        batch_size = 2; width = 4; height = 5
        layer = mfl.Shuffle(shape=[width, height], axes=[1,2])
        x = tf.random.uniform(shape=[batch_size, width, height], dtype=tf.keras.backend.floatx())
        
        # Observe
        x_target = layer(x=x)

        # Graph mode
        @tf.function
        def graph_call(x): return layer(x=x)
        x_observed = graph_call(x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_max((x_observed-x_target)**2).numpy(), second=0)

    def test_earger_vs_graph_determinant_equivalence(self):
        """Tests whether the call method of Shuffle is equivalent in graphs mode and in regular eager mode."""

        # Initialize
        batch_size = 2; width = 4; height = 5
        layer = mfl.Shuffle(shape=[width, height], axes=[1,2])
        x = tf.random.uniform(shape=[batch_size, width, height], dtype=tf.keras.backend.floatx())
        
        # Observe
        det_target = layer.compute_jacobian_determinant(x=x)
        
        # Graph mode
        @tf.function
        def graph_determinant(x): return layer.compute_jacobian_determinant(x=x)

        det_observed = graph_determinant(x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(det_target.shape), tuple2=tuple(det_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_max((det_observed-det_target)**2).numpy(), second=0)

    def test_earger_vs_graph_round_trip_equivalence(self):
        """Tests whether the call method of Shuffle is equivalent in graphs mode and in regular eager mode."""

        # Initialize
        batch_size = 2; width = 4; height = 5
        layer = mfl.Shuffle(shape=[width, height], axes=[1,2])
        x = tf.random.uniform(shape=[batch_size, width, height], dtype=tf.keras.backend.floatx())
        
        # Observe
        x_target = layer.invert(y_hat=layer(x))
        
        # Graph mode
        @tf.function
        def graph_round_trip(x): return layer.invert(y_hat=layer(x=x))

        x_observed = graph_round_trip(x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_max((x_observed-x_target)**2).numpy(), second=0)

    def test_build_layer(self):
        """Tests whether the build method of Shuffle works as intended."""

        # Initialize
        batch_size = 2; width = 4; height = 5
        layer = mfl.Shuffle(shape=[width, height], axes=[1,2])
        
        # Build
        layer.build(input_shape=[batch_size, width, height])

        # Evaluate
        self.assertEqual(first=layer.built, second=True)
        self.assertEqual(first=hasattr(layer, '_forward_permutation_'), second=True)
        self.assertEqual(first=hasattr(layer, '_inverse_permutation_'), second=True)

    def test_build_model(self):
        """Tests whether the build method of Shuffle works as intended inside a FlowModel."""

        # Initialize
        batch_size = 2; width = 4; height = 5
        layer = mfl.Shuffle(shape=[width, height], axes=[1,2])
        model = mfl.FlowModel(flow_layers=[layer])
        input_shape = (batch_size, width, height)

        # Build
        model.build(input_shape=input_shape)

        # Evaluate
        self.assertEqual(first=model.built, second=True)
        self.assertEqual(first=model.flow_layers[0].built, second=True)
        self.assertEqual(first=hasattr(model.flow_layers[0], '_forward_permutation_'), second=True)
        self.assertEqual(first=hasattr(model.flow_layers[0], '_inverse_permutation_'), second=True)

class TestCheckerBoard(unittest.TestCase):

    def test_call_2_axes_input_1_even_axis_permutation(self):
        """Tests whether the call method correctly swops indices on a 2 axis input with even dimension count along the permutation axis."""
        
        # Initialize
        permutation_layer = mfl.CheckerBoard(shape=[4], axes=[1])
        x = tf.constant([[4,6,3,2], [1,3,7,8]], dtype=tf.keras.backend.floatx())
        y = tf.constant([[6,4,2,3], [3,1,8,7]], dtype=tf.keras.backend.floatx())

        # Observe
        y_hat = permutation_layer(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y.shape), tuple2=tuple(y_hat.shape))
        self.assertEqual(first=tf.reduce_sum((y-y_hat)**2).numpy(), second=0)

    def test_call_2_axes_input_1_odd_axis_permutation(self):
        """Tests whether the call method correctly swops indices on a 2 axis input with odd dimension count along the permutation axis."""
        
        # Initialize
        permutation_layer = mfl.CheckerBoard(shape=[5], axes=[1])
        x = tf.constant([[4,6,3,2,1], [1,3,7,8,4]], dtype=tf.keras.backend.floatx())
        y = tf.constant([[6,4,2,3,1], [3,1,8,7,4]], dtype=tf.keras.backend.floatx())

        # Observe
        y_hat = permutation_layer(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y.shape), tuple2=tuple(y_hat.shape))
        self.assertEqual(first=tf.reduce_sum((y-y_hat)**2).numpy(), second=0)

    def test_call_3_axes_input_2_even_even_axes_permutation(self):
        """Tests whether the call method correctly swops indices on a 3 axis input with even dimension count along the two permutation axes."""
        
        # Initialize
        permutation_layer = mfl.CheckerBoard(shape=[2,4], axes=[1,2])
        x = tf.constant([[[4,6,3,2], [1,3,7,8]],
                         [[8,3,5,2], [9,7,8,2]]], dtype=tf.keras.backend.floatx())
        y = tf.constant([[[6,4,2,3], [3,1,8,7]],
                         [[3,8,2,5], [7,9,2,8]]], dtype=tf.keras.backend.floatx())

        # Observe
        y_hat = permutation_layer(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y.shape), tuple2=tuple(y_hat.shape))
        self.assertEqual(first=tf.reduce_sum((y-y_hat)**2).numpy(), second=0)

    def test_call_3_axes_input_2_even_odd_axes_permutation(self):
        """Tests whether the call method correctly swops indices on a 3 axis input with even dimension count along the first and odd count along the second permutation axis."""
        
        # Initialize
        permutation_layer = mfl.CheckerBoard(shape=[2,5], axes=[1,2])
        x = tf.constant([[[4,6,3,2,9], [1,3,7,8,5]],
                         [[8,3,5,2,4], [9,7,8,2,1]]], dtype=tf.keras.backend.floatx())
        y = tf.constant([[[6,4,2,3,5], [3,1,8,7,9]],
                         [[3,8,2,5,1], [7,9,2,8,4]]], dtype=tf.keras.backend.floatx())

        # Observe
        y_hat = permutation_layer(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y.shape), tuple2=tuple(y_hat.shape))
        self.assertEqual(first=tf.reduce_sum((y-y_hat)**2).numpy(), second=0)

    def test_call_3_axes_input_2_odd_odd_axes_permutation(self):
        """Tests whether the call method correctly swops indices on a 3 axis input with odd dimension count along the two permutation axes."""
        
        # Initialize
        permutation_layer = mfl.CheckerBoard(shape=[3,5], axes=[1,2])
        x = tf.constant([[[4,6,3,2,9], [1,3,7,8,5], [5,3,7,9,0]],
                         [[8,3,5,2,4], [9,7,8,2,1], [1,3,6,0,9]]], dtype=tf.keras.backend.floatx())
        y = tf.constant([[[6,4,2,3,5], [3,1,8,7,9], [3,5,9,7,0]],
                         [[3,8,2,5,1], [7,9,2,8,4], [3,1,0,6,9]]], dtype=tf.keras.backend.floatx())

        # Observe
        y_hat = permutation_layer(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y.shape), tuple2=tuple(y_hat.shape))
        self.assertEqual(first=tf.reduce_sum((y-y_hat)**2).numpy(), second=0)

    def test_call_4_axes_3_even_even_even_permutation(self):
        """Tests whether the call method correctly swops indices on a 4 axis input with even dimension count along the three permutation axes."""
        
        # Initialize
        permutation_layer = mfl.CheckerBoard(shape=[2,2,4], axes=[1,2,3])
        x = tf.constant([[[[4,6,3,2], [1,3,7,8]],
                          [[8,3,5,2], [9,7,8,2]]],
                         [[[7,3,4,2], [5,8,9,6]],
                          [[0,2,4,6], [0,1,3,2]]], 
                          ], dtype=tf.keras.backend.floatx())

        y = tf.constant([[[[6,4,2,3], [3,1,8,7]],
                          [[3,8,2,5], [7,9,2,8]]],
                         [[[3,7,2,4], [8,5,6,9]],
                          [[2,0,6,4], [1,0,2,3]]]], dtype=tf.keras.backend.floatx())

        # Observe
        y_hat = permutation_layer(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y.shape), tuple2=tuple(y_hat.shape))
        self.assertEqual(first=tf.reduce_sum((y-y_hat)**2).numpy(), second=0)

    def test_call_4_axes_3_even_odd_even_permutation(self):
        """Tests whether the call method correctly swops indices on a 4 axis input with even dimension count along the first and third permutation axes and odd count along the second axis."""
        
        # Initialize
        permutation_layer = mfl.CheckerBoard(shape=[2,3,4], axes=[1,2,3])
        x = tf.constant([[[[4,6,3,2], [1,3,7,8], [3,6,9,7]],
                          [[8,3,5,2], [9,7,8,2], [1,3,6,9]]],
                         [[[7,3,4,2], [5,8,9,6], [2,4,3,7]],
                          [[0,2,4,6], [0,1,3,2], [1,7,0,5]]]], dtype=tf.keras.backend.floatx())

        y = tf.constant([[[[6,4,2,3], [3,1,8,7], [6,3,7,9]],
                          [[3,8,2,5], [7,9,2,8], [3,1,9,6]]],
                         [[[3,7,2,4], [8,5,6,9], [4,2,7,3]],
                          [[2,0,6,4], [1,0,2,3], [7,1,5,0]]]], dtype=tf.keras.backend.floatx())

        # Observe
        y_hat = permutation_layer(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y.shape), tuple2=tuple(y_hat.shape))
        self.assertEqual(first=tf.reduce_sum((y-y_hat)**2).numpy(), second=0)

    def test_call_4_axes_3_even_odd_odd_permutation(self):
        """Tests whether the call method correctly swops indices on a 4 axis input with odd dimension count along the second and third permutation axes and even count along the first axis."""
        
        # Initialize
        permutation_layer = mfl.CheckerBoard(shape=[2,3,5], axes=[1,2,3])
        x = tf.constant([[[[4,6,3,2,5], [1,3,7,8,9], [3,6,9,7,0]],
                          [[8,3,5,2,1], [9,7,8,2,3], [1,3,6,9,5]]],
                         [[[7,3,4,2,0], [5,8,9,6,3], [2,4,3,7,1]],
                          [[0,2,4,6,7], [0,1,3,2,8], [1,7,0,5,9]]]], dtype=tf.keras.backend.floatx())

        y = tf.constant([[[[6,4,2,3,9], [3,1,8,7,5], [6,3,7,9,5]],
                          [[3,8,2,5,3], [7,9,2,8,1], [3,1,9,6,0]]],
                         [[[3,7,2,4,3], [8,5,6,9,0], [4,2,7,3,9]],
                          [[2,0,6,4,8], [1,0,2,3,7], [7,1,5,0,1]]]], dtype=tf.keras.backend.floatx())

        # Observe
        y_hat = permutation_layer(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y.shape), tuple2=tuple(y_hat.shape))
        self.assertEqual(first=tf.reduce_sum((y-y_hat)**2).numpy(), second=0)

    def test_call_4_axes_3_odd_odd_odd_permutation(self):
        """Tests whether the call method correctly swops indices on a 4 axis input with odd dimension count along the three permutation axes."""
        
        # Initialize
        permutation_layer = mfl.CheckerBoard(shape=[3,3,5], axes=[1,2,3])
        x = tf.constant([[[[4,6,3,2,5], [1,3,7,8,9], [3,6,9,7,0]],
                          [[8,3,5,2,1], [9,7,8,2,3], [1,3,6,9,5]],
                          [[1,5,3,7,4], [1,3,8,7,2], [7,3,5,2,9]]],
                         [[[7,3,4,2,0], [5,8,9,6,3], [2,4,3,7,1]],
                          [[0,2,4,6,7], [0,1,3,2,8], [1,7,0,5,9]],
                          [[6,8,3,5,2], [1,3,6,9,8], [2,4,3,6,8]]]], dtype=tf.keras.backend.floatx())

        y = tf.constant([[[[6,4,2,3,9], [3,1,8,7,5], [6,3,7,9,5]],
                          [[3,8,2,5,3], [7,9,2,8,1], [3,1,9,6,0]],
                          [[5,1,7,3,2], [3,1,7,8,4], [3,7,2,5,9]]],
                         [[[3,7,2,4,3], [8,5,6,9,0], [4,2,7,3,9]],
                          [[2,0,6,4,8], [1,0,2,3,7], [7,1,5,0,1]],
                          [[8,6,5,3,8], [3,1,9,6,2], [4,2,6,3,8]]]], dtype=tf.keras.backend.floatx())

        # Observe
        y_hat = permutation_layer(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y.shape), tuple2=tuple(y_hat.shape))
        self.assertEqual(first=tf.reduce_sum((y-y_hat)**2).numpy(), second=0)

    def test_call_and_inverse_3_axes_input_even_even(self):
        """Tests whether the inverse method is indeed providing the inverse of the call on a 3 axes input along 2 axes with even dimension count."""
        
        # Initialize
        batch_size = 2; width = 4; height = 6
        permutation_layer = mfl.CheckerBoard(shape=[width, height], axes=[1,2])
        x = tf.random.uniform(shape=[batch_size, width, height], dtype=tf.keras.backend.floatx())
        
        # Observe
        y_hat = permutation_layer(x=x)
        x_hat = permutation_layer.invert(y_hat=y_hat)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x.shape), tuple2=tuple(x_hat.shape))
        self.assertEqual(first=tf.reduce_sum((x-x_hat)**2).numpy(), second=0)

    def test_call_and_inverse_3_axes_input_even_odd(self):
        """Tests whether the inverse method is indeed providing the inverse of the call on a 3 axes input along 2 axes even and odd dimension count."""
        
        # Initialize
        batch_size = 2; width = 4; height = 7
        permutation_layer = mfl.CheckerBoard(shape=[width, height], axes=[1,2])
        x = tf.random.uniform(shape=[batch_size, width, height], dtype=tf.keras.backend.floatx())
        
        # Observe
        y_hat = permutation_layer(x=x)
        x_hat = permutation_layer.invert(y_hat=y_hat)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x.shape), tuple2=tuple(x_hat.shape))
        self.assertEqual(first=tf.reduce_sum((x-x_hat)**2).numpy(), second=0)

    def test_call_and_inverse_4_axes_input_even_odd(self):
        """Tests whether the inverse method is indeed providing the inverse of the call on a 4 axes input along 2 axes even and odd dimension count."""
        
        # Initialize
        batch_size = 2; width = 4; height = 7; channel_count = 3
        permutation_layer = mfl.CheckerBoard(shape=[width, height], axes=[1,2])
        x = tf.random.uniform(shape=[batch_size, width, height, channel_count], dtype=tf.keras.backend.floatx())
        
        # Observe
        y_hat = permutation_layer(x=x)
        x_hat = permutation_layer.invert(y_hat=y_hat)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x.shape), tuple2=tuple(x_hat.shape))
        self.assertEqual(first=tf.reduce_sum((x-x_hat)**2).numpy(), second=0)

    def test_earger_vs_graph_call_equivalence(self):
        """Tests whether the call method of Checkerboad is equivalent in graphs mode and in regular eager mode."""

        # Initialize
        batch_size = 2; width = 4; height = 7; channel_count = 3
        layer = mfl.CheckerBoard(shape=[width, height], axes=[1,2])
        x = tf.random.uniform(shape=[batch_size, width, height, channel_count], dtype=tf.keras.backend.floatx())
        
        # Observe
        x_target = layer(x=x)

        # Graph mode
        @tf.function
        def graph_call(x): return layer(x=x)
        x_observed = graph_call(x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_max((x_observed-x_target)**2).numpy(), second=0)

    def test_earger_vs_graph_determinant_equivalence(self):
        """Tests whether the call method of Checkerboard is equivalent in graphs mode and in regular eager mode."""

        # Initialize
        batch_size = 2; width = 4; height = 7; channel_count = 3
        layer = mfl.CheckerBoard(shape=[width, height], axes=[1,2])
        x = tf.random.uniform(shape=[batch_size, width, height, channel_count], dtype=tf.keras.backend.floatx())
        
        # Observe
        det_target = layer.compute_jacobian_determinant(x=x)
        
        # Graph mode
        @tf.function
        def graph_determinant(x): return layer.compute_jacobian_determinant(x=x)

        det_observed = graph_determinant(x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(det_target.shape), tuple2=tuple(det_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_max((det_observed-det_target)**2).numpy(), second=0)

    def test_earger_vs_graph_round_trip_equivalence(self):
        """Tests whether the call method of Reflection is equivalent in graphs mode and in regular eager mode."""

        
        # Initialize
        batch_size = 2; width = 4; height = 7; channel_count = 3
        layer = mfl.CheckerBoard(shape=[width, height], axes=[1,2])
        x = tf.random.uniform(shape=[batch_size, width, height, channel_count], dtype=tf.keras.backend.floatx())
        
        # Observe
        x_target = layer.invert(y_hat=layer(x))
        
        # Graph mode
        @tf.function
        def graph_round_trip(x): return layer.invert(y_hat=layer(x=x))

        x_observed = graph_round_trip(x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_max((x_observed-x_target)**2).numpy(), second=0)

    def test_build_layer(self):
        """Tests whether the build method of CheckerBoard works as intended."""

        # Initialize
        batch_size = 12; width = 4; height = 7; channel_count = 3
        layer = mfl.CheckerBoard(shape=[width, height], axes=[1,2])

        # Build

        layer.build(input_shape=[batch_size, width, height, channel_count])

        # Evaluate
        self.assertEqual(first=layer.built, second=True)
        self.assertEqual(first=hasattr(layer, '_forward_permutation_'), second=True)
        self.assertEqual(first=hasattr(layer, '_inverse_permutation_'), second=True)

    def test_build_model(self):
        """Tests whether the build method of CheckerBoard works as intended inside a FlowModel."""

        # Initialize
        batch_size = 12; width = 4; height = 7; channel_count = 3
        layer = mfl.CheckerBoard(shape=[width, height], axes=[1,2])
        model = mfl.FlowModel(flow_layers=[layer])
        
        # Build
        model.build(input_shape=[batch_size, width, height, channel_count])

        # Evaluate
        self.assertEqual(first=model.built, second=True)
        self.assertEqual(first=model.flow_layers[0].built, second=True)
        self.assertEqual(first=hasattr(model.flow_layers[0], '_forward_permutation_'), second=True)
        self.assertEqual(first=hasattr(model.flow_layers[0], '_inverse_permutation_'), second=True)

class TestReflection(unittest.TestCase):

    def test_call_2_axes_input_along_1_axis(self):
        """Tests whether the call method works on a 2_axes input along 1 axis."""
        
        # Initialize
        dimension_count = 3
        reflection_layer = mfl.Reflection(shape=[dimension_count], axes=[1], reflection_count=2)
        reflection_layer.build(input_shape=None) # Input shape is ignored since it was provided in init, now in build, we create the layer's variables
        reflection_normals = tf.math.l2_normalize(tf.Variable([[1,1,0],[0,0,-1]], dtype=tf.keras.backend.floatx()), axis=1)
        reflection_layer._reflection_normals_.assign(reflection_normals) # For predictability
        x = tf.constant([[1,2,3],[4,5,6]], dtype=tf.keras.backend.floatx())
        x_target = tf.constant([[-2,-1,-3],[-5,-4,-6]], dtype=tf.keras.backend.floatx())

        # Observe
        x_observed = reflection_layer(x=x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_call_3_axes_input_along_1_axis(self):
        """Tests whether the call method works on a 3 axes input along 1 axis."""
        
        # Initialize
        dimension_count = 3
        reflection_layer = mfl.Reflection(shape=[dimension_count], axes=[1], reflection_count=2)
        reflection_layer.build(input_shape=None) # Input shape is ignored since it was provided in init, now in build, we create the layer's variables
        reflection_normals = tf.math.l2_normalize(tf.Variable([[1,1,0],[0,0,-1]], dtype=tf.keras.backend.floatx()), axis=1)
        reflection_layer._reflection_normals_.assign(reflection_normals) # For predictability
        x = tf.constant([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]], dtype=tf.keras.backend.floatx())
        x = tf.transpose(x, [0,2,1])
        x_target = tf.constant([[[-2,-1,-3],[-5,-4,-6]], [[-8,-7,-9],[-11,-10,-12]]], dtype=tf.keras.backend.floatx())
        x_target = tf.transpose(x_target, [0,2,1])

        # Observe
        x_observed = reflection_layer(x=x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_call_3_axes_input_along_2_axis(self):
        """Tests whether the call method works on a 3 axes input along 2 axes."""
        
        # Initialize
        reflection_layer = mfl.Reflection(shape=[2,3], axes=[1,2], reflection_count=2)
        reflection_layer.build(input_shape=None) # Input shape is ignored since it was provided in init, now in build, we create the layer's variables
        reflection_normals = tf.math.l2_normalize(tf.Variable([[1,1,0,0,0,0],[0,0,-1,0,0,0]], dtype=tf.keras.backend.floatx()), axis=1)
        reflection_layer._reflection_normals_.assign(reflection_normals) # For predictability
        x = tf.constant([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]], dtype=tf.keras.backend.floatx())
        x_target = tf.constant([[[-2,-1,-3],[4,5,6]], [[-8,-7,-9],[10,11,12]]], dtype=tf.keras.backend.floatx())
        
        # Observe
        x_observed = reflection_layer(x=x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_sum((x_target-x_observed)**2).numpy(), second=0)

    def test_call_and_inverse_2_axes_input_along_1_axis(self):
        """Tests whether the inverse method is indeed providing the inverse of the call on a 2_axes input along 1 axis."""
        
        # Initialize
        dimension_count = 3
        reflection_layer = mfl.Reflection(shape=[dimension_count], axes=[1], reflection_count=2)
        x = tf.constant([[1,2,3],[4,5,6]], dtype=tf.keras.backend.floatx())

        # Observe
        y_hat = reflection_layer(x=x)
        x_hat = reflection_layer.invert(y_hat=y_hat)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x.shape), tuple2=tuple(x_hat.shape))
        self.assertAlmostEqual(first=tf.reduce_sum((x-x_hat)**2).numpy(), second=0)

    def test_call_and_inverse_3_axes_input_along_2_axes(self):
        """Tests whether the inverse method is indeed providing the inverse of the call on a 3_axes input along two axes."""
        
        # Initialize
        batch_size = 2; width = 4; height = 5
        reflection_layer = mfl.Reflection(shape=[width, height], axes=[1,2], reflection_count=2)
        x = tf.random.uniform([batch_size, width, height], dtype=tf.keras.backend.floatx())

        # Observe
        y_hat = reflection_layer(x=x)
        x_hat = reflection_layer.invert(y_hat=y_hat)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x.shape), tuple2=tuple(x_hat.shape))
        self.assertAlmostEqual(first=tf.reduce_sum((x-x_hat)**2).numpy(), second=0)

    def test_load_and_save_4D_input_along_2_axes(self):
        """Tests whether the model provides the same shuffling after persistent storage."""

        # Initialize
        width = 4; height = 5; dimension_count = 3
        reflection_layer = mfl.Reflection(shape=[width, height], axes=[1,2], reflection_count=10)
        model = mfl.FlowModel(flow_layers=[reflection_layer])
        
        x = tf.random.uniform(shape=[10, width, height, dimension_count], dtype=tf.keras.backend.floatx())
        
        # Observe first
        y_hat_1 = model(x=x)
        
        # Save and load
        path = os.path.join(os.getcwd(), "temporary_model_directory_for_reflection_model_unit_test.weights.h5")
        model.save_weights(path)
        del model
        loaded_reflection_layer = mfl.Reflection(shape=[width, height], axes=[1,2], reflection_count=10)
        loaded_model = mfl.FlowModel(flow_layers=[loaded_reflection_layer])
        loaded_model.build(input_shape=x.shape)
        loaded_model.load_weights(path)
        os.remove(path)
    
        y_hat_2 = loaded_model(x=x)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y_hat_1.shape), tuple2=tuple(y_hat_2.shape))
        self.assertAlmostEqual(first=tf.reduce_sum((y_hat_1-y_hat_2)**2).numpy(), second=0)

    def test_call_triangular_jacobian_2_axes_input(self):
        """Tests whether the call method of Reflection produces a triangular jacobian on 2-axes inputs."""

        # Initialize
        layer = mfl.Reflection(shape=[7], axes=[1], reflection_count=1)
        x = tf.reshape(tf.range(14,dtype=tf.keras.backend.floatx()), [2,7])

        # Compute jacobian
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = layer(x)
            J = tape.jacobian(y,x)
        J = tf.reduce_sum(J, axis=2) # This axis is redundant, see section on batch jacobians in https://www.tensorflow.org/guide/advanced_autodiff#jacobians
        
        # Verify determinants
        x_observed = layer.compute_jacobian_determinant(x=x)
        for j in range(J.shape[0]):
            self.assertAlmostEqual(first=np.log(np.abs(np.linalg.det(J[j].numpy()))), second=x_observed[j].numpy(), places=5)

    def test_earger_vs_graph_call_equivalence(self):
        """Tests whether the call method of Reflection is equivalent in graphs mode and in regular eager mode."""

        # Initialize
        layer = mfl.Reflection(shape=[3,4], axes=[1,2], reflection_count=2)
        x = tf.reshape(tf.range(0,24*5,dtype=tf.keras.backend.floatx()), [2,3,4,5])

        # Observe
        x_target = layer(x)
        
        # Graph mode
        @tf.function
        def graph_call(x): return layer(x)

        x_observed = graph_call(x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_max((x_observed-x_target)**2).numpy(), second=0)

    def test_earger_vs_graph_determinant_equivalence(self):
        """Tests whether the call method of Reflection is equivalent in graphs mode and in regular eager mode."""

        # Initialize
        layer = mfl.Reflection(shape=[3,4], axes=[1,2], reflection_count=2)
        x = tf.reshape(tf.range(0,24*5,dtype=tf.keras.backend.floatx()), [2,3,4,5])

        # Observe
        det_target = layer.compute_jacobian_determinant(x=x)
        
        # Graph mode
        @tf.function
        def graph_determinant(x): return layer.compute_jacobian_determinant(x)

        det_observed = graph_determinant(x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(det_target.shape), tuple2=tuple(det_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_max((det_observed-det_target)**2).numpy(), second=0)

    def test_earger_vs_graph_round_trip_equivalence(self):
        """Tests whether the call method of Reflection is equivalent in graphs mode and in regular eager mode."""

        # Initialize
        layer = mfl.Reflection(shape=[3,4], axes=[1,2], reflection_count=2)
        x = tf.reshape(tf.range(0,24*5,dtype=tf.keras.backend.floatx()), [2,3,4,5])

        # Observe
        x_target = layer.invert(y_hat=layer(x))
        
        # Graph mode
        @tf.function
        def graph_round_trip(x): return layer.invert(y_hat=layer(x=x))

        x_observed = graph_round_trip(x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_max((x_observed-x_target)**2).numpy(), second=0)

    def test_build_layer(self):
        """Tests whether the build method of Reflection works as intended."""

        # Initialize
        batch_size = 2; width = 4; height = 5
        layer = mfl.Reflection(shape=[width, height], axes=[1,2], reflection_count=2)

        # Build
        layer.build(input_shape=[batch_size, width, height])

        # Evaluate
        self.assertEqual(first=layer.built, second=True)
        self.assertEqual(first=hasattr(layer, '_reflection_normals_'), second=True)

    def test_build_model(self):
        """Tests whether the build method of Reflection works as intended inside a FlowModel."""

        # Initialize
        batch_size = 2; width = 4; height = 5
        layer = mfl.Reflection(shape=[width, height], axes=[1,2], reflection_count=2)
        model = mfl.FlowModel(flow_layers=[layer])
        input_shape = (batch_size, width, height)

        # Build
        model.build(input_shape=input_shape)

        # Evaluate
        self.assertEqual(first=model.built, second=True)
        self.assertEqual(first=model.flow_layers[0].built, second=True)
        self.assertEqual(first=hasattr(model.flow_layers[0], '_reflection_normals_'), second=True)

class TestActivationNormalization(unittest.TestCase):
    def test_init_1_axis(self):
        """Tests whether an instance of ActivationNormalization can be created for a 1-axis input."""

        # Initialize
        mfl.ActivationNormalization(axes=[2], shape=[5])

    def test_init_2_axes(self):
        """Tests whether an instance of ActivationNormalization can be created for a 2-axes input."""

        # Initialize
        mfl.ActivationNormalization(axes=[1,2], shape=[2,5])
    
    def test_call_2_axes_input_along_axis_1(self):
        """Tests whether the call method of ActivatonNormalization can normalize 2_axes inputs along axis 1."""

        # Initialize
        layer = mfl.ActivationNormalization(shape=[3], axes=[1])
        
        # Generate data
        x = tf.random.normal([1024,3], dtype=tf.keras.backend.floatx(), mean=6.0, stddev=2.5)
        
        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=0))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=0))
        
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

    def test_call_3_axes_input_along_axes_1_2(self):
        """Tests whether the call method of ActivatonNormalization can normalize 3_axes inputs along axes 1 and 2."""

        # Initialize
        layer = mfl.ActivationNormalization(shape=[3,4], axes=[1,2])

        # Generate data
        x = tf.random.normal([8,3,4], dtype=tf.keras.backend.floatx(), mean=6.0, stddev=2.5)
        
        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=0))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=0))
        
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

    def test_call_3_axes_input_along_axes_1(self):
        """Tests whether the call method of ActivatonNormalization can normalize 3_axes inputs along axis 1."""

        # Initialize
        layer = mfl.ActivationNormalization(shape=[3], axes=[1])
        x = tf.reshape(tf.range(0,24,dtype=tf.keras.backend.floatx()), [2,3,4])
        
        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=[0,2]))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=[0,2]))
        
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
        x = tf.reshape(tf.range(0,24*5,dtype=tf.keras.backend.floatx()), [2,3,4,5])

        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=[0,2,3]))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=[0,2,3]))

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
        x = tf.reshape(tf.range(0,24*5,dtype=tf.keras.backend.floatx()), [2,3,4,5])

        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=[0,3]))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=[0,3]))

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
        x = tf.reshape(tf.range(0,24*5,dtype=tf.keras.backend.floatx()), [2,3,4,5])

        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=[0,2]))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=[0,2]))

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

    def test_invert_4D_input_along_axes_1_3(self):
        """Tests whether the call and invert method of ActivatonNormalization can normalize and un-normalize 4D inputs along axes 1 and 3."""

        # Initialize
        layer = mfl.ActivationNormalization(shape=[3,5], axes=[1,3])
        x = tf.reshape(tf.range(0,24*5,dtype=tf.keras.backend.floatx()), [2,3,4,5])

        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=[0,2]))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=[0,2]))

        # Target
        x_target = x

        # Observe
        y_hat = layer(x)
        x_observed = layer.invert(y_hat=y_hat)

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x.shape))
        self.assertAlmostEqual(first=tf.reduce_mean((x_observed-x_target)**2).numpy(), second=0)

    def test_compute_jacobian_determinant_2_axes_axis_1(self):
        """Tests whether the activation normalization layer can compute the jacobian determinant on 2_axes inputs"""
                
        # Initialize
        layer = mfl.ActivationNormalization(shape=[3], axes=[1])
        x = tf.reshape(tf.range(0,1,delta=1/24,dtype=tf.keras.backend.floatx())**2, [8,3])
        
        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=0))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=0))
        
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
            self.assertAlmostEqual(first=np.log(np.prod(np.diagonal(J[j].numpy()))), second=x_observed[j].numpy(), places=5)

    def test_compute_jacobian_determinant_3_axes_axis_1(self):
        """Tests whether the activation normalization layer can compute the jacobian determinant on 3_axes inputs"""
                
        # Initialize
        layer = mfl.ActivationNormalization(shape=[3], axes=[1])
        x = tf.reshape(tf.range(0,1,delta=1/60,dtype=tf.keras.backend.floatx())**2, [5,3,4])
        
        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=[0,2]))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=[0,2]))

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

    def test_compute_jacobian_determinant_3_axes_axis_2(self):
        """Tests whether the activation normalization layer can compute the jacobian determinant on 3_axes inputs"""
                
        # Initialize
        layer = mfl.ActivationNormalization(shape=[4], axes=[2])
        x = tf.reshape(tf.range(0,1,delta=1/60,dtype=tf.keras.backend.floatx())**2, [5,3,4])

        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=[0,1]))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=[0,1]))
        
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

    def test_compute_jacobian_determinant_3_axes_axes_1_2(self):
        """Tests whether the activation normalization layer can compute the jacobian determinant on 3_axes inputs"""
                
        # Initialize
        layer = mfl.ActivationNormalization(shape=[3,4], axes=[1,2])
        x = tf.reshape(tf.range(0,1,delta=1/60,dtype=tf.keras.backend.floatx())**2, [5,3,4])
        
        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=0))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=0))
        
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

    def test_compute_jacobian_determinant_4_axes_axes_1_2(self):
        """Tests whether the activation normalization layer can compute the jacobian determinant on 4D inputs"""
                
        # Initialize
        layer = mfl.ActivationNormalization(shape=[3,4], axes=[1,2])
        x = tf.reshape(tf.range(0,1,delta=1/120,dtype=tf.keras.backend.floatx())**2, [5,3,4,2])
        
        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=[0,3]))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=[0,3]))
        
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
            self.assertAlmostEqual(first=np.log(np.prod(np.diagonal(J[j].numpy()))), second=x_observed[j].numpy(), places=5)

    def test_earger_vs_graph_call_equivalence(self):
        """Tests whether the call method of ActivationNormalization is equivalent in graphs mode and in regular eager mode."""

        # Initialize
        layer = mfl.ActivationNormalization(shape=[3,4], axes=[1,2])
        x = tf.reshape(tf.range(0,24*5,dtype=tf.keras.backend.floatx()), [2,3,4,5])

        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=[0,3]))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=[0,3]))

        # Observe
        x_target = layer(x)
        
        # Graph mode
        layer = mfl.ActivationNormalization(shape=[3,4], axes=[1,2])
        
        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=[0,3]))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=[0,3]))

        @tf.function
        def graph_call(x): return layer(x)

        x_observed = graph_call(x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_max((x_observed-x_target)**2).numpy(), second=0)

    def test_earger_vs_graph_determinant_equivalence(self):
        """Tests whether the call method of ActivationNormalization is equivalent in graphs mode and in regular eager mode."""

        # Initialize
        layer = mfl.ActivationNormalization(shape=[3,4], axes=[1,2])
        x = tf.reshape(tf.range(0,24*5,dtype=tf.keras.backend.floatx()), [2,3,4,5])

        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=[0,3]))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=[0,3]))

        # Observe
        det_target = layer.compute_jacobian_determinant(x=x)
        
        # Graph mode
        layer = mfl.ActivationNormalization(shape=[3,4], axes=[1,2])
        
        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=[0,3]))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=[0,3]))

        @tf.function
        def graph_determinant(x): return layer.compute_jacobian_determinant(x=x)

        det_observed = graph_determinant(x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(det_target.shape), tuple2=tuple(det_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_max((det_observed-det_target)**2).numpy(), second=0)

    def test_earger_vs_graph_round_trip_equivalence(self):
        """Tests whether the call method of ActivationNormalization is equivalent in graphs mode and in regular eager mode."""

        # Initialize
        layer = mfl.ActivationNormalization(shape=[3,4], axes=[1,2])
        x = tf.reshape(tf.range(0,24*5,dtype=tf.keras.backend.floatx()), [2,3,4,5])

        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=[0,3]))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=[0,3]))

        # Observe
        x_target = layer.invert(y_hat=layer(x))
        
        # Graph mode
        layer = mfl.ActivationNormalization(shape=[3,4], axes=[1,2])
        
        # Set the layer's variables to simulate standard normalization
        layer.variables[0].assign(tf.math.reduce_mean(x, axis=[0,3]))  # Set mean to observed mean
        layer.variables[1].assign(tf.math.reduce_std(x, axis=[0,3]))

        @tf.function
        def graph_round_trip(x): return layer.invert(y_hat=layer(x=x))

        x_observed = graph_round_trip(x)
        
        # Evaluate
        self.assertTupleEqual(tuple1=tuple(x_target.shape), tuple2=tuple(x_observed.shape))
        self.assertAlmostEqual(first=tf.reduce_max((x_observed-x_target)**2).numpy(), second=0)

    def test_build_layer(self):
        """Tests whether the build method of ActivationNormalization works as intended."""

        # Initialize
        batch_size, width, height, channel_count = 2,3,4,5
        layer = mfl.ActivationNormalization(shape=[width, height], axes=[1,2])
        
        # Build
        layer.build(input_shape=[batch_size, width, height, channel_count])

        # Evaluate
        self.assertEqual(first=layer.built, second=True)
        self.assertEqual(first=hasattr(layer, '_location_'), second=True)
        self.assertEqual(first=hasattr(layer, '_scale_'), second=True)

    def test_build_model(self):
        """Tests whether the build method of ActivationNormalization works as intended inside a FlowModel."""

        # Initialize
        # Initialize
        batch_size, width, height, channel_count = 2,3,4,5
        layer = mfl.ActivationNormalization(shape=[width, height], axes=[1,2])
        model = mfl.FlowModel(flow_layers=[layer])
        input_shape = (batch_size, width, height, channel_count)

        # Build
        model.build(input_shape=input_shape)

        # Evaluate
        self.assertEqual(first=model.built, second=True)
        self.assertEqual(first=model.flow_layers[0].built, second=True)
        self.assertEqual(first=hasattr(model.flow_layers[0], '_location_'), second=True)
        self.assertEqual(first=hasattr(model.flow_layers[0], '_scale_'), second=True)

class TestFlowModel(unittest.TestCase):

    def create_model(stage_count: int, M, N) -> mfl.FlowModel:
        
        # Prepare a sequence of layers
        layers = [None] * (6*stage_count+1)
        
        # Start with a normalization layer
        layers[0] = mfl.ActivationNormalization(axes=[1], shape=[N])
        
        for i in range(stage_count):
            
            # Reflection layer, flips point cloud about a hyperplane
            layers[6*i+1] = mfl.Reflection(axes=[1], shape=[N], reflection_count=1)
            
            # Coupling block, stretches some regions of space more than others
            # Couple first half of dimensions, 
            mask_1 = mms.CheckerBoard(axes=[1], shape=[N]) # We need to mask out half of the dimensions
            compute_coupling_parameters_1 = tf.keras.Sequential(layers=[tf.keras.layers.Dense(units=4*N, activation='relu'), tf.keras.layers.Dense(units=N, activation=None)]) # We use half of the dimensions to compute coupling parameters to be added to the other half of the dimension
            layers[6*i+2] = mfl.AdditiveCoupling(axes=[1], shape=[N], compute_coupling_parameters=compute_coupling_parameters_1, mask=mask_1) # This layer applies the mask, computes the coupling parameters and adds them to the other half of dimensions
            layers[6*i+3] = mfl.CheckerBoard(axes=[1], shape=[N]) # This layer permutes the dimensions in line with the previously chosen mask

            # Couple second half of dimensions
            compute_coupling_parameters_2 = tf.keras.Sequential(layers=[tf.keras.layers.Dense(units=4*N, activation='relu'), tf.keras.layers.Dense(units=N, activation=None)])
            mask_2 = mms.CheckerBoard(axes=[1], shape=[N])
            layers[6*i+4] = mfl.AdditiveCoupling(axes=[1], shape=[N], compute_coupling_parameters=compute_coupling_parameters_2, mask=mask_2)
            layers[6*i+5] = mfl.CheckerBoard(axes=[1], shape=[N])

            # End with another normalization layer
            layers[6*i+6] = mfl.ActivationNormalization(axes=[1], shape=[N])
        
        # Construct the network
        flow_model = mfl.FlowModel(layers) # Essentially a keras Sequential model with the added benefit of computing the jacbian determinant and inverse of the overall flow
        flow_model.build(input_shape=[M, N]) # Pass a symbolic tensor through the model to help the tensorflow backend register all computations

        # Outputs
        return flow_model

    def test_compute_jacobian_determinant_2_axes_axis_1(self):
        """Tests whether the a multi-layerd flow model can compute the jacobian determinant on 2-axes inputs along axis 1"""
         
        # Initialize
        # Reproducability
        gtds.reset_random_number_generators(seed=123)

        batch_size = 8; dimensionality = 3
        flow_model = TestFlowModel.create_model(stage_count=2, M=batch_size, N=dimensionality)
        x = tf.reshape(tf.range(0,1,delta=1/(batch_size*dimensionality),dtype=tf.keras.backend.floatx())**2, [batch_size, dimensionality])
        
        # Compute jacobian
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = flow_model(x)
            J = tape.jacobian(y,x)
        J = tf.reduce_sum(J, axis=2) # This axis is redundant, see section on batch jacobians in https://www.tensorflow.org/guide/advanced_autodiff#jacobians
        
        # Verify determinants
        x_observed = flow_model.compute_jacobian_determinant(x=x)
        for j in range(J.shape[0]):
            self.assertAlmostEqual(first=np.log(np.abs(np.linalg.det(J[j].numpy()))), second=x_observed[j].numpy(), places=5)

    def test_call_triangular_jacobian_2_axes_input_checker_board_1_axis_mask(self):
        """Tests whether the call method of AdditiveCoupling produces a triangular jacobian on 2-axes inputs 
        with a checker board 1 axis mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.CheckerBoard(axes=[1], shape=[7])
        layer = mfl.AdditiveCoupling(axes=[1], shape=[7], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14,dtype=tf.keras.backend.floatx()), [2,7])

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
    '''
    def test_call_triangular_jacobian_3_axes_input_checker_board_2_axes_mask(self):
        """Tests whether the call method of AdditiveCoupling produces a triangular jacobian on 3-axes inputs 
        with checker board 2_axes mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.CheckerBoard(axes=[1,2], shape=[2,7])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[2,7], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14*3,dtype=tf.keras.backend.floatx()), [3,2,7])

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

    def test_call_triangular_jacobian_4_axes_input_checker_board_2_axes_mask(self):
        """Tests whether the call method of AdditiveCoupling produces a triangular jacobian on 4-axes inputs 
        with checker board 2_axes mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=1, kernel_size=[2,2], padding='same')
        mask = mms.CheckerBoard(axes=[1,2], shape=[5,6])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[5,6], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(2*5*6,dtype=tf.keras.backend.floatx()), [2,5,6,1]) # Shape == [batch size, height, width, channel count]

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

    def test_compute_jacobian_determinant_2_axes_input_chcker_board_1_axis_mask(self):
        """Tests whether the compute_jacobian_determinant method of AdditiveCoupling correctly computes the determinant on 
        2-axes inputs with a checker board 1 axis mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.CheckerBoard(axes=[1], shape=[7])
        layer = mfl.AdditiveCoupling(axes=[1], shape=[7], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14,dtype=tf.keras.backend.floatx()), [2,7])

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

    def test_compute_jacobian_determinant_3_axes_input_checker_board_2_axes_mask(self):
        """Tests whether the compute_jacobian_determinant method of AdditiveCoupling correctly computes the determinant on 
        3-axes inputs with a checker board 2 axes mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[tf.newaxis,:]),
            tf.keras.layers.Dense(units=7),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x))]) 
        mask = mms.CheckerBoard(axes=[1,2], shape=[2,7])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[2,7], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(14*3,dtype=tf.keras.backend.floatx()), [3,2,7])

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

    def test_compute_jacobian_determinant_4_axes_input_checker_board_2_axes_mask(self):
        """Tests whether the call method of AdditiveCoupling produces a triangular jacobian on 4-axes inputs 
        with checker board 2 axes mask."""

        # Initialize
        compute_coupling_parameters = tf.keras.layers.Conv2D(filters=1, kernel_size=[2,2], padding='same')
        mask = mms.CheckerBoard(axes=[1,2], shape=[5,6])
        layer = mfl.AdditiveCoupling(axes=[1,2], shape=[5,6], compute_coupling_parameters=compute_coupling_parameters, mask=mask)
        x = tf.reshape(tf.range(2*5*6,dtype=tf.keras.backend.floatx()), [2,5,6,1]) # Shape == [batch size, height, width, channel count]

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
        def build_flow_model():
            """Function to create a fresh, unbuilt model architecture."""
            compute_coupling_parameters = tf.keras.layers.Conv2D(filters=1, kernel_size=[2,2], padding='same')
            mask = mms.CheckerBoard(axes=[1,2], shape=[5,6])
            model = mfl.FlowModel()
            model.add(mfl.AdditiveCoupling(axes=[1,2], shape=[5,6], compute_coupling_parameters=compute_coupling_parameters, mask=mask))
            return model

        model = build_flow_model()
        x = tf.reshape(tf.range(2*5*6,dtype=tf.keras.backend.floatx()), [2,5,6,1]) 

        # Build the model explicitly before saving weights to ensure all weights exist
        model.build(input_shape=x.shape) 

        # Observe 
        y_hat_1 = model(x=x)
        path = os.path.join(os.getcwd(), "temporary_model_directory_for_additive_coupling_layer_unit_test.weights.h5")
        model.save_weights(path)

        # Delete everything and clear session
        del model
        #tf.keras.backend.clear_session() 

        # Initialize the second model using the *same* build function
        loaded_model = build_flow_model()

        # Build the loaded model with the correct shape 
        loaded_model.build(input_shape=x.shape)

        # Load the weights (will now match the topology exactly)
        loaded_model.load_weights(path)
        os.remove(path)
            
        # Observe second - this should now match y_hat_1
        y_hat_2 = loaded_model(x=x)
        

        # Evaluate
        self.assertTupleEqual(tuple1=tuple(y_hat_1.shape), tuple2=tuple(y_hat_2.shape))
        self.assertEqual(first=tf.reduce_sum((y_hat_1-y_hat_2)**2).numpy(), second=0)
    '''