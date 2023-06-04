import numpy as np
import tensorflow as tf
from typing import Any, Tuple, List, Callable
from abc import ABC
from gyoza.utilities import tensors as utt

class BasicFullyConnectedNet(tf.keras.Model):
    def __init__(self, latent_channel_count:int, output_channel_count:int, depth: int, use_tanh:bool=False, use_batch_normalization:bool=False):
        
        # Super
        super(BasicFullyConnectedNet, self).__init__()
        
        # Compile list of layers
        channel_counts = [latent_channel_count] * depth + [output_channel_count]
        layers = []
        for d in range(depth + 1):
            layers.append(tf.keras.layers.Dense(units=channel_counts[d]))
            if use_batch_normalization: layers.append(tf.keras.layers.BatchNormalization(channel_counts[d]), axis=-1)
            if d < depth or not use_tanh: layers.append(tf.keras.layers.LeakyReLU())
        
        if use_tanh: layers.append(tf.keras.layers.Tanh())
        
        # Attributes
        self.sequential = tf.keras.Sequential(layers)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Applies the forward operation to x.
        
        Inputs:
        - x: tensor that should be passed through the network.
        
        Outputs:
        - y_hat: prediction."""
        
        # Predict
        y_hat = self.sequential(x)

        # Outputs:
        return y_hat

class FlowLayer(tf.keras.Model, ABC):
    """Defines methods for flow layers."""

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Executes the operation of this layer in the forward direction.
        
        Inputs:
        - x: the data to tranform.
        
        Outputs:
        - y_hat: the transformed version of x."""
        
        raise NotImplemented()

    def invert(self, x: tf.Tensor) -> tf.Tensor:
        """Executes the operation of this layer in the inverse direction.
        
        Inputs:
        - x: the data to tranform.
        
        Outputs:
        - y_hat: the transformed version of x."""
        
        raise NotImplemented()

    def compute_logarithmic_determinant(self, x: tf.Tensor) -> tf.Tensor:
        """Computes the logarithmic determinant of this layer's forward operation.
        
        Inputs:
        - x: the data at which the determinant shall be computed.

        Outputs:
        - logarithmic_determinant: a measure of how much this layer contracts or dilates space at the point x. Shape == [instance]"""
        
        raise NotImplemented()

class Mask(FlowLayer, ABC):
    pass

class CheckerBoardMask(Mask):
    """A mask that applies a checkerboard pattern to its input."""

    def __init__(self, axes: List[int]) -> None:
        """Constructor for this class.
        
        Inputs:
        - axes: The axes along which the checker board pattern shall be applied. Support one or two axes."""

        # Input validity
        assert len(axes) <= 2, "There must be at most two axes along which the checker board pattern shall be applied."

        self.__axes__ = axes

    def call(self, x:tf. Tensor) -> tf.Tensor:

        # Case 1-dimensional
        if len(self.__axes__) == 1:

            # Initialize
            mask = tf.zeros_like(input=x)

            # Move axis 
            mask = utt.move_axis(x=mask, from_index=self.__axes__[0], to_index= 0)

            # Set ones
            mask[::2,...] = 1

            # Revert axis
            mask = utt.move_axis(x=mask, from_index=0, to_index=self.__axes__[0])
            


        # Case 2-dimensional
        # Iterate axes
        



class Shuffle(FlowLayer):
    """Shuffles inputs along a given axis. The permutation used for shuffling is randomly chosen 
    once during initialization. Thereafter it is saved as a non-trainable tensorflow.Variable in a private attribute.
    Shuffling is thus deterministic and supports loading and saving."""

    def __init__(self, channel_count, channel_axis: int = -1):
        """Initializes the instance of this class.
        
        Inputs:
        - channel_count: number of channels that should be shuffled.
        - channel_axis: axis of shuffling."""

        # Super
        super(Shuffle, self).__init__()
        
        # Attributes
        self.__channel_count__ = channel_count
        self.__axis__ = channel_axis
        
        permutation = tf.random.shuffle(tf.range(channel_count))
        self.__forward_permutation__ = tf.Variable(permutation, trainable=False)
        self.__inverse_permutation__ = tf.Variable(tf.argsort(permutation), trainable=False)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        
        # Shuffle
        y_hat = tf.gather(x, self.permutation, axis=self.__axis__)
        
        # Outputs
        return y_hat
    
    def invert(self, x: tf.Tensor) -> tf.Tensor:
        
        # Shuffle
        y_hat = tf.gather(x, self.__inverse_permutation__, axis=self.__axis__)
        
        # Outputs
        return y_hat
    
    def compute_logarithmic_determinant(self, x: tf.Tensor) -> tf.Tensor:

        # Outputs
        return 0

class CouplingLayer(FlowLayer, ABC):
    """This layer splits its input x in the middle into two parts x_1, x_2 along the channel axis. For uneven channel counts one
    half is one unit larger than the other half. Then the following update rule is applied:\n
    
    y_hat_1 = x_1\n
    y_hat_2 = g(x_2, m(x_1))\n

    where y_hat_1 and y_hat_2 and concatenated along the channel axis to give the overall output y_hat. The function g(a,b) is the 
    coupling law that has to be invertible with respect to its first argument given the second, e.g. g(a,b) = a + b or g(a,b) = a*b 
    for b != 0. The function m(a) is unconstrained, e.g. an artificial neural network. The implementation of g(a,b) and m(a) 
    distinguishes coupling layers. 
    
    References:
    - L. Dinh, D. Krueger & Y. Bengio (2015). NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION.
    """

    def __init__(self, m: Callable, channel_count: int, channel_axis: int = -1):
        """Constructor for this class.

        Inputs:
        - m: the function that shall be used to map the first half of the input to call() to weights used to transform the second 
            half of that x in g(). Its inputs shall be a tensor with channel axis at channel_axis and channel count equal to 
            channel_count // 2. Its output shall be a tensor or list of tensors that can be used to transform the second half of x
            inside g(). The output requirement on m thus depends on the subclass specific implementation for g(). 
        - channel_count: the total number of channels of the input to this layer.
        - channel_axis: the axis along which coupling shall be executed."""

        # Attributes
        self.__m__ = m
        self.channel_count = channel_count
        self.channel_axis = channel_axis

    def call(self, x: tf.Tensor) -> tf.Tensor:
        
        # Split
        partition_point = self.channel_count // 2
        x_1, x_2 = tf.split(x, num_or_size_splits=[partition_point, self.channel_count - partition_point], axis=self.channel_axis)
         
        # Couple
        y_hat_1 = x_1
        y_hat_2 = self.__g__(a=x_2, b=self.__m__(a=x_1))

        # Concatenate
        y_hat = tf.concat([y_hat_1, y_hat_2], axis=self.channel_axis)

        # Outputs
        return y_hat
    
    def __g__(self, a: tf.Tensor, b: tf.Tensor or List[tf.Tensor]) -> tf.Tensor:
        """This function implements an invertible coupling law for inputs a and b. It is invertible w.r.t. a, given b.
        
        Inputs:
        - a: Tensor of arbitrary shape whose channel axis is the same as self.channel_axis. This Tensor is supposed to be the
            second half of x in call() and shall be coupled with b.
        - b: Tensor or list of tensors whose shape is the same as that of a. These tensors shall be used to transform a.
        
        Outputs:
        - y_hat: The coupled tensor of same shape as a."""

        raise NotImplementedError()
    
    def __inverse_g__(self, a: tf.Tensor, b: tf.Tensor or List[tf.Tensor]) -> tf.Tensor:
        """This function implements the inverse coupling law for __g__(). 
        
        Inputs:
        - a: Tensor of arbitrary shape whose channel axis is the same as self.channel_axis. This Tensor is supposed to be the
            second half of x in call() and in invert() and it shall be decoupled from b.
        - b: Tensor or list of tensors whose shape is the same as that of a. These tensors shall be used to decouple a.
        
        Outputs:
        - y_hat: The decoupled tensor of same shape as a."""

        raise NotImplementedError()
    
    def invert(self, x: tf.Tensor) -> tf.Tensor:
        
        # Split
        partition_point = self.channel_count // 2
        x_1, x_2 = tf.split(x, num_or_size_splits=[partition_point, self.channel_count - partition_point], axis=self.channel_axis)
         
        # Decouple
        y_hat_1 = x_1
        y_hat_2 = self.__inverse_g__(a=x_2, b=self.__m__(a=x_1))

        # Concatenate
        y_hat = tf.concat([y_hat_1, y_hat_2], axis=self.channel_axis)

        # Outputs
        return y_hat
    
class AdditiveCouplingLayer(CouplingLayer):

    def __g__(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        
        # Couple
        y_hat = a + b

        # Outputs
        return y_hat
    
    def __inverse_g__(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        
        # Decouple
        y_hat = a - b

        # Outputs
        return y_hat
    
    def compute_logarithmic_determinant(self, x: tf.Tensor) -> tf.Tensor:
        
        # Outputs
        return 0

class ActivationNormalization(FlowLayer):
    """A trainable channel-wise location and scale transform of the data. Is initialized to produce zero mean and unit variance."""

    def __init__(self, channel_count: int, channel_axis: int = -1):
        """Initializer for this class.
        
        Inputs:
        - channel_count: number of channels for which the transform shall be executed.
        - channel_axis: the axis along which the transform shall be executed. Each entry along this axis will have 
            its own transformation applied along all other axis."""

        # Super
        super().__init__()
        
        # Attributes
        self.__location__ = tf.Variable(tf.zeros(channel_count), trainable=True)
        self.__scale__ = tf.Variable(tf.ones(channel_count), trainable=True)
        self.__is_initialized__ = False
        self.__channel_axis__ = channel_axis

    def __initialize__(self, x: tf.Tensor) -> None:
        """This method shall be used to lazily initialize the variables of self.
        
        Inputs:
        - x: data that is propagated through this layer.
        """

        # Move the channel axis to the end
        new_order = list(range(len(x.shape)))
        a = new_order[self.__channel_axis__]; del new_order[self.__channel_axis__]; new_order.append(a)
        x_new = tf.stop_gradient(tf.permute(x, new_order)) # Shape == [other axes , channel count]

        # Flatten per channel
        x_new = tf.stop_gradient(tf.reshape(x_new, [np.multiply(new_order[:-1]), -1])) # Shape == [product of all other axes, channel count]

        # Compute mean and variance channel-wise
        mean = tf.stop_gradient(tf.reduce_mean(x_new, axis=0)) # Shape == [channel count] 
        variance = tf.stop_gradient(tf.math.reduce_variance(x_new, axis=0)) # Shape == [channel count]
        
        # Update attributes
        self.__location__.assign(mean)
        self.__scale__.assign(variance)

    def __reshape_variables__(self, x: tf.Tensor) -> Tuple[tf.Variable, tf.Variable]:
        """Formats the variables need for this layer such that they are compatible with the shape of the data passed through the layer.
        
        Inputs:
        - x: the data to be passed through the layer. It's shape must have as many channels along the channel axis as specified during initialization.
        
        Outputs:
        - location: the reshaped location attribute of self.
        - scale: the reshaped scale attribute of self. """

        # Cast variables to shape compatible with x
        new_shape = [1] * len(x.shape); new_shape[self.__channel_axis__] = len(self.__location__)
        location = tf.reshape(self.__location__, new_shape) # Shape has ones along every axis except for the channel axis where it is equal to channel count.
        scale = tf.reshape(self.__scale__, new_shape) # Shape equal that of location 
        
        # Outputs
        return location, scale

    def __scale_to_non_zero__(self) -> None:
        """Corrects the scale attribute where it is equal to zero by adding a constant epsilon. This is useful to prevent scaling by 0 which is not invertible."""
        
        # Correct scale where it is equal to zero to prevent division by zero
        epsilon = tf.stop_gradient(tf.constant(1e-6 * (self.__scale__.numpy() == 0), dtype=self.__scale__.dtype)) 
        self.__scale__.assing(self.__scale__ + epsilon)

    def __prepare_variables_for_computation__(self, x:tf.Tensor) -> Tuple[tf.Variable, tf.Variable]:
        """Prepares the variables for computation with data. This involves adjusting the scale to be non-zero and ensuring variable shapes are compatible with the data.
        
        Inputs:
        - x: example data. It's shape must agree with expectation of self.__reshape_variables__.

        Outputs:
        - location, scale: the prepared location and scale attributes."""

        # Preparations
        self.__scale_to_non_zero__()
        location, scale = self.__reshape_variables__(x=x)

        # Outputs
        return location, scale

    def call(self, x: tf.Tensor) -> tf.Tensor:

        # Ensure initialization of variables
        if not self.__is_initialized__: self.__initialize__(x=x)

        # Transform
        scale, location = self.__prepare_variables_for_computation__(x=x)
        y_hat = (x-location) / (scale) 

        # Outputs
        return y_hat
        
    def invert(self, x: tf.Tensor) -> tf.Tensor:

        # Transform
        scale, location = self.__prepare_variables_for_computation__(x=x)
        y_hat = scale * x + location

        # Outputs
        return y_hat
           
    def compute_logarithmic_determinant(self, x: tf.Tensor) -> tf.Tensor:

        # Count elements per instance (ignoring channels)
        instance_count = x.shape[0]
        element_shape = x.shape; del element_shape[self.__channel_axis__]; del element_shape[0] # Shape ignoring instance and channel axes
        element_count = tf.math.reduce_prod(element_shape)
        
        # Compute logarithmic determinant
        logarithmic_determinant = element_count * tf.math.reduce_sum(tf.math.log(1/(tf.abs(self.scale)))) # All channel for a single element 
        logarithmic_determinant = tf.ones([instance_count], dtype=x.dtype)

        # Outputs
        return logarithmic_determinant

class SequentialFlowNetwork(FlowLayer):
    """Manages flow through layers with forward and inverse support."""

    def __init__(self, layers: List[FlowLayer]):
        
        # Super
        super(SequentialFlowNetwork, self).__init__()
        
        # Attributes
        self.layers = layers

    def call(self, x: tf.Tensor) -> tf.Tensor:
        
        # Transform
        for layer in self.layers: x = layer(x=x)
        y_hat = x

        # Outputs
        return y_hat
    
    def invert(self, x: tf.Tensor) -> tf.Tensor:
        
        # Transform
        for layer in self.layers: x = layer.inverse(x=x)
        y_hat = x

        # Outputs
        return y_hat

    def compute_logarithmic_determinant(self, x: tf.Tensor) -> tf.Tensor:
        
        # Transform
        logarithmic_determinant = 0
        for layer in self.layers: 
            logarithmic_determinant += layer.compute_logarithmic_determinant(x=x) 
            x = layer(x=x)
            
        # Outputs
        return logarithmic_determinant
'''
class VectorTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        self.config = config

        self.in_channel = retrieve(config, "Transformer/in_channel")
        self.n_flow = retrieve(config, "Transformer/n_flow")
        self.depth_submodules = retrieve(config, "Transformer/hidden_depth")
        self.hidden_dim = retrieve(config, "Transformer/hidden_dim")
        modules = [ActivationNormalization, DoubleVectorCouplingBlock, Shuffle]
        self.realnvp = EfficientVRNVP(modules, self.in_channel, self.n_flow, self.hidden_dim,
                                   hidden_depth=self.depth_submodules)

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        input = input.squeeze()
        out, logdet = self.realnvp(input)
        return out[:, :, None, None], logdet

    def reverse(self, out):
        out = out.squeeze()
        return self.realnvp(out, reverse=True)[0][:, :, None, None]

class FactorTransformer(VectorTransformer):

    def __init__(self, config):
        super().__init__(config)
        self.n_factors = retrieve(config, "Transformer/n_factors", default=2)
        self.factor_config = retrieve(config, "Transformer/factor_config", default=list())

    def forward(self, input):
        out, logdet = super().forward(input)
        if self.factor_config:
            out = torch.split(out, self.factor_config, dim=1)
        else:
            out = torch.chunk(out, self.n_factors, dim=1)
        return out, logdet

    def reverse(self, out):
        out = torch.cat(out, dim=1)
        return super().reverse(out)
'''



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from gyoza.utilities import math as gum

    # Generate some data
    instance_count = 100
    x, y = np.meshgrid(np.arange(start=-1, stop=1, step=0.1), np.arange(start=-1, stop=1, step=0.1))
    x = np.reshape(x,[-1,]); y = np.reshape(y, [-1])
    x, y = gum.swirl(x=x,y=y)
    x = tf.transpose([x,y], [1,0]); del y
    labels = ([0] * (len(x)//2)) + ([1] * (len(x)//2))
    
    # Further transformation
    #shuffle = Shuffle(channel_count=2)
    #basic_fully_connected = BasicFullyConnectedNet(latent_channel_count=2, output_channel_count=2, depth=2)
    
    #tmp = shuffle(x=x)
    #y_hat = basic_fully_connected(x=tmp)
    
    # Visualization
    plt.figure()
    plt.scatter(x[:,0],x[:,1],c=labels)
    plt.show()