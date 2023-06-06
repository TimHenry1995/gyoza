import numpy as np
import tensorflow as tf
from typing import Any, Tuple, List, Callable
from abc import ABC
from gyoza.utilities import tensors as utt

class BasicFullyConnectedNet(tf.keras.Model):
    """This class provides a basic fully connected network. It essentially passes data through several
    :class:`tensorflow.keras.layers.Dense` layers and applies optional batch normalization. 
    
    :param int latent_channel_count: The number of channels maintained between intermediate layers. 
    :param int output_channel_count: The number of channels of the final layer.
    :param int depth: The number of layers to be used in between the input and output. If set to 0, there will only be a single 
        layer mapping from input to output. If set to 1, then there will be 1 intermediate layer, etc. 
    :param bool, optional use_tanh: Indicates whether each layer shall use the hyperbolic rangent activaction function. If set to false, 
        then a leaky relu is used. Defaults to False.
    :param bool, optional use_batch_normalization: Indicates whether each layer shall use batch normalization or not. Defaults to False."""

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
        
        :param x: The data tensor that should be passed through the network.
        :type x: :class:`tensorflow.Tensor` 
        :return: y_hat (:class:`tensorflow.Tensor`) - The prediction."""
        
        # Predict
        y_hat = self.sequential(x)

        # Outputs:
        return y_hat

class FlowLayer(tf.keras.Model, ABC):
    """Abstract base class for flow layers
    
    References:
        - "Density estimation using Real NVP" by Laurent Dinh and Jascha Sohl-Dickstein and Samy Bengio.
    """

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Executes the operation of this layer in the forward direction.

        :param x: The data to be tranformed. Assumed to be of shape [batch size, ...].
        :type x: :class:`tensorflow.Tensor`
        :return: y_hat (:class:`tensorflow.Tensor`) - The output of the transformation of shape [batch size, ...]."""        
        raise NotImplemented()

    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        """Executes the operation of this layer in the inverse direction.

        :param y_hat: The data to be transformed. Assumed to be of shape [batch size, ...].
        :type y_hat: :class:`tensorflow.Tensor`
        :return: x (:class:`tensorflow.Tensor`) - The output of the transformation of shape [batch size, ...]."""        

        raise NotImplemented()

    def compute_logarithmic_determinant(self, x: tf.Tensor) -> tf.Tensor:
        """Computes the logarithmic determinant of this layer's forward operation.

        :param x: The data at which the determinant shall be computed. Assumed to be of shape [batch size, ...].
        :type x: :class:`tensorflow.Tensor`
        :return: logarithmic_determinant (:class:`tensorflow.Tensor`) - A measure of how much this layer contracts or dilates space at the point x. Shape == [batch size].
        """        

        raise NotImplemented()

class Mask(FlowLayer, ABC):
    pass

class CheckerBoardMask(Mask):
    """A mask that applies a checkerboard pattern to its input. Creates a mask and saves it as non-trainable tf.Variable in 
        an attribute. The mask is thus deterministic and it enables loading and saving the model.
        
        :param shape: The shape of the checker board pattern, assumed to be of minimal dimensionality at most 2.
            That means, a 2D spatial mask would have shape, e.g. [64, 128], disregarding the fact that 
            batch and channel axes exist. The mask will be broadcast during use in call.
        :type shape: :class:`List[int]`"""

    def __init__(self, shape: List[int]) -> None:

        # Input validity
        assert len(shape) <= 2, "There must be at most two axes along which the checker board pattern shall be applied."

        # Initialize
        mask = np.zeros(shape=shape)

        # Case 1-dimensional
        if len(shape) == 1:

            # Set ones
            mask[::2,...] = 1

        # Case 2-dimensional
        # Iterate axes

        # Attributes
        self.__mask__ = tf.Variable(initial_value=mask, trainable=False)

    def call(self, x:tf. Tensor) -> tf.Tensor:
        raise NotImplementedError()
        
class Shuffle(FlowLayer):
    """Shuffles inputs along a given axis. The permutation used for shuffling is randomly chosen 
    once during initialization. Thereafter it is saved as a non-trainable tensorflow.Variable in a private attribute.
    Shuffling is thus deterministic and supports loading and saving.
    
    :param int channel_count: The number of channels that should be shuffled.
    :param int channel_axis: The axis of shuffling."""

    def __init__(self, channel_count, channel_axis: int = -1):

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
    
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        
        # Shuffle
        x = tf.gather(y_hat, self.__inverse_permutation__, axis=self.__axis__)
        
        # Outputs
        return x
    
    def compute_logarithmic_determinant(self, x: tf.Tensor) -> tf.Tensor:

        # Outputs
        return 0

class CouplingLayer(FlowLayer, ABC):
    """This layer splits its input x in the middle into two parts x_1, x_2 along the channel axis. For uneven channel counts one
    half is one unit larger than the other half. Then the following update rule is applied:\n
    
        - y_hat_1 = x_1\n
        - y_hat_2 = g(x_2, m(x_1))\n

    where y_hat_1 and y_hat_2 and concatenated along the channel axis to give the overall output y_hat. The function g(a,b) is the 
    coupling law that has to be invertible with respect to its first argument given the second, e.g. g(a,b) = a + b or g(a,b) = a*b 
    for b != 0. The function m(a) is unconstrained, e.g. an artificial neural network. The implementation of g(a,b) and m(a) 
    distinguishes coupling layers. 
    
    :param m: The function that shall be used to map the first half of the input to call() to weights used to transform the second 
            half of that x in g(). Its inputs shall be a tensor with channel axis at channel_axis and channel count equal to 
            channel_count // 2. Its output shall be a tensor or list of tensors that can be used to transform the second half of x
            inside g(). The output requirement on m thus depends on the subclass specific implementation for g(). 
    :type m: :class:`Callable`
    :param int channel_count: The total number of channels of the input to this layer.
    :param int channel_axis: The axis along which coupling shall be executed.

    
    References:
        - "NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION" by Laurent Dinh and David Krueger and Yoshua Bengio.
    """

    def __init__(self, m: Callable, channel_count: int, channel_axis: int = -1):

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
        """This function implements an invertible coupling law for inputs ``a`` and ``b``. It is invertible w.r.t. ``a``, given ``b``.
        
        :param a: Tensor of arbitrary shape whose channel axis is the same as self.channel_axis. This Tensor is supposed to be the
            second half of x in :meth:`call` and shall be coupled with ``b``.
        :type a: :class:`tensorflow.Tensor`
        :param b: Tensor or list of tensors whose shape is the same as that of ``a``. These tensors shall be used to transform ``a``.
        :type b: :class:`tensorflow.Tensor`, :class:`List[tensorflow.Tensow]`
        
        :return: y_hat (:class:`tensorflow.Tensor`) - The coupled tensor of same shape as ``a``."""

        raise NotImplementedError()
    
    def __inverse_g__(self, a: tf.Tensor, b: tf.Tensor or List[tf.Tensor]) -> tf.Tensor:
        """This function implements the inverse coupling law for :meth:`__g__`. 
        
        :param a: Tensor of arbitrary shape whose channel axis is the same as :py:attr:`self.channel_axis`. This Tensor is supposed to be the
            second half of x in :py:meth:`call` and in :py:meth:`invert` and it shall be decoupled from ``b``.
        :type a: :class:`tensorflow.Tensor`
        :param b: Tensor or list of tensors whose shape is the same as that of ``a``. These tensors shall be used to decouple ``a``.
        :type b: :class:`tensorflow.Tensor`
        :return: y_hat (:class:`tensorflow.Tensor`) - The decoupled tensor of same shape as ``a``."""

        raise NotImplementedError()
    
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        
        # Split
        partition_point = self.channel_count // 2
        y_hat_1, y_hat_2 = tf.split(y_hat, num_or_size_splits=[partition_point, self.channel_count - partition_point], axis=self.channel_axis)
         
        # Decouple
        x_1 = y_hat_1
        x_2 = self.__inverse_g__(a=y_hat_2, b=self.__m__(a=y_hat_1))

        # Concatenate
        x = tf.concat([x_1, x_2], axis=self.channel_axis)

        # Outputs
        return x
    
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
        
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:

        # Transform
        scale, location = self.__prepare_variables_for_computation__(x=y_hat)
        x = scale * y_hat + location

        # Outputs
        return x
           
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
    
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        
        # Transform
        for layer in self.layers: y_hat = layer.inverse(x=y_hat)
        x = y_hat

        # Outputs
        return x

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