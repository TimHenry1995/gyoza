import numpy as np
import tensorflow as tf
from typing import Any, Tuple, List, Callable
from abc import ABC
from gyoza.utilities import tensors as utt

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
        raise NotImplementedError()

    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        """Executes the operation of this layer in the inverse direction. It is thus the counterpart to :py:meth:`call`.

        :param y_hat: The data to be transformed. Assumed to be of shape [batch size, ...].
        :type y_hat: :class:`tensorflow.Tensor`
        :return: x (:class:`tensorflow.Tensor`) - The output of the transformation of shape [batch size, ...]."""        

        raise NotImplementedError()

    def compute_logarithmic_determinant(self, x: tf.Tensor) -> tf.Tensor:
        """Computes the logarithmic determinant of this layer's :py:meth:`call` operation.

        :param x: The data at which the determinant shall be computed. Assumed to be of shape [batch size, ...].
        :type x: :class:`tensorflow.Tensor`
        :return: logarithmic_determinant (:class:`tensorflow.Tensor`) - A measure of how much this layer contracts or dilates space at the point ``x``. Shape == [batch size].
        """        

        raise NotImplementedError()

class Shuffle(FlowLayer):
    """Shuffles inputs along a given axis. The permutation used for shuffling is randomly chosen 
    once during initialization. Thereafter it is saved as a non-trainable :class:`tensorflow.Variable` in a private attribute.
    Shuffling is thus deterministic and supports persistence, e.g. via :py:meth:`tensorflow.keras.Model.load_weights` or :py:meth:`tensorflow.keras.Model.save_weights`.
    
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
    """This layer couples the input ``x`` with itself inside the method :py:meth:`call`. In doing so, :py:meth:`call` 
    splits ``x`` into two halves ``x_1``, ``x_2`` using a binary mask.
    The coupling of ``x_2`` (data half) with ``x_1`` 
    (weight half) then relies on the built-in method :py:meth:`__couple__` and externally provided function 
    :py:func:`compute_weights` to compute

        - ``weights`` = :py:func:`compute_weights` , where a = ``mask`` * ``x``
        - ``y_hat`` = ``mask`` * ``x`` + (1- ``mask`` ) * :py:meth:`__couple__` , where a = ``x`` , b = ``weights``
        
    The method :py:meth:`__couple__` is implemented by the specific subclass of :class:`CouplingLayer`. It couples its first 
    parameter ``a`` (:class:`tensorflow.Tensor`) with its second parameter ``b`` (:class:`tensorflow.Tensor` or 
    :class:`List[tensorflow.Tensor]`). This coupling is usually a simple transformation such as __g__(a,b) = a + b or 
    __g__(a,b) = a*b for b != 0. Consequently, :py:meth:`__g__` is trivially invertible and for :py:meth:`invert` and has a trivial 
    Jacobian determinant needed for :py:meth:`compute_logarithmic_determinant`).

    :param m: The function that shall be used to map ``x_1`` ( first half of ``x`` in :py:meth:`call`) to coupling weights. It 
        takes a single input ``a`` (:class:`tensorflow.Tensor`) with channel axis at ``channel_axis`` and channel count equal to 
        ``channel_count`` // 2. Its output shall be a :class:`tensorflow.Tensor` or :class:`List[tensorflow.Tensor]`. 
        Important: The required shape of :py:func:`m` 's output thus depends on the subclass specific implementation for :py:func`__g__`. 
    :type m: :class:`Callable`
    :param int channel_count: The total number of channels of the input ``x`` to :py:meth:`call`.
    :param int channel_axis: The axis along which coupling shall be executed in :py:meth:`call`.

    
    References:

        - "NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION" by Laurent Dinh and David Krueger and Yoshua Bengio.
        - "Density estimation using real nvp" by Laurent Dinh, Jascha Sohl-Dickstein and Samy Bengio.
    """

    def __init__(self, compute_weights: Callable, flatten: Callable, channel_count: int, channel_axis: int = -1):

        # Attributes
        self.compute_weights = compute_weights
        self.flatten = flatten
        self.channel_count = channel_count
        """The total number of channels of the input ``x`` to :py:meth:`call`."""
        self.channel_axis = channel_axis
        """The axis along which coupling shall be executed in :py:meth:`call`."""

    def call(self, x: tf.Tensor) -> tf.Tensor:
        
        # Split
        partition_point = self.channel_count // 2
        x_1, x_2 = tf.split(x, num_or_size_splits=[partition_point, self.channel_count - partition_point], axis=self.channel_axis)
         
        # Couple
        y_hat_1 = x_1
        y_hat_2 = self.__g__(a=x_2, b=self.compute_weights(a=x_1))

        # Concatenate
        y_hat = tf.concat([y_hat_1, y_hat_2], axis=self.channel_axis)

        # Outputs
        return y_hat
    
    def __g__(self, a: tf.Tensor, b: tf.Tensor or List[tf.Tensor]) -> tf.Tensor:
        """This function implements an invertible coupling law for inputs ``a`` and ``b``. It is invertible w.r.t. ``a``, given ``b``.
        
        :param a: Tensor of arbitrary shape whose channel axis is the same as :py:attr:`self.channel_axis`. 
        :type a: :class:`tensorflow.Tensor`
        :param b: Constitutes the weights that shall be used to transform ``a``.
        :type b: :class:`tensorflow.Tensor` or :class:`List[tensorflow.Tensow]`
        
        :return: y_hat (:class:`tensorflow.Tensor`) - The coupled tensor of same shape as ``a``."""

        raise NotImplementedError()
    
    def __inverse_g__(self, a: tf.Tensor, b: tf.Tensor or List[tf.Tensor]) -> tf.Tensor:
        """This function implements the inverse coupling law for :meth:`__g__`. 
        
        :param a: Tensor of arbitrary shape whose channel axis is the same as :py:attr:`self.channel_axis`.
        :type a: :class:`tensorflow.Tensor`
        :param b: Tensor or list of tensors. These tensors shall be the weights used to decouple ``a``.
        :type b: :class:`tensorflow.Tensor`
        :return: y_hat (:class:`tensorflow.Tensor`) - The decoupled tensor of same shape as ``a``."""

        raise NotImplementedError()
    
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        
        # Split
        partition_point = self.channel_count // 2
        y_hat_1, y_hat_2 = tf.split(y_hat, num_or_size_splits=[partition_point, self.channel_count - partition_point], axis=self.channel_axis)
         
        # Decouple
        x_1 = y_hat_1
        x_2 = self.__inverse_g__(a=y_hat_2, b=self.compute_weights(a=y_hat_1))

        # Concatenate
        x = tf.concat([x_1, x_2], axis=self.channel_axis)

        # Outputs
        return x
    
    def compute_logarithmic_determinant(self, x: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()
    
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
    """A trainable channel-wise location and scale transform of the data. Is initialized to produce zero mean and unit variance.
    
    :param channel_count: The number of channels for which the transformation shall be executed.
    :type channel_count: int
    :param channel_axis: The axis along which the transformation shall be executed. Each entry along this axis will have 
        its own transformation applied along all other axis."""

    def __init__(self, channel_count: int, channel_axis: int = -1):

        # Super
        super().__init__()
        
        # Attributes
        self.__location__ = tf.Variable(tf.zeros(channel_count), trainable=True)
        """The value by which each data point shall be translated."""

        self.__scale__ = tf.Variable(tf.ones(channel_count), trainable=True)
        """The value by which each data point shall be scaled."""

        self.__is_initialized__ = False
        """An indicator for whether lazy initialization has been executed previously."""

        self.__channel_axis__ = channel_axis
        """The axis along which a data point shall be transformed."""

    def __initialize__(self, x: tf.Tensor) -> None:
        """This method shall be used to lazily initialize the variables of self.
        
        :param x: The data that is propagated through :py:meth:`call`.
        :type x: :class:`tensorflow.Tensor`"""

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

    def __scale_to_non_zero__(self) -> None:
        """Mutating method that corrects the :py:attr:`__scale__` attribute where it is equal to zero by adding a constant epsilon. 
        This is useful to prevent scaling by 0 which is not invertible."""
        
        # Correct scale where it is equal to zero to prevent division by zero
        epsilon = tf.stop_gradient(tf.constant(1e-6 * (self.__scale__.numpy() == 0), dtype=self.__scale__.dtype)) 
        self.__scale__.assing(self.__scale__ + epsilon)

    def __prepare_variables_for_computation__(self, x:tf.Tensor) -> Tuple[tf.Variable, tf.Variable]:
        """Prepares the variables for computation with data. This involves adjusting the scale to be non-zero and ensuring variable shapes are compatible with the data.
        
        :param x: Data to be passed through :py:meth:`call`. It's shape must agree with input ``x`` of :py:meth:`self.__reshape_variables__`.

        :return: 
            - location (tensorflow.Variable) - The :py:attr:`__location__` attribute shaped to fit ``x``. 
            - scale (tensorflow.Variable) - The :py:attr:`__scale__` attribute ensured to be non-zero and shaped to fit ``x``."""

        # Preparations
        self.__scale_to_non_zero__()
        axes = list(range(len(x.shape))); axes.remove(self.__channel_axis__)
        location = utt.expand_axes(x=self.__location__, axes=axes)
        scale = utt.expand_axes(x=self.__scale__, axes=axes)

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
    """This network manages flow through several :class:`FlowLayer` objects in a single path sequential way."""

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