import numpy as np
import tensorflow as tf
from typing import Any, Tuple, List, Callable
from abc import ABC
from gyoza.utilities import tensors as utt
import gyoza.modelling.masks as mms

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

    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:
        """Computes the jacobian determinant of this layer's :py:meth:`call` on a logarithmic scale. The
        natural logarithm is chosen for numerical stability.

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
    
    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:

        # Outputs
        return 0

class CouplingLayer(FlowLayer, ABC):
    """This layer couples the input ``x`` with itself inside the method :py:meth:`call`. In doing so, :py:meth:`call` 
    obtains two copies of x, referred to as x_1, x_2 using a binary mask and its negative (1-mask), respectively. The half x_1 
    is mapped to coupling parameters via a user-provided model, called :py:meth:``compute_coupling_parameters``. This can be e.g. an 
    artificial neural network. Next, :py:meth:`call` uses the internally defined :py:meth:`__couple__` method to couple x_2 with 
    those parameters. This coupling is designed to be trivially invertible, given the parameters. It can be for instance y_hat = 
    x + parameters, which has the trivial inverse x = y_hat - parameters. Due to the splitting of x and the fact that 
    :py:func:`compute_coupling_parameters` will only be evaluated in the forward direction, the overall :py:meth:`call` 
    method will be trivially invertible. Similarly, its Jacobian determinant remains trivial and thus tractable.

    :param compute_coupling_parameters: The function that shall be used to compute parameters. See the placeholder member
        :py:meth:`compute_coupling_parameters` for a detailed description of requirements.
    :type compute_coupling_parameters: :class:`tensorflow.keras.Model`
    :param mask: The mask used to select one half of the data while discarding the other half.
    :type mask: :class:`gyoza.modelling.masks.Mask`
    
    References:

        - "NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION" by Laurent Dinh and David Krueger and Yoshua Bengio.
        - "Density estimation using real nvp" by Laurent Dinh, Jascha Sohl-Dickstein and Samy Bengio.
    """

    def __init__(self, compute_coupling_parameters: tf.keras.Model, mask: mms.Mask):

        # Super
        super(CouplingLayer, self).__init__()

        # Attributes
        self.compute_coupling_parameters = compute_coupling_parameters
        
        self.__mask__ = mask
        """(:class:`gyoza.modelling.masks.Mask`) - The mask used to select one half of the data while discarding the other half."""

    @staticmethod
    def __assert_parameter_validity__(parameters: tf.Tensor or List[tf.Tensor]) -> bool:
        """Determines whether the parameters are valid for coupling.
       
        :param parameters: The parameters to be checked.
        :type parameters: :class:`tensorflow.Tensor` or :class:`List[tensorflow.Tensor]`
        """

        # Assertion
        assert isinstance(parameters, tf.Tensor), f"For this coupling layer parameters is assumed to be of type tensorflow.Tensor, not {type(parameters)}"
    
    def compute_coupling_parameters(self, x: tf.Tensor) -> tf.Tensor:
        """A callable, e.g. a :class:`tensorflow.keras.Model` object that maps ``x`` to coupling parameters used to couple 
        ``x`` with itself. The model may be arbitrarily complicated and does not have to be invertible.
        
        :param x: The data to be transformed. Shape [batch size, ...] has to allow for masking via 
            :py:attr:`self.__mask__`.
        :type x: :class:`tensorflow.Tensor`
        :return: y_hat (:class:`tensorflow.Tensor`) - The transformed version of ``x``. It's shape must support the Hadamard product
            with ``x``."""
        
        raise NotImplementedError()

    def call(self, x: tf.Tensor) -> tf.Tensor:

        # Split x
        x_1 = self.__mask__.apply(x=x)

        # Compute parameters
        coupling_parameters = self.compute_coupling_parameters(x_1)
        self.__assert_parameter_validity__(parameters=coupling_parameters)

        # Couple
        y_hat_1 = x_1
        y_hat_2 = self.__mask__.apply(x=self.__couple__(x=x, parameters=coupling_parameters), is_positive=False)

        # Combine
        y_hat = y_hat_1 + y_hat_2

        # Outputs
        return y_hat
    
    def __couple__(self, x: tf.Tensor, parameters: tf.Tensor or List[tf.Tensor]) -> tf.Tensor:
        """This function implements an invertible coupling for inputs ``x`` and ``parameters``.
        
        :param x: The data to be transformed. Shape assumed to be [batch size, ...] where ... depends on axes of :py:attr:`self.__mask__`. 
        :type x: :class:`tensorflow.Tensor`
        :param parameters: Constitutes the parameters that shall be used to transform ``x``. It's shape is assumed to support the 
            Hadamard product with ``x``.
        :type parameters: :class:`tensorflow.Tensor` or :class:`List[tensorflow.Tensow]`
        :return: y_hat (:class:`tensorflow.Tensor`) - The coupled tensor of same shape as ``x``."""

        raise NotImplementedError()
    
    def __decouple__(self, y_hat: tf.Tensor, parameters: tf.Tensor or List[tf.Tensor]) -> tf.Tensor:
        """This function is the inverse of :py:meth:`__couple__`.
        
        :param y_hat: The data to be transformed. Shape assumed to be [batch size, ...] where ... depends on axes :py:attr:`self.__mask__`.
        :type y_hat: :class:`tensorflow.Tensor`
        :param parameters: Constitutes the parameters that shall be used to transform ``y_hat``. It's shape is assumed to support the 
            Hadamard product with ``x``.
        :type parameters: :class:`tensorflow.Tensor` or :class:`List[tensorflow.Tensow]`
        :return: y_hat (:class:`tensorflow.Tensor`) - The decoupled tensor of same shape as ``y_hat``."""

        raise NotImplementedError()
    
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        
        # Split
        y_hat_1 = self.__mask__.apply(x=y_hat)

        # Compute parameters
        coupling_parameters = self.compute_coupling_parameters(y_hat_1)
        self.__assert_parameter_validity__(parameters=coupling_parameters)
        
        # Decouple
        x_1 = y_hat_1
        x_2 = self.__mask__.apply(x=self.__decouple__(y_hat=y_hat, parameters=coupling_parameters), is_positive=False)

        # Combine
        x = x_1 + x_2

        # Outputs
        return x
    
class AdditiveCouplingLayer(CouplingLayer):
    """This couplign layer implements an additive coupling of the form y = x + parameters"""

    def __init__(self, compute_coupling_parameters: tf.keras.Model, mask: tf.Tensor):
        
        # Super
        super(AdditiveCouplingLayer, self).__init__(compute_coupling_parameters=compute_coupling_parameters, mask=mask)

    def __couple__(self, x: tf.Tensor, parameters: tf.Tensor or List[tf.Tensor]) -> tf.Tensor:
        
        # Couple
        y_hat = x + parameters

        # Outputs
        return y_hat
    
    def __decouple__(self, y_hat: tf.Tensor, parameters: tf.Tensor or List[tf.Tensor]) -> tf.Tensor:
        
        # Decouple
        x = y_hat - parameters

        # Outputs
        return x
    
    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:
        
        # Shape
        batch_size = x.shape[0]

        # Outputs
        return tf.zeros([batch_size], dtype=tf.float32)

class AffineCouplingLayer(CouplingLayer):
    """This coupling layer implements an affine coupling of the form y = scale * x + location, where scale = exp(parameters[0])
    and location = parameters[1]. To prevent division by zero during decoupling, the exponent of parameters[0] is used as scale."""

    def __init__(self, compute_coupling_parameters: tf.keras.Model, mask: tf.Tensor):
        
        # Super
        super(AffineCouplingLayer, self).__init__(compute_coupling_parameters=compute_coupling_parameters, mask=mask)

    @staticmethod
    def __assert_parameter_validity__(parameters: tf.Tensor or List[tf.Tensor]) -> bool:

        # Assert
        is_valid = type(parameters) == type([]) and len(parameters) == 2
        is_valid = is_valid and type(parameters[0]) == tf.Tensor and type(parameters[1]) == tf.Tensor
                                                                          
        assert is_valid, f"For this coupling layer parameters is assumed to be of type List[tensorflow.Tensor], not {type(parameters)}."
    
    def __couple__(self, x: tf.Tensor, parameters: tf.Tensor or List[tf.Tensor]) -> tf.Tensor:
        
        # Unpack
        scale = tf.exp(parameters[0])
        location = parameters[1]

        # Couple
        y_hat = scale * x + location

        # Outputs
        return y_hat
    
    def __decouple__(self, y_hat: tf.Tensor, parameters: tf.Tensor or List[tf.Tensor]) -> tf.Tensor:
        
        # Unpack
        scale = tf.exp(parameters[0])
        location = parameters[1]

        # Decouple
        x = (y_hat - location) / scale

        # Outputs
        return x
    
    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:
        
        # Split x
        x_1 = self.__mask__.apply(x=x)

        # Compute parameters
        coupling_parameters = self.compute_coupling_parameters(x_1)

        # Determinant
        logarithmic_scale = coupling_parameters[0]
        logarithmic_determinant = 0
        for axis in self.__mask__.__axes__:
            logarithmic_determinant += tf.reduce_sum(logarithmic_scale, axis=axis)

        # Outputs
        return logarithmic_determinant

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
           
    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:

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

    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:
        
        # Transform
        logarithmic_determinant = 0
        for layer in self.layers: 
            logarithmic_determinant += layer.compute_jacobian_determinant(x=x) 
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