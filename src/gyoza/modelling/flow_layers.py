import numpy as np
import tensorflow as tf
from typing import Any, Tuple, List, Callable
from abc import ABC
import abc
from gyoza.utilities import tensors as utt
import gyoza.modelling.masks as mms
import copy as cp
from gyoza.modelling import losses as mls

class FlowLayer(tf.keras.Model, ABC):
    """Abstract base class for flow layers. Any input to this layer is assumed to have ``shape`` along ``axes`` as specified during
    initialization.
    
    :param shape: The shape of the input that shall be transformed by this layer. If you have e.g. a tensor [batch size, width, 
        height, channel count] and you want this layer to transform along width and height you enter [width, height] as shape. If you 
        want the layer to operate on the channels you provide [channel count] instead.
    :type shape: List[int]
    :param axes: The axes of transformation. In the example for ``shape`` on width and height you would enter [1,2] here, In the 
        example for channels you would enter [3] here. ``axes`` is assumed not to contain the axis 0, i.e. the batch axis.
    :type axes: List[int]

    References:

        - "Density estimation using Real NVP" by Laurent Dinh and Jascha Sohl-Dickstein and Samy Bengio.
        - "Glow: Generative Flow with Invertible 1x1 Convolutions" by Diederik P. Kingma and Prafulla Dhariwal
        - "NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION" by Laurent Dinh, David Krueger and Yoshua Bengio
        - "GLOWin: A Flow-based Invertible Generative Framework for Learning Disentangled Feature Representations in Medical Images" by Aadhithya Sankar, Matthias Keicher, Rami Eisawy, Abhijeet Parida, Franz Pfister, Seong Tae Kim and  Nassir Navab1,6,†
        - "A Disentangling Invertible Interpretation Network for Explaining Latent Representations" by Patrick Esser, Robin Rombach and Bjorn Ommer
    """

    @abc.abstractmethod
    def __init__(self, shape: List[int], axes: List[int], **kwargs):
        """This constructor shall be used by subclasses only"""

        # Super
        super(FlowLayer, self).__init__(**kwargs)

        # Input validity
        assert len(shape) == len(axes), f"The input shape ({shape}) is expected to have as many entries as the input axes ({axes})."
        for i in range(len(axes)-1):
            assert axes[i] < axes[i+1], f"The axes in input axes ({axes}) are assumed to be strictly ascending"

        assert 0 not in axes, f"The input axes ({axes}) must not contain the batch axis, i.e. 0."

        # Attributes
        self.__shape__ = cp.copy(shape)
        """The shape of the input that shall be transformed by this layer. For detail, see constructor of :class:`FlowLayer`"""

        self.__axes__ = cp.copy(axes)
        """The axes of transformation. For detail, see constructor of :class:`FlowLayer`"""

    @abc.abstractmethod
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Executes the operation of this layer in the forward direction.

        :param x: The data to be tranformed. Assumed to be of shape [batch size, ...].
        :type x: :class:`tensorflow.Tensor`
        :return: y_hat (:class:`tensorflow.Tensor`) - The output of the transformation of shape [batch size, ...]."""        
        raise NotImplementedError()

    @abc.abstractmethod
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        """Executes the operation of this layer in the inverse direction. It is thus the counterpart to :py:meth:`call`.

        :param y_hat: The data to be transformed. Assumed to be of shape [batch size, ...].
        :type y_hat: :class:`tensorflow.Tensor`
        :return: x (:class:`tensorflow.Tensor`) - The output of the transformation of shape [batch size, ...]."""        

        raise NotImplementedError()

    @abc.abstractmethod
    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:
        """Computes the jacobian determinant of this layer's :py:meth:`call` on a logarithmic scale. The
        natural logarithm is chosen for numerical stability.

        :param x: The data at which the determinant shall be computed. Assumed to be of shape [batch size, ...].
        :type x: :class:`tensorflow.Tensor`
        :return: logarithmic_determinant (:class:`tensorflow.Tensor`) - A measure of how much this layer contracts or dilates space at the point ``x``. Shape == [batch size].
        """        

        raise NotImplementedError()

class Shuffle(FlowLayer):
    """Shuffles inputs along the given axes. The permutation used for shuffling is randomly chosen once during initialization. 
    Thereafter it is saved as a private attribute. Shuffling is thus deterministic from there on.
    """

    def __init__(self, shape: List[int], axes: List[int], **kwargs):

        # Super
        super(Shuffle, self).__init__(shape=shape, axes=axes, **kwargs)
        
        # Attributes
        unit_count = tf.reduce_prod(shape).numpy()
        permutation = tf.random.shuffle(tf.range(unit_count))
        self.__forward_permutation__ = tf.Variable(permutation, trainable=False, name="forward_permutation") # name is needed for getting and setting weights
        self.__inverse_permutation__ = tf.Variable(tf.argsort(permutation), trainable=False, name="inverse_permutation")
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        
        # Initialize
        old_shape = cp.copy(x.shape)

        # Flatten along self.__axes__ to fit permutation matrix
        x = utt.flatten_along_axes(x=x, axes=self.__axes__)

        # Shuffle
        y_hat = tf.gather(x, self.__forward_permutation__, axis=self.__axes__[0])

        # Unflatten to restore original shape
        y_hat = tf.reshape(y_hat, shape=old_shape)
        
        # Outputs
        return y_hat
    
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        
        # Initialize
        old_shape = y_hat.shape

        # Flatten along self.__axes__ to fit permutation matrix
        y_hat = utt.flatten_along_axes(x=y_hat, axes=self.__axes__)

        # Shuffle
        y_hat = tf.gather(y_hat, self.__inverse_permutation__, axis=self.__axes__[0])

        # Unflatten to restore original shape
        x = tf.reshape(y_hat, shape=old_shape)
        
        # Outputs
        return x
    
    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:

        # Copmute
        batch_size = x.shape[0]
        logarithmic_determinant = tf.zeros([batch_size], dtype=tf.float32)

        # Outputs
        return logarithmic_determinant

class Coupling(FlowLayer, ABC):
    """This layer couples the input ``x`` with itself inside the method :py:meth:`call`. In doing so, :py:meth:`call` 
    obtains two copies of x, referred to as x_1, x_2 using a binary mask and its negative (1-mask), respectively. The half x_1 
    is mapped to coupling parameters via a user-provided model, called :py:meth:``compute_coupling_parameters``. This can be e.g. an 
    artificial neural network. Next, :py:meth:`call` uses the internally defined :py:meth:`__couple__` method to couple x_2 with 
    those parameters. This coupling is designed to be trivially invertible, given the parameters. It can be for instance y_hat = 
    x + parameters, which has the trivial inverse x = y_hat - parameters. Due to the splitting of x and the fact that 
    :py:func:`compute_coupling_parameters` will only be evaluated in the forward direction, the overall :py:meth:`call` 
    method will be trivially invertible. Similarly, its Jacobian determinant remains trivial and thus tractable.

    :param shape: See base class :class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :class:`FlowLayer`.
    :type axes: List[int]
    :param compute_coupling_parameters: The function that shall be used to compute parameters. See the placeholder member
        :py:meth:`compute_coupling_parameters` for a detailed description of requirements.
    :type compute_coupling_parameters: :class:`tensorflow.keras.Model`
    :param mask: The mask used to select one half of the data while discarding the other half.
    :type mask: :class:`gyoza.modelling.masks.Mask`
    
    References:

        - "NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION" by Laurent Dinh and David Krueger and Yoshua Bengio.
        - "Density estimation using real nvp" by Laurent Dinh, Jascha Sohl-Dickstein and Samy Bengio.
    """

    def __init__(self, shape: List[int], axes: List[int], compute_coupling_parameters: tf.keras.Model, mask: mms.Mask, **kwargs):

        # Super
        super(Coupling, self).__init__(shape=shape, axes=axes, **kwargs)

        # Input validity
        shape_message = f"The shape ({shape}) provided to the coupling layer and that provided to the mask ({mask.__mask__.shape}) are expected to be the same."
        assert len(shape) == len(mask.__mask__.shape), shape_message
        for i in range(len(shape)):
            assert shape[i] == mask.__mask__.shape[i], shape_message

        axes_message = f"The axes ({axes}) provided to the coupling layer and that provided to the mask ({mask.__axes__}) are expected to be the same."
        assert len(axes) == len(mask.__axes__), axes_message
        for i in range(len(axes)):
            assert axes[i] == mask.__axes__[i], axese_message

        # Attributes
        self.__compute_coupling_parameters__ = compute_coupling_parameters
        """(Callable) used inside the wrapper :py:meth:`compute_coupling_parameters`"""
        
        self.__mask__ = mask
        """(:class:`gyoza.modelling.masks.Mask`) - The mask used to select one half of the data while discarding the other half."""

    @staticmethod
    def __assert_parameter_validity__(parameters: tf.Tensor or List[tf.Tensor]) -> bool:
        """Determines whether the parameters are valid for coupling.
       
        :param parameters: The parameters to be checked.
        :type parameters: :class:`tensorflow.Tensor` or List[:class:`tensorflow.Tensor`]
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
        
        # Propagate
        # Here we can not guarantee that the provided function uses x as name for first input.
        # We thus cannot use keyword input x=x. We have to trust that the first input is correctly interpreted as x.
        y_hat = self.__compute_coupling_parameters__(x)

        # Outputs
        return y_hat

    def call(self, x: tf.Tensor) -> tf.Tensor:

        # Split x
        x_1 = self.__mask__.call(x=x)

        # Compute parameters
        coupling_parameters = self.compute_coupling_parameters(x_1)
        self.__assert_parameter_validity__(parameters=coupling_parameters)

        # Couple
        y_hat_1 = x_1
        y_hat_2 = self.__mask__.call(x=self.__couple__(x=x, parameters=coupling_parameters), is_positive=False)

        # Combine
        y_hat = y_hat_1 + y_hat_2

        # Outputs
        return y_hat
    
    @abc.abstractmethod
    def __couple__(self, x: tf.Tensor, parameters: tf.Tensor or List[tf.Tensor]) -> tf.Tensor:
        """This function implements an invertible coupling for inputs ``x`` and ``parameters``.
        
        :param x: The data to be transformed. Shape assumed to be [batch size, ...] where ... depends on axes of :py:attr:`self.__mask__`. 
        :type x: :class:`tensorflow.Tensor`
        :param parameters: Constitutes the parameters that shall be used to transform ``x``. It's shape is assumed to support the 
            Hadamard product with ``x``.
        :type parameters: :class:`tensorflow.Tensor` or List[:class:`tensorflow.Tensow`]
        :return: y_hat (:class:`tensorflow.Tensor`) - The coupled tensor of same shape as ``x``."""

        raise NotImplementedError()
    
    @abc.abstractmethod
    def __decouple__(self, y_hat: tf.Tensor, parameters: tf.Tensor or List[tf.Tensor]) -> tf.Tensor:
        """This function is the inverse of :py:meth:`__couple__`.
        
        :param y_hat: The data to be transformed. Shape assumed to be [batch size, ...] where ... depends on axes :py:attr:`self.__mask__`.
        :type y_hat: :class:`tensorflow.Tensor`
        :param parameters: Constitutes the parameters that shall be used to transform ``y_hat``. It's shape is assumed to support the 
            Hadamard product with ``x``.
        :type parameters: :class:`tensorflow.Tensor` or List[:class:`tensorflow.Tensow`]
        :return: y_hat (:class:`tensorflow.Tensor`) - The decoupled tensor of same shape as ``y_hat``."""

        raise NotImplementedError()
    
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        
        # Split
        y_hat_1 = self.__mask__.call(x=y_hat)

        # Compute parameters
        coupling_parameters = self.compute_coupling_parameters(y_hat_1)
        self.__assert_parameter_validity__(parameters=coupling_parameters)
        
        # Decouple
        x_1 = y_hat_1
        x_2 = self.__mask__.call(x=self.__decouple__(y_hat=y_hat, parameters=coupling_parameters), is_positive=False)

        # Combine
        x = x_1 + x_2

        # Outputs
        return x
    
class AdditiveCoupling(Coupling):
    """This coupling layer implements an additive coupling of the form y = x + parameters"""

    def __init__(self, shape: List[int], axes: List[int], compute_coupling_parameters: tf.keras.Model, mask: tf.Tensor, **kwargs):
        
        # Super
        super(AdditiveCoupling, self).__init__(shape=shape, axes=axes, compute_coupling_parameters=compute_coupling_parameters, mask=mask, **kwargs)

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
        
        # Copmute
        batch_size = x.shape[0]
        logarithmic_determinant = tf.zeros([batch_size], dtype=tf.float32)

        # Outputs
        return logarithmic_determinant

class AffineCoupling(Coupling):
    """This coupling layer implements an affine coupling of the form y = scale * x + location, where scale = exp(parameters[0])
    and location = parameters[1]. To prevent division by zero during decoupling, the exponent of parameters[0] is used as scale."""

    def __init__(self, shape: List[int], axes: List[int], compute_coupling_parameters: tf.keras.Model, mask: tf.Tensor, **kwargs):
        
        # Super
        super(AffineCoupling, self).__init__(shape=shape, axes=axes, compute_coupling_parameters=compute_coupling_parameters, mask=mask, **kwargs)

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
        x_1 = self.__mask__.call(x=x)

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
    """A trainable location and scale transformation of the data. For each unit of the specified input shape, a scale and a location 
    parameter is used. That is, if shape = [width, height] then 2 * width * height many parameters are used. Each pair of location and
    scale is initialized to produce mean equal to 0 and variance equal to 1 for its unit. To allow for invertibility, the scale parameter 
    has to be non-zero and is therefore chosen to be on an exponential scale. Each unit thus has the following activation 
    normalization:
    
    - y_hat = (x-l)/s, where s and l are the scale and location parameters for this unit, respectively.

    :param shape: See base class :class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :class:`FlowLayer`.
    :type axes: List[int]
    
    References:

    - "Glow: Generative Flow with Invertible 1x1 Convolutions" by Diederik P. Kingma and Prafulla Dhariwal
    - "A Disentangling Invertible Interpretation Network for Explaining Latent Representations" by Patrick Esser, Robin Rombach and Bjorn Ommer
    """
    
    def __init__(self, shape: List[int], axes: List[int], **kwargs):

        # Super
        super(ActivationNormalization, self).__init__(shape=shape, axes=axes)
        
        # Attributes
        self.__location__ = tf.Variable(tf.zeros(shape), trainable=True, name="__location__")
        """The value by which each data point shall be translated."""

        self.__scale__ = tf.Variable(tf.ones(shape), trainable=True, name="__scale__")
        """The value by which each data point shall be scaled."""

        self.__is_initialized__ = False
        """An indicator for whether lazy initialization has been executed previously."""

    def __lazy_init__(self, x: tf.Tensor) -> None:
        """This method shall be used to lazily initialize the variables of self.
        
        :param x: The data that is propagated through :py:meth:`call`.
        :type x: :class:`tensorflow.Tensor`"""

        # Move self.__axes__ to the end
        for a, axis in enumerate(self.__axes__): x = utt.move_axis(x=x, from_index=axis-a, to_index=-1) # Relies on assumption that axes are ascending

        # Flatten other axes
        other_axes = list(range(len(x.shape)))[:-len(self.__axes__)]
        x = utt.flatten_along_axes(x=x, axes=other_axes) # Shape == [product of all other axes] + self.__shape__

        # Compute mean and standard deviation 
        mean = tf.stop_gradient(tf.math.reduce_mean(x, axis=0)) # Shape == self.__shape__ 
        standard_deviation = tf.stop_gradient(tf.math.reduce_std(x, axis=0)) # Shape == self.__shape__ 
        
        # Update attributes
        self.__location__.assign(mean)
        scale = tf.math.log(standard_deviation+1e-16) # To initialze it for unit variance we need to use log here
        self.__scale__.assign(scale)

    def __prepare_variables_for_computation__(self, x:tf.Tensor) -> Tuple[tf.Variable, tf.Variable]:
        """Prepares the variables for computation with data. This involves adjusting the scale to be non-zero and ensuring variable shapes are compatible with the data.
        
        :param x: Data to be passed through :py:meth:`call`. It's shape must agree with input ``x`` of :py:meth:`self.__reshape_variables__`.

        :return: 
            - location (tensorflow.Variable) - The :py:attr:`__location__` attribute shaped to fit ``x``. 
            - scale (tensorflow.Variable) - The :py:attr:`__scale__` attribute ensured to be non-zero and shaped to fit ``x``."""

        # Shape variables to fit x
        axes = list(range(len(x.shape)))
        for axis in self.__axes__: axes.remove(axis)
        location = utt.expand_axes(x=self.__location__, axes=axes)
        scale = utt.expand_axes(x=self.__scale__, axes=axes)

        # Outputs
        return location, scale

    def call(self, x: tf.Tensor) -> tf.Tensor:

        # Ensure initialization of variables
        if not self.__is_initialized__: self.__lazy_init__(x=x)

        # Transform
        location, scale = self.__prepare_variables_for_computation__(x=x)
        y_hat = (x-location) / (tf.math.exp(scale))

        # Outputs
        return y_hat
        
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:

        # Transform
        scale, location = self.__prepare_variables_for_computation__(x=y_hat)
        x = tf.math.exp(scale) * y_hat + location

        # Outputs
        return x
           
    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:

        # Count elements per instance 
        batch_size = x.shape[0]
        unit_count = 1
        for axis in range(1,len(x.shape)):
            if axis not in self.__axes__:
                unit_count *= x.shape[axis] 
        
        # Compute logarithmic determinant
        # By defintion: sum across units for ln(1/scale), where scale = exp(self.__scale__)
        # Rewriting to: sum across units for ln(0) - ln(scale)
        # Rewriting to: -1 * sum across units for ln(scale)
        # rewriting to -1 sum across units for ln(exp(self.__scale__)) which result in:
        logarithmic_determinant = -1 * unit_count * tf.math.reduce_sum(self.__scale__) # All channel for a single unit 
        logarithmic_determinant = tf.ones(shape=[batch_size]) * logarithmic_determinant

        # Outputs
        return logarithmic_determinant

# TODO: Test it
class SequentialFlowNetwork(FlowLayer):
    """This network manages flow through several :class:`FlowLayer` objects in a single path sequential way.
    
    :param sequence: A list of layers.
    :type sequence: List[:class:`FlowLayer`]
    """

    def __init__(self, sequence: List[FlowLayer], **kwargs):
        
        # Super
        super(SequentialFlowNetwork, self).__init__(shape=[], axes=[], **kwargs) # Shape and axes are set to empty lists here because the individual layers may have different shapes and axes of
        
        # Attributes
        self.sequence = sequence
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        
        # Transform
        for layer in self.sequence: x = layer(x=x)
        y_hat = x

        # Outputs
        return y_hat
    
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        
        # Transform
        for layer in self.sequence: y_hat = layer.inverse(x=y_hat)
        x = y_hat

        # Outputs
        return x

    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:
        
        # Transform
        logarithmic_determinant = 0
        for layer in self.sequence: 
            logarithmic_determinant += layer.compute_jacobian_determinant(x=x) 
            x = layer(x=x)
            
        # Outputs
        return logarithmic_determinant

class SupervisedFactorNetwork(SequentialFlowNetwork):

    def __init__(self, sequence: List[FlowLayer], **kwargs):
        super().__init__(sequence=sequence, **kwargs)
        self.__loss__ = None

    def train_step(self, data):
        """_summary_

        :param data: A tuple containg the batch of X and y, respectively. X is assumed to be a tensorflow.Tensor of shape [batch size,
            2, ...] where 2 indicates the pair x_a, x_b of same factor and ... is the shape of one input instance that has to fit 
            through :py:attr:`self.sequence`. The tensorflow.Tensor y shall contain the factor indices of shape [batch size].
        :type data: Tuple(tensorflow.Tensor, tensorflow.Tensor)
        :return: metrics (Dict[str:tensroflow.keras.metrics.Metric]) - A dictionary of training metrics.
        """
        
        # Unpack inputs
        X, y = data
        z_a = X[:,0,:]; z_b = X[:,1,:]

        # Lazy initialization
        if self.__loss__ == None: self.__loss__ = mls.SupervisedFactorLoss(factor_channel_counts=tf.reduce_prod(x_a.shape[1:]).numpy(), )

        with tf.GradientTape() as tape:
            # First instance
            z_tilde_a = self(z_a, training=True)  # Forward pass
            j_a = self.compute_jacobian_determinant(x=x_a)
            
            # Second instance
            z_tilde_b = self(z_b, training=True)
            j_b = self.compute_jacobian_determinant(x=z_b)
            
            # Compute loss
            loss = self.__loss__(y_true=y, y_pred=(z_tilde_a, z_tilde_b, j_a, j_b))

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric]