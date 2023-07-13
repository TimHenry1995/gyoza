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
        height, color] and you want this layer to transform along width and height, you enter the shape [width, height]. If you 
        want the layer to operate on the color you provide [color dimension count] instead.
    :type shape: List[int]
    :param axes: The axes of transformation. In the example for ``shape`` on width and height you would enter [1,2] here, In the 
        example for color you would enter [3] here. Although axes are counted starting from zero, it is assumed that ``axes`` 
        does not contain the axis 0, i.e. the batch axis.
    :type axes: List[int]

    References:

        - "Density estimation using Real NVP" by Laurent Dinh and Jascha Sohl-Dickstein and Samy Bengio.
        - "Glow: Generative Flow with Invertible 1x1 Convolutions" by Diederik P. Kingma and Prafulla Dhariwal
        - "NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION" by Laurent Dinh, David Krueger and Yoshua Bengio
        - "GLOWin: A Flow-based Invertible Generative Framework for Learning Disentangled Feature Representations in Medical Images" by Aadhithya Sankar, Matthias Keicher, Rami Eisawy, Abhijeet Parida, Franz Pfister, Seong Tae Kim and  Nassir Navab1,6,†
        - "A Disentangling Invertible Interpretation Network for Explaining Latent Representations" by Patrick Esser, Robin Rombach and Bjorn Ommer
    """

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
        dimension_count = tf.reduce_prod(shape).numpy()
        permutation = tf.range(dimension_count,0,delta=-1)-1#tf.random.shuffle(tf.range(dimension_count))
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

        # Compute
        batch_size = x.shape[0]
        logarithmic_determinant = tf.zeros([batch_size], dtype=tf.keras.backend.floatx())

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
            assert axes[i] == mask.__axes__[i], axes_message

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
        
        # Compute
        batch_size = x.shape[0]
        logarithmic_determinant = tf.zeros([batch_size], dtype=tf.keras.backend.floatx())

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
    """A trainable location and scale transformation of the data. For each dimension of the specified input shape, a scale and a location 
    parameter is used. That is, if shape == [width, height], then 2 * width * height many parameters are used. Each pair of location and
    scale is initialized to produce mean equal to 0 and variance equal to 1 for its dimension. To allow for invertibility, the scale parameter 
    has to be non-zero and is therefore chosen to be on an exponential scale. Each dimension thus has the following activation 
    normalization:
    
    - y_hat = (x-l)/s, where s and l are the scale and location parameters for this dimension, respectively.

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
        self.__location__ = tf.Variable(tf.zeros(shape, dtype=tf.keras.backend.floatx()), trainable=True, name="__location__")
        """The value by which each data point shall be translated."""

        self.__scale__ = tf.Variable(tf.ones(shape, dtype=tf.keras.backend.floatx()), trainable=True, name="__scale__")
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
        x = utt.flatten_along_axes(x=x, axes=other_axes) # Shape == [product of all other axes, *self.__shape__]

        # Compute mean and standard deviation 
        mean = tf.stop_gradient(tf.math.reduce_mean(x, axis=0)) # Shape == self.__shape__ 
        standard_deviation = tf.stop_gradient(tf.math.reduce_std(x, axis=0)) # Shape == self.__shape__ 
        
        # Update attributes first call will have standardizing effect
        self.__location__.assign(mean)
        self.__scale__.assign(standard_deviation)

        # Update initialization state
        self.__is_initialized__ = True

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
        y_hat = (x - location) / scale

        # Outputs
        return y_hat
        
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:

        # Transform
        scale, location = self.__prepare_variables_for_computation__(x=y_hat)
        x =  y_hat * scale + location

        # Outputs
        return x
           
    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:

        # Count dimensions over remaining axes (for a single instance)
        batch_size = x.shape[0]
        dimension_count = 1
        for axis in range(1,len(x.shape)):
            if axis not in self.__axes__:
                dimension_count *= x.shape[axis] 
        
        # Compute logarithmic determinant
        # By defintion: sum across dimensions for ln(scale), where scale = exp(self.__scale__)
        # rewriting to: sum across dimensions for ln(exp(self.__scale__)) which results in:
        logarithmic_determinant = - dimension_count * tf.math.reduce_sum(tf.math.log(self.__scale__)) # single instance 
        logarithmic_determinant = tf.ones(shape=[batch_size], dtype=tf.keras.backend.floatx()) * logarithmic_determinant

        # Outputs
        return logarithmic_determinant

class Reflection(FlowLayer):
    """This layer reflects a data point around ``reflection_count`` learnable normals using the :ref:`Householder transform 
    <https://en.wikipedia.org/wiki/Householder_transformation>`: . In this context, the normal is the unit length vector orthogonal 
    to the hyperplane of reflection. When ``axes`` contains more than a single entry, the input is first flattened along these
    axes, then reflected and then unflattened to original shape.

    :param shape: See base class :class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :class:`FlowLayer`. IMPORTANT: These axes are distinct from the learnable reflection axes.
    :type axes: List[int]
    :param reflection_count: The number of successive reflections that shall be executed.
    :type reflection_count: int

    Referenes:

        - "Gaussianization Flows" by Chenlin Meng, Yang Song, Jiaming Song and Stefano Ermon
    """

    def __init__(self, shape: List[int], axes: List[int], reflection_count: int, **kwargs):
        # Super
        super(Reflection, self).__init__(shape=shape, axes=axes, **kwargs)

        # Attributes
        dimension_count = tf.reduce_prod(shape).numpy()
        reflection_normals = tf.math.l2_normalize(tf.random.uniform(shape=[reflection_count, dimension_count], dtype=tf.keras.backend.floatx()), axis=1) 
        
        self.__reflection_normals__ = tf.Variable(reflection_normals, trainable=True, name="reflection_normals") # name is needed for getting and setting weights
        """(:class:`tensorflow.Tensor`) - These are the axes along which an instance is reflected. Shape == [reflection count, dimension count] where dimension count is the product of the shape of the input instance along :py:attr:`self.__axes__`."""

        self.__inverse_mode__ = False
        "(bool) - Indicates whether the reflections shall be executed in reversed order (True) or forward order (False)."


    def __reflect__(self, x: tf.Tensor) -> tf.Tensor:
        """This function executes all the reflections of self in a sequence by multiplying ``x`` with the corresponding Householder 
            matrices that are constructed from :py:attr:`__reflection_normals__`. This method provides the backward reflection if 
            :py:attr:`self.__inverse_mode` == True and forward otherwise.

        :param x: The flattened data of shape [..., dimension count], where dimension count is the product of the :py:attr:`__shape__` as 
            specified during initialization of self. It is assumed that all axes except for :py:attr:`__axes__` (again, see 
            initialization of self) are moved to ... in the aforementioned shape of ``x``.
        :type x: :class:`tensorfflow.Tensor`
        :return: x_new (:class:`tensorfflow.Tensor`) - The rotated version of ``x`` with same shape.
        """

        # Convenience variables
        reflection_count = self.__reflection_normals__.shape[0]
        dimension_count = self.__reflection_normals__.shape[1]

        # Ensure reflection normal is of unit length 
        self.__reflection_normals__.assign(tf.math.l2_normalize(self.__reflection_normals__, axis=1))

        # Pass x through the sequence of reflections
        x_new = x
        indices = list(range(reflection_count))
        if self.__inverse_mode__: 
            # Note: Householder reflections are involutory (their own inverse) https://en.wikipedia.org/wiki/Householder_transformation
            # One can thus invert a sequence of refelctions by reversing the order of the individual reflections
            indices.reverse()

        for r in indices:
            v_r = self.__reflection_normals__[r][:, tf.newaxis] # Shape == [dimension count, 1]
            R = tf.eye(dimension_count, dtype=tf.keras.backend.floatx()) - 2.0 * v_r * tf.transpose(v_r, conjugate=True)
            x_new = tf.linalg.matvec(R, x_new)

        # Outputs
        return x_new
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        
        # Initialize
        old_shape = cp.copy(x.shape)

        # Flatten along self.__axes__ to fit reflection matrix
        x = utt.flatten_along_axes(x=x, axes=self.__axes__)

        # Move this flat axis to the end for multiplication with reflection matrices
        x = utt.move_axis(x=x, from_index=self.__axes__[0], to_index=-1)

        # Reflect
        y_hat = self.__reflect__(x=x)

        # Move axis back to where it came from
        y_hat = utt.move_axis(x=y_hat, from_index=-1, to_index=self.__axes__[0])

        # Unflatten to restore original shape
        y_hat = tf.reshape(y_hat, shape=old_shape)
        
        # Outputs
        return y_hat
    
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        
        # Prepare self for inversion
        previous_mode = self.__inverse_mode__
        self.__inverse_mode__ = True

        # Call forward method (will now function as inverter)
        x = self(x=y_hat)

        # Undo the setting of self to restore the method's precondition
        self.__inverse_mode__ = previous_mode

        # Outputs
        return x
    
    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:
        
        # It is known that Householder reflections have a determinant of -1 https://math.stackexchange.com/questions/504199/prove-that-the-determinant-of-a-householder-matrix-is-1
        # It is also known that det(AB) = det(A) det(B) https://proofwiki.org/wiki/Determinant_of_Matrix_Product
        # This layer applies succesive reflections as matrix multiplications and thus the determinant of the overall transformation is
        # -1 or 1, depending on whether an even or odd number of reflections are concatenated. Yet on logarithmic scale it is always 0.
        
        # Compute
        batch_size = x.shape[0]
        logarithmic_determinant = tf.zeros([batch_size], dtype=tf.keras.backend.floatx())

        # Outputs
        return logarithmic_determinant

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
        for layer in reversed(self.sequence): y_hat = layer.invert(y_hat=y_hat)
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

    def __init__(self, sequence: List[FlowLayer], factor_dimension_count: List[int], **kwargs):
        super().__init__(sequence=sequence, **kwargs)
        self.__loss__ = mls.SupervisedFactorLoss(factor_dimension_counts=factor_dimension_count)

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
        
        with tf.GradientTape() as tape:
            # First instance
            z_tilde_a = self(z_a, training=True)  # Forward pass
            j_a = self.compute_jacobian_determinant(x=z_a)
            
            # Second instance
            z_tilde_b = self(z_b, training=True)
            j_b = self.compute_jacobian_determinant(x=z_b)
            
            # Compute loss
            loss = self.__loss__.compute(y_true=y, y_pred=(z_tilde_a, z_tilde_b, j_a, j_b))

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Outputs
        return loss