import numpy as np
import tensorflow as tf
from typing import Any, Tuple, List, Callable
from abc import ABC
import abc
from gyoza.utilities import tensors as utt
import gyoza.modelling.masks as mms
import copy as cp
from gyoza.modelling import losses as mls
import random

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

        - `"Density estimation using Real NVP" by Laurent Dinh and Jascha Sohl-Dickstein and Samy Bengio. <https://arxiv.org/abs/1605.08803>`_
        - `"Glow: Generative Flow with Invertible 1x1 Convolutions" by Diederik P. Kingma and Prafulla Dhariwal. <https://arxiv.org/abs/1807.03039>`_
        - `"NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION" by Laurent Dinh, David Krueger and Yoshua Bengio <https://arxiv.org/abs/1410.8516>`_
        - `"GLOWin: A Flow-based Invertible Generative Framework for Learning Disentangled Feature Representations in Medical Images" by Aadhithya Sankar, Matthias Keicher, Rami Eisawy, Abhijeet Parida, Franz Pfister, Seong Tae Kim and  Nassir Navab <https://arxiv.org/abs/2103.10868>`_
        - `"Gaussianization Flows" by Chenlin Meng, Yang Song, Jiaming Song and Stefano Ermon <https://arxiv.org/abs/2003.01941>`_
        - `"A Disentangling Invertible Interpretation Network for Explaining Latent Representations" by Patrick Esser, Robin Rombach and Bjorn Ommer <https://arxiv.org/abs/2004.13166>`_
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
        """(:class:`List[int]`) - The shape of the input that shall be transformed by this layer. For detail, see constructor of :class:`FlowLayer`"""

        self.__axes__ = cp.copy(axes)
        """(:class:`List[int]`) - The axes of transformation. For detail, see constructor of :class:`FlowLayer`"""

    def fit(self, epoch_count: int, batch_count:int, X:tf.Tensor=None, Y: tf.Tensor=None, batch_size: int=None, iterator: Callable=None) -> tf.Tensor:
        """Fits self to data. Assumes that the model is compiled with loss and an optimizer. Unless overwritten by the subclass, this 
        fit method relies on a train_step method that passes an X batch through the model, flattens its output and after flattening 
        the Y batch, computes the loss to apply gradient descent. If ``X``, ``Y``, ``batch_size`` are specified, then no iterator has 
        to be specified. In that case, the fit method creates an iterator that samples ``batch_size`` instances from X and Y uniformly
        at random. Alternatively, ``X``, ``Y`` and ``batch_size`` can be omitted if an iterator is provided. 

        :param epoch_count: The number of times self shall be calibrated on `iterator`. 
        :type epoch_count: int
        :param batch_count: The number of times a new batch shall be drawn from the `iterator` per epoch. 
        :type batch_count: int
        :param X: Input data to be fed through the network. Shape is assumed to be [instance count, ...], where ... needs to be 
            compatible with the network.
        :type X: :class:`tensorflow.Tensor`, optional
        :param Y: Expected output data to be obtained after feeding ``X`` through the network. Shape is assumed to be [instance count, 
            ...], where ... needs to be compatible with the network and indexing is assumed to be synchronous with ``X``.
        :type Y: :class:`tensorflow.Tensor`, optional
        :param batch_size: The number of instance that shall be sampled per batch.
        :param iterator: An iterator that produces X, Y pairs of shape [batch_size, ...], where ... needs to be compatible with the 
            network. 
        :type iterator: :class:`tf.keras.utils.Sequential`
        :return: 
            - epoch_loss_means (:class:`tensorflow.Tensor`) - The mean loss per epoch. Length == [``epoch_count``].
            - epoch_loss_standard_deviations (:class:`tensorflow.Tensor`) - The standard_deviation of loss per epoch. Length == [``epoch_count``].
        """
        
        # Input validity
        if type(X) != type(None) or type(Y) != type(None) or type(batch_size) != type(None):
            assert type(X) != type(None) and type(Y) != type(None) and type(batch_size) != type(None) and type(iterator) == type(None), f"If X, Y or batch size are provided, then all need to be provided while iterator shall be none. "
        else:
            assert type(iterator) != type(None), f"If neither X, Y, nor batch_size are specified, then an iterator must be provided."

        # Initialization
        epoch_loss_means = [None] * epoch_count
        epoch_loss_standard_deviations = [None] * epoch_count
        batch_losses = [None] * batch_count
        if type(iterator) == type(None):
            def iterate(X,Y, batch_size):
                instance_count = X.shape[0]
                while True:
                    indices = np.random.randint(0,instance_count,size=[batch_size])
                    yield X[indices], Y[indices]
            iterator = iterate(X=X, Y=Y, batch_size=batch_size)
        
        # Iterate epochs
        for e in range(epoch_count):
            
            # Iterate batches
            for b in range(batch_count): batch_losses[b] = self.train_step(data=next(iterator)).numpy()
            epoch_loss_means[e] = np.mean(batch_losses)
            epoch_loss_standard_deviations[e] = np.std(batch_losses)

        # Outputs
        return epoch_loss_means, epoch_loss_standard_deviations

    def train_step(self, data):
        """Performs one step of gradient descent. Assumes self.optimzer and self.loss are initialized.

        :param data: A tuple containg the batch of X and Y, respectively. X is assumed to be a tensorflow.Tensor of shape [batch size,
            ...] where ... is the shape of one input instance that has to fit through :py:attr:`self.sequence`. The tensorflow.Tensor
            Y shall be of same shape of X.
        :type data: Tuple(tensorflow.Tensor, tensorflow.Tensor)
        :return: loss: A scalar for the loss observed before applying the train step.
        """
        
        # Unpack inputs
        X, Y = data
        
        with tf.GradientTape() as tape:
            # Predict
            Y_hat = self(X, training=True)  # Forward pass
            
            # Flatten to apply loss (most keras losses expect flat inputs)
            Y_flat = tf.reshape(Y, shape=[-1])
            Y_hat_flat = tf.reshape(Y_hat, shape=[-1])
            
            # Compute loss
            loss = self.loss(y_true=Y_flat, y_pred=Y_hat_flat)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Outputs
        return loss

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

class Permutation(FlowLayer):
    """This layer flattens its input :math:`x` along ``axes``, then reorders the dimensions using ``permutation`` and reshapes 
    :math:`x` to its original shape. 

    :param shape: See base class :class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :class:`FlowLayer`.
    :type axes: List[int]
    :param permutation: A new order of the indices in the interval [0, product(``shape``)).
    :type permutation: List[int]
    """

    def __init__(self, shape: List[int], axes: List[int], permutation: List[int], **kwargs):

        # Input validity
        dimension_count = tf.reduce_prod(shape).numpy()
        assert len(permutation) == dimension_count, f'The input permutation was expected to have length {dimension_count} based on the number of dimensions in the shape input but it was found to have length {len(permutation)}.'

        # Super
        super(Permutation, self).__init__(shape=shape, axes=axes, **kwargs)
    
        # Attributes
        permutation = tf.constant(permutation)
        self.__forward_permutation__ = tf.Variable(permutation, trainable=False, name="forward_permutation") # name is needed for getting and setting weights
        """(:class:`tensorflow.Variable`) - Stores the permutation vector for the forward operation."""
        
        self.__inverse_permutation__ = tf.Variable(tf.argsort(permutation), trainable=False, name="inverse_permutation")
        """(:class:`tensorflow.Variable`) - Stores the permutation vector for the inverse operation."""

    def call(self, x: tf.Tensor) -> tf.Tensor:
        
        # Initialize
        old_shape = cp.copy(x.shape)

        # Flatten along self.__axes__ to fit permutation vector
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

class Shuffle(Permutation):
    """Shuffles input :math:`x`. The permutation used for shuffling is randomly chosen once during initialization. 
    Thereafter it is saved as a private attribute. Shuffling is thus deterministic. **IMPORTANT:** The shuffle function is defined on 
    a vector, yet by the requirement of :class:`Permutation`, inputs :math:`x` to this layer are allowed to have more than one axis 
    in ``axes``. As described in :class:`Permutation`, an input :math:`x` is first flattened along ``axes`` and thus the shuffling can
    be applied. For background information see :class:`Permutation`.
    
    :param shape: See base class :class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :class:`FlowLayer`.
    :type axes: List[int]
    """

    def __init__(self, shape: List[int], axes: List[int], **kwargs):

        # Super
        dimension_count = tf.reduce_prod(shape).numpy()
        permutation = list(range(dimension_count)); random.shuffle(permutation)
        super(Shuffle, self).__init__(shape=shape, axes=axes, permutation=permutation, **kwargs)
    
class Heaviside(Permutation):
    """Swops the first and second half of input :math:`x` as inspired by the `Heaviside 
    <https://en.wikipedia.org/wiki/Heaviside_step_function>`_ function.  **IMPORTANT:** The Heaviside function is defined on a vector, 
    yet by the requirement of :class:`Permutation`, inputs :math:`x` to this layer are allowed to have more than one axis in ``axes``.
    As described in :class:`Permutation`, an input :math:`x` is first flattened along ``axes`` and thus the swopping can be applied. 
    For background information see :class:`Permutation`.

    :param shape: See base class :class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :class:`FlowLayer`.
    :type axes: List[int]
    """

    def __init__(self, shape: List[int], axes: List[int], **kwargs):

        # Super
        dimension_count = tf.reduce_prod(shape).numpy()
        permutation = list(range(dimension_count//2, dimension_count)) + list(range(dimension_count//2))
        super(Heaviside, self).__init__(shape=shape, axes=axes, permutation=permutation, **kwargs)
 
class CheckerBoard(Permutation):
    """Swops the entries of inputs :math:`x` as inspired by the `checkerboard <https://en.wikipedia.org/wiki/Check_(pattern)>`_
    pattern. Swopping is done to preserve adjacency of cells within :math:`x`. **IMPORTANT:** The checkerboard pattern is usually
    defined on a matrix, i.e. 2 axes. Yet, here it is possible to specify any number of axes.

    :param axes: See base class :class:`FlowLayer`.
    :type axes: :class:`List[int]`
    :param shape: See base class :class:`FlowLayer`. 
    :type shape: :class:`List[int]`
    """

    @staticmethod
    def is_end_of_axis(index: int, limit: int, direction: int) -> bool:
        """Determines whether an ``index`` iterated in ``direction`` is at the end of a given axis.

        :param index: The index to be checked.
        :type index: int
        :param limit: The number of elements along the axis. An index is considered to be at the end if it is equal to ``limit``-1 
            and ``direction`` == 1. 
        :type limit: int
        :param direction: The direction in which the index is iterated. A value of 1 indicates incremental, -1 indicates decremental.
        :type direction: int
        :return: (bool) - An indicator for whether the endpoint has been reached.
        """
        if direction == 1: # Incremental
            return index == limit -1
        else: # Decremental
            return index == 0

    @staticmethod
    def generate_rope_indices(shape: List[int]) -> List[int]:
        """Generates indices to traverse a tensor of ``shape``. The traversal follows a rope fitted along the axes by prioritizing
        later axes before earlier axes.

        :param shape: The shape of the tensor to be traversed.
        :type shape: :class:`List[int]`
        :yield: current_indices (:class:`List[int]`) - The indices pointing to the current cell in the tensor. It provides one index
            along each axis of ``shape``.
        """
        dimension_count = np.product(shape)
        current_indices = [0] * len(shape)
        yield current_indices
        directions = [1] * len(shape)
        for d in range(dimension_count):
            # Increment index counter (with carry on to next axes if needed)
            for s in range(len(shape)-1,-1,-1): 
                if CheckerBoard.is_end_of_axis(index=current_indices[s], limit=shape[s], direction=directions[s]):
                    directions[s] = -directions[s]
                else:
                    current_indices[s] += directions[s]
                    break

            yield current_indices

    def __init__(self, shape: List[int], axes: List[int], **kwargs):

        # Set up permutation vector
        dimension_count = np.product(shape)
        tensor = np.reshape(np.arange(dimension_count), shape)
        rope_values = [None] * dimension_count
        
        # Unravel tensor
        rope_index_generator = CheckerBoard.generate_rope_indices(shape=shape)
        for d in range(dimension_count): rope_values[d] = tensor[tuple(next(rope_index_generator))]

        # Swop every two adjacent values
        for d in range(0, 2*(dimension_count//2), 2):
            tmp = rope_values[d]
            rope_values[d] = rope_values[d+1]
            rope_values[d+1] = tmp

        # Ravel tensor
        rope_index_generator = CheckerBoard.generate_rope_indices(shape=shape)
        for d in range(dimension_count): tensor[tuple(next(rope_index_generator))] = rope_values[d]

        # Flattened tensor now gives permutation
        permutation = list(np.reshape(tensor, [-1]))

        # Super
        super(CheckerBoard, self).__init__(shape=shape, axes=axes, permutation=permutation, **kwargs)
    
class Coupling(FlowLayer, ABC):
    r"""This layer couples the input :math:`x` with itself inside the method :py:meth:`call` by implementing the following formulae:
    
    .. math::
        :nowrap:

        \begin{eqnarray}
            x_1 & = w * x \\
            x_2 & = (1-w) * x \\
            y_1 & = x_1 \\
            y_2 & = f(x_2, g(x_1)) \\
            y   & = y_1 + y_2,
        \end{eqnarray}

    with ``mask`` :math:`w`, function :py:meth:`compute_coupling_parameters` :math:`g` and coupling law :math:`f`. As can be seen 
    from the formula, the ``mask`` :math:`w` is used to select half of the input :math:`x` in :math:`x_1` and the other half in 
    :math:`x_2`. While :math:`y_1` is set equal to :math:`x_1`, the main contribution of this layer is in the computation of 
    :math:`y_2`. That is, the coupling law :math:`f` computes :math:`y_2` as a trivial combination, e.g. sum or product of :math:`x_2` 
    and coupling parameters :math:`g(x_1)`. The function :math:`g` is a model of arbitrary complexity and it is thus possible to 
    create non-linear mappings from :math:`x` to :math:`y`. The coupling law :math:`f` is chosen by this layer to be trivially 
    invertible and to have tractable Jacobian determinant which ensures that the overall layer also has these two properties.

    :param shape: See base class :class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :class:`FlowLayer`.
    :type axes: List[int]
    :param compute_coupling_parameters: See the placeholder member :py:meth:`compute_coupling_parameters` for a detailed description 
        of requirements.
    :type compute_coupling_parameters: :class:`tensorflow.keras.Model`
    :param mask: The mask used to select one half of the data while discarding the other half.
    :type mask: :class:`gyoza.modelling.masks.Mask`
    
    References:

        - `"NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION" by Laurent Dinh and David Krueger and Yoshua Bengio. <https://arxiv.org/abs/1410.8516>`_
        - `"Density estimation using Real NVP" by Laurent Dinh and Jascha Sohl-Dickstein and Samy Bengio. <https://arxiv.org/abs/1605.08803>`_
    """

    def __init__(self, shape: List[int], axes: List[int], compute_coupling_parameters: tf.keras.Model, mask: mms.Mask, **kwargs):

        # Super
        super(Coupling, self).__init__(shape=shape, axes=axes, **kwargs)

        # Input validity
        shape_message = f"The shape ({shape}) provided to the coupling layer and that provided to the mask ({mask.__mask__.shape}) are expected to be the same."
        assert len(shape) == len(mask.__shape__), shape_message
        for i in range(len(shape)):
            assert shape[i] == mask.__shape__[i], shape_message

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
        :return: y_hat (:class:`tensorflow.Tensor`) - The transformed version of ``x``. It's shape must support the Hadamard product with ``x``."""
        
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
    """This coupling layer implements an additive coupling law of the form :math:`f(x_2, c(x_1) = x_2 + c(x_1)`. For details on the
    encapsulating theory refer to :class:`Coupling`.
    
    References:

        - `"Density estimation using Real NVP" by Laurent Dinh and Jascha Sohl-Dickstein and Samy Bengio. <https://arxiv.org/abs/1605.08803>`_
    """

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
    """This coupling layer implements an affine coupling law of the form :math:`f(x_2, c(x_1) = e^s x_2 + t`, where :math:`s, t = c(x)`. 
    To prevent division by zero during decoupling, the exponent of :math:`s` is used as scale. For details on the encapsulating 
    theory refer to :class:`Coupling`. 
    
    References:

        - `"Density estimation using Real NVP" by Laurent Dinh and Jascha Sohl-Dickstein and Samy Bengio. <https://arxiv.org/abs/1605.08803>`_
    """

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
    """A trainable location and scale transformation of the data. For each dimension of the specified input shape, a scale and a 
    location parameter is used. That is, if shape == [width, height], then 2 * width * height many parameters are used. Each pair of 
    location and scale is initialized to produce mean equal to 0 and variance equal to 1 for its dimension. To allow for 
    invertibility, the scale parameter is constrained to be non-zero. To simplifiy computation of the jacobian determinant on 
    logarithmic scale, the scale parameter is here constrained to be positive. Each dimension has the following activation 
    normalization:
    
    - y_hat = (x-l)/s, where s > 0 and l are the scale and location parameters for this dimension, respectively.

    :param shape: See base class :class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :class:`FlowLayer`.
    :type axes: List[int]
    
    References:

    - `"Glow: Generative Flow with Invertible 1x1 Convolutions" by Diederik P. Kingma and Prafulla Dhariwal. <https://arxiv.org/abs/1807.03039>`_
    - `"A Disentangling Invertible Interpretation Network for Explaining Latent Representations" by Patrick Esser, Robin Rombach and Bjorn Ommer. <https://arxiv.org/abs/2004.13166>`_
    """
    
    def __init__(self, shape: List[int], axes: List[int], **kwargs):

        # Super
        super(ActivationNormalization, self).__init__(shape=shape, axes=axes)
        
        # Attributes
        self.__location__ = tf.Variable(tf.zeros(shape, dtype=tf.keras.backend.floatx()), trainable=True, name="__location__")
        """The value by which each data point shall be translated."""

        self.__scale__ = tf.Variable(tf.ones(shape, dtype=tf.keras.backend.floatx()), trainable=True, name="__scale__", constraint=lambda x: tf.clip_by_value(x, clip_value_min=1e-6, clip_value_max=x.dtype.max))
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
        """Prepares the variables for computation with data. This ensures variable shapes are compatible with ``x``.
        
        :param x: Data to be passed through :py:meth:`call`. It's shape must agree with input ``x`` of 
            :py:meth:`self.__reshape_variables__`.
        :type x: :class:`tensorflow.Tensor`

        :return: 
            - location (tensorflow.Variable) - The :py:attr:`__location__` attribute shaped to fit ``x``. 
            - scale (tensorflow.Variable) - The :py:attr:`__scale__` attribute shaped to fit ``x``."""

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
        y_hat = (x - location) / scale # Scale is positive due to constraint

        # Outputs
        return y_hat
        
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:

        # Transform
        location, scale = self.__prepare_variables_for_computation__(x=y_hat)
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
        # By defintion: sum across dimensions for ln(scale)
        logarithmic_determinant = - dimension_count * tf.math.reduce_sum(tf.math.log(self.__scale__)) # single instance, scale is positive due to constraint, the - sign in front is because the scale is used in the denominator
        logarithmic_determinant = tf.ones(shape=[batch_size], dtype=tf.keras.backend.floatx()) * logarithmic_determinant

        # Outputs
        return logarithmic_determinant

class Reflection(FlowLayer):
    """This layer reflects a data point around ``reflection_count`` learnable normals using the `Householder transform 
    <https://en.wikipedia.org/wiki/Householder_transformation>`_. In this context, the normal is the unit length vector orthogonal to
    the hyperplane of reflection. When ``axes`` contains more than a single entry, the input is first flattened along these axes, 
    then reflected and then unflattened to original shape.

    :param shape: See base class :class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :class:`FlowLayer`. **IMPORTANT**: These axes are distinct from the learnable reflection axes.
    :type axes: List[int]
    :param reflection_count: The number of successive reflections that shall be executed. Expected to be at least 1.
    :type reflection_count: int

    Referenes:

        - `"Gaussianization Flows" by Chenlin Meng, Yang Song, Jiaming Song and Stefano Ermon <https://arxiv.org/abs/2003.01941>`_
    """

    def __init__(self, shape: List[int], axes: List[int], reflection_count: int, **kwargs):
        # Input validity
        assert 1 <= reflection_count, f'The input reflection_count was expected to be at least 1 but found to be {reflection_count}.'
        
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
        """(List[:class:`FlowLayer`]) - Stores the sequence of flow layers through which the data shall be passed."""

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
    """This network is a :class:`SequentialFlowNetwork` that can be used to disentangle factors, e.g. to understand representations
    in latent spaces of regular neural networks. It automatically uses the :class:`losses.SupervisedFactorLoss` to compute its losses.
    It also overrides the :class:`FlowLayer`'s implementation for train_step to accomodate for the fact that calibration does not
    simply use single instances but pairs of instances and their similarity.
    
    :param sigma: A measure of how tight clusters in the output space shall be. It is used to set up the factorized loss.
    :type sigma: float

    References:

       - `"A Disentangling Invertible Interpretation Network for Explaining Latent Representations" by Patrick Esser, Robin Rombach and Bjorn Ommer <https://arxiv.org/abs/2004.13166>`_
    """
    
    def __init__(self, sequence: List[FlowLayer], dimensions_per_factor: List[int], sigma: float = 0.975, **kwargs):
        super().__init__(sequence=sequence, **kwargs)
        self.__dimensions_per_factor__ = cp.copy(dimensions_per_factor) 
        self.__sigma__ = sigma
        """(List[int]) - A list that indicates for each factor (matched by index) how many dimensions are used."""

    @staticmethod
    def estimate_factor_dimensionalities(Z_ab: np.ndarray, Y_ab: np.ndarray) -> List[int]:
        """Estimates the dimensionality of each factor and thus helps to use the constructor of this class. Internally, for each 
        factor the instance pairs are selected such they represent a similar characteristic along that factor. The correlation of 
        instance pairs is then obtained for each dimension. For a given factor, the sum of these correlations (relative to the 
        overall sum) determines the number of dimensions. **Important:** If the factors of this model are categorical, it is 
        covnenient to use this function with with regular training inputs ``X_ab``, ``Y_ab`` but such that instance pairs with a 
        zero row in ``Y_ab`` are filtered out for efficiency. If there are quantitative factors, then the caller needs to ensure 
        that their ``Y_ab`` is still binary, e.g. by discretizing the quantiative factors during computation of ``Y_ab``.
        
        :param Z_ab: A sample of input instances, arranged in pairs. These instances shall be drawn from the same propoulation as 
            the inputs to this flow model during inference, yet flattened. Shape == [instance count, 2, dimension count], where 2 
            is due to pairing. 
        :type Z_ab: :class:`numpy.ndarray`
        :param Y_ab: The factor-wise similarity of instances in each pair of ``Z_ab``. **IMPORTANT:** Here, it is assumed that the 
            residual factor is at index 0 AND that the values of ``Y_ab`` are either 0 or 1. Shape == [instance count, factor count].
        :type Y_ab: :class:`numpy.ndarray`

        :return:
            - dimensions_per_factor (List[int]) - The number of dimensions per factor (including the residual factor), summing up to the dimensionality of ``Z``. Ordering is the same is in ``Y_ab``.
        """

        # Input validity 
        assert len(Z_ab.shape) == 3, f"The input Z_ab was expected to have shape [instance count, 2, dimension count], but has shape {Z_ab.shape}."
        assert len(Y_ab.shape) == 2, f"The input Y_ab was expected to have shape [instance count, factor count], including the residual factor, but has shape {Y_ab.shape}."
        assert Z_ab.shape[0] == Y_ab.shape[0], f"The inputs Z_ab and Y_ab were expected to have the same number of instances along the 0th axis, but have shape {Z_ab.shape}, {Y_ab.shape}, respectively."

        # Iterate factors
        instance_count, _, dimension_count = Z_ab.shape
        factor_count = Y_ab.shape[1]
        S = [None] * factor_count # Raw dimension counts per factor (Equation 11 of reference paper)
        S[0] = dimension_count # Ensures equal contribtion of the residual factor if all other factors are represented in Z
        for f in range(1, factor_count): # Residual factor at index 0 is already covered
            # Select only the instances that have the same class along this factor
            Z_ab_similar = Z_ab[Y_ab[:,f] == 1,:]
            
            # Compute correlation between pairs for each dimension (Equation 11 of reference paper)
            S[f] = 0
            for d in range(dimension_count): 
                S[f] += np.corrcoef(Z_ab_similar[:,0,d], Z_ab_similar[:,1,d])[0,1] # corrcoef gives 2x2 matrix. [0,1] selects the correlation of interest. 

        # Rescale S to make its entries add up to dimension_count
        N = np.exp(S)
        N = N / np.sum(N) * dimension_count
        N = np.floor(N) # Get integer dimension counts. N might not add up to dimension_count at this point
        N[0] += dimension_count - np.sum(N) # Move spare dimensions to residual factor to ensure sum(N) == dimension_count
        
        # Format
        dimensions_per_factor = list(np.array(N, dtype=np.int32))

        # Outputs
        return list(dimensions_per_factor)

    def train_step(self, data):
        """Performs one step of gradient descent. Assumes self.optimzer and self.loss are initialized.

        :param data: A tuple containg the batch of X and Y, respectively. X is assumed to be a tensorflow.Tensor of shape [batch size,
            2, ...] where 2 indicates the pair x_a, x_b of same factor and ... is the shape of one input instance that has to fit 
            through :py:attr:`self.sequence`. The tensorflow.Tensor Y shall contain the factor indices of shape [batch size].
        :type data: Tuple(tensorflow.Tensor, tensorflow.Tensor)
        :return: loss: A scalar for the loss observed before applying the train step.
        """
        
        # Ensure loss exists
        if not hasattr(self, "loss") or type(self.loss) != mls.SupervisedFactorLoss:
            self.loss = mls.SupervisedFactorLoss(dimensions_per_factor=self.__dimensions_per_factor__, sigma=self.__sigma__); del self.__sigma__

        # Unpack inputs
        X, Y = data
        z_a = X[:,0,:]; z_b = X[:,1,:]
        
        with tf.GradientTape() as tape:
            # First instance
            z_tilde_a = self(z_a, training=True)  # Forward pass
            j_a = self.compute_jacobian_determinant(x=z_a)
            
            # Second instance
            z_tilde_b = self(z_b, training=True)
            j_b = self.compute_jacobian_determinant(x=z_b)
            
            # Compute loss
            loss = self.loss(y_true=Y, y_pred=(z_tilde_a, z_tilde_b, j_a, j_b))

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Outputs
        return loss