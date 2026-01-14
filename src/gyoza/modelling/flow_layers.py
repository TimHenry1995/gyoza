import numpy as np
import tensorflow as tf
from typing import Any, Tuple, List, Callable, Generator
from abc import ABC
import abc
from gyoza.utilities import tensors as utt
import gyoza.modelling.masks as mms
import copy as cp
from gyoza.modelling import losses as mls
import random

class FlowModel(tf.keras.Model):
    """A class for flow models. It subclasses :py:class:`tensorflow.keras.Model` and assumes a list of flow-layers as input
    which will always be executed in the given sequence.
    
    :param flow_layers: A list of flow layers.
    :type flow_layers: List[:py:class:`FlowLayer`]"""


    def __init__(self, flow_layers):
        
        # Super
        super().__init__()
        
        # Input validity
        assert all([isinstance(layer, FlowLayer) for layer in flow_layers]), f"The input `layers` provided to the FlowModel needs to be an array of FlowLayers but was {[type(layer) for layer in flow_layers]}."
        self.flow_layers = flow_layers
    
    def build(self, input_shape):
        # Call to super
        super().build(input_shape=input_shape)
        
        # Iterate flow layers 
        inputs = tf.keras.Input(shape=input_shape[1:])
        for layer in self.flow_layers:
            
            # Build layer
            if not layer.built: layer.build(input_shape=tf.keras.ops.shape(inputs))

            # Update input shape for next layer
            inputs, _ = layer(inputs=inputs)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Executes the operation of this layer in the forward direction.

        :param inputs: The data to be tranformed. Assumed to be of shape [batch size, ...].
        :type inputs: :py:class:`tensorflow.Tensor`
        :return: y_hat (:py:class:`tensorflow.Tensor`) - The output of the transformation of shape [batch size, ...]."""

        # Transform
        logarithmic_determinant = 0.0 * tf.keras.ops.sum(inputs, axis=list(range(1, len(tf.keras.ops.shape(inputs)))))
        y_hat = inputs
        for layer in self.flow_layers:
            y_hat, logarithmic_determinant_l = layer(inputs=y_hat)
            logarithmic_determinant += logarithmic_determinant_l

        # Outputs
        return y_hat, logarithmic_determinant
    
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        """Executes the operation of this layer in the inverse direction. It is thus the counterpart to :py:meth:`call`.

        :param y_hat: The data to be transformed. Assumed to be of shape [batch size, ...].
        :type y_hat: :py:class:`tensorflow.Tensor`
        :return: x (:py:class:`tensorflow.Tensor`) - The output of the transformation of shape [batch size, ...]."""        

        # Transform
        for layer in reversed(self.flow_layers): y_hat = layer.invert(y_hat=y_hat)
        x = y_hat

        # Outputs
        return x

    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:
        """Computes the Jacobian determinant of this model's :py:meth:`call` at the input `x` on a logarithmic scale. The
        natural logarithm is chosen for numerical stability. The layer's :py:meth:`FlowLayer.call` method is executed on
        input x before the Jacobian determinant is computed to allow the layer's internal variables to be set which
        in turn allows the determinant computation to ignore the input `x` and focus on the layer's variables if needed.

        :param x: The data at which the determinant shall be computed. Assumed to be of shape [batch size, ...].
        :type x: :py:class:`tensorflow.Tensor`
        :return: logarithmic_determinant (:py:class:`tensorflow.Tensor`) - A measure of how much this layer contracts or dilates space at the point ``x``. Shape == [batch size].
        """ 

        # Transform
        logarithmic_determinant = 0.0 * tf.keras.ops.sum(x, axis=list(range(1, len(x.shape))))
        for layer in self.flow_layers: 
            x_tmp = layer(x=x) # Do this before the determinant is computed to ensure the layer's internal variables are set correctly
            logarithmic_determinant += layer.compute_jacobian_determinant(x=x) 
            x = x_tmp
            
        # Outputs
        return logarithmic_determinant
    '''
    def forward(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Executes the :py:meth:`.FlowLayer.forward` method of each layer from :py:attr:`.FlowModel.flow_layers`. 

        :param x: The data to be tranformed. Assumed to be of shape [batch size, ...].
        :type x: :py:class:`tensorflow.Tensor`
        :return: (y_hat, logarithmic_determinant) - The output of the transformation and the logarithmic jacobian determinant. Shape of y_hat == [batch size, ...], shape of logarithmic_determinant == [batch size]."""

        # Transform
        logarithmic_determinant = 0.0 * tf.keras.ops.sum(x, axis=list(range(1, len(x.shape))))
        y_hat = x
        for layer in self.flow_layers:
            y_hat, logarithmic_determinant_l = layer.forward(x=y_hat)
            logarithmic_determinant += logarithmic_determinant_l

        # Outputs
        return y_hat, logarithmic_determinant'''
    
    def get_config(self):
        # 1. Get the base config (handles name, trainable, etc.)
        config = super().get_config()
        
        # 2. Serialize the layers using Keras's built-in utility
        # This converts the Layer objects into a format (JSON-safe) Keras understands
        config.update({
            "flow_layers": [tf.keras.layers.serialize(layer) for layer in self.flow_layers]
        })
        return config

    @classmethod
    def from_config(cls, config):
        # This is the 'undo' button for get_config
        # It turns the serialized JSON back into actual Layer objects
        layers_config = config.pop("flow_layers")
        flow_layers = [tf.keras.layers.deserialize(l) for l in layers_config]
        return cls(flow_layers=flow_layers, **config)

class DisentanglingFlowModel(FlowModel):
    """This network is a :py:class:`FlowModel` that can be used to disentangle factors, e.g. to understand representations
    in latent spaces of regular neural networks. The model can be used on single instances for inference via :py:meth:`DisentanglingFloWModel.call`
    and :py:meth:`DisentanglingFloWModel.forward`. To train the model, the regular :py:meth:`tensorflow.keras.Model.fit` can be used whereby inputs
    need to be provided as a tuple (Z_a, Z_b) drawn from the joint distribution that aligns with the factorized correlations specified in Y_ab.
    Z_a and Z_b shall be of shape [batch size, ...] and Y_ab of shape [batch size, factor count]. Here, the first factor is always assumed to be the residual factor. 
    The :py:class:`DisentanglingFlowModel` overrides the :py:class:`FlowModel`'s implementation for train_step to accomodate for the fact that calibration does not
    simply use single instances but pairs of instances and their similarity. It also automatically uses the :py:class:`losses.SupervisedFactorLoss` to compute 
    deviations from optimality during fitting.

    :param flow_layers: A list of flow layers.
    :type flow_layers: List[:py:class:`FlowLayer`]

    References:

       - `"A Disentangling Invertible Interpretation Network for Explaining Latent Representations" by Patrick Esser, Robin Rombach and Bjorn Ommer <https://arxiv.org/abs/2004.13166>`_
    """
    
    def __init__(self, flow_layers: List["FlowLayer"]):#, dimensions_per_factor: List[int]):
        super().__init__(flow_layers)
        #self._dimensions_per_factor_ = cp.copy(dimensions_per_factor)
        #"""(List[int]) - A list that indicates for each factor (matched by index) how many dimensions are used."""
    
    def train_step(self, data):
        # Data comes from the iterator: (X_pair, Y_true)
        # X_pair is expected to be (X_a, X_b)
        (Z_a, Z_b), Z_ab = data

        with tf.GradientTape() as tape:
            # 1. Use your "Nuclear Option" logic inside the tape
            y_a, j_a = self(Z_a)
            y_b, j_b = self(Z_b)

            # 2. Reconstruct the y_pred structure expected by your loss
            y_pred = tf.concat([
                y_a, y_b, 
                tf.reshape(j_a, (-1, 1)), 
                tf.reshape(j_b, (-1, 1))
            ], axis=-1)

            # 3. Compute the loss
            loss = self.compute_loss(x=None, y=Z_ab, y_pred=y_pred)

        # 4. Compute gradients and update
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}
    
class FlowLayer(tf.keras.layers.Layer):
    """Abstract base class for flow layers. Any input to this layer is assumed to have ``shape`` along ``axes`` as specified during
    initialization.
    
    :param shape: The shape of the input that shall be transformed by this layer. If you have e.g. a tensor [batch size, width, 
        height, color] and you want this layer to transform along width and height, you enter the shape [width, height]. If you 
        want the layer to operate on the color you provide the shape [color] instead.
    :type shape: List[int]
    :param axes: The axes of transformation. In the example for ``shape`` on width and height you would enter [1,2] here, In the 
        example for color you would enter [3] here. Although axes are counted starting from zero, it is assumed that ``axes`` 
        does not contain the axis 0, i.e. the batch axis. Furthermore, axes are assumed to be ascending but they do not have to be contiguous.
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
        self._shape_ = cp.copy(shape)
        """(:py:class:`List[int]`) - The shape of the input that shall be transformed by this layer. For detail, see constructor of :py:class:`FlowLayer`"""

        self._axes_ = cp.copy(axes)
        """(:py:class:`List[int]`) - The axes of transformation. For detail, see constructor of :py:class:`FlowLayer`"""

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Executes the operation of this layer in the forward direction.

        :param inputs: The data to be tranformed. Assumed to be of shape [batch size, ...].
        :type inputs: :py:class:`tensorflow.Tensor`
        :return: y_hat (:py:class:`tensorflow.Tensor`) - The output of the transformation of shape [batch size, ...]."""        
        raise NotImplementedError()

    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        """Executes the operation of this layer in the inverse direction. It is thus the counterpart to :py:meth:`call`.

        :param y_hat: The data to be transformed. Assumed to be of shape [batch size, ...].
        :type y_hat: :py:class:`tensorflow.Tensor`
        :return: x (:py:class:`tensorflow.Tensor`) - The output of the transformation of shape [batch size, ...]."""        

        raise NotImplementedError()

    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:
        """Computes the jacobian determinant of this layer's :py:meth:`call` at the input on a logarithmic scale. The
        natural logarithm is chosen for numerical stability. The subclassing layer can assume that the :py:meth:`.FlowLayer.call`
        function was called on `x` right before :py:meth:`.FlowLayer.compute_jacobian_determinant` is called. Hence,
        if the call method adjust the layer's parameters, the compute_jacobian_determinant method can rely on the variables
        being in set in accordance with `x`. **Important:** compute_jacobian_determinant is expected to work on eager tensors
        as well as symbolic tensors. 

        :param x: The data at which the determinant shall be computed. Assumed to be of shape [batch size, ...].
        :type x: :py:class:`tensorflow.Tensor`
        :return: logarithmic_determinant (:py:class:`tensorflow.Tensor`) - A measure of how much this layer contracts or dilates space at the point ``x``. Shape == [batch size].
        """        

        raise NotImplementedError()
    '''
    def forward(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Executes the :py:meth:`.FlowLayer.call` and the :py:meth:`.FlowLayer.compute_jacobian_determinant` methods of self. 
        The determinant is computed **at** the layer's input **after** the layer has processed its input, thus allowing any input-related variables to be updated in call 
        before the determinant is computed.

        :param x: The data to be tranformed. Assumed to be of shape [batch size, ...].
        :type x: :py:class:`tensorflow.Tensor`
        :return: (y_hat, logarithmic_determinant) - The output of the transformation and the logarithmic jacobian determinant. Shape of y_hat == [batch size, ...], shape of logarithmic_determinant == [batch size]."""

        # Transform
        y_hat = self(x)
        logarithmic_determinant = self.compute_jacobian_determinant(x)
        
        # Outputs
        return y_hat, logarithmic_determinant'''

class Permutation(FlowLayer):
    """This layer flattens its input :math:`x` along ``axes``, then reorders the dimensions using ``permutation`` and reshapes 
    :math:`x` to its original shape. It is volume-preserving, meaning it has a Jacobian determinant of 1 (or 0 on logarithmic scale).

    :param shape: See base class :py:class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :py:class:`FlowLayer`.
    :type axes: List[int]
    :param permutation: A new order of the indices in the interval [0, product(``shape``)).
    :type permutation: List[int]
    """

    def __init__(self, shape: List[int], axes: List[int], permutation: List[int], **kwargs):

        # Input validity
        dimension_count = int(np.prod(shape))
        assert len(permutation) == dimension_count, f'The input permutation was expected to have length {dimension_count} based on the number of dimensions in the shape input but it was found to have length {len(permutation)}.'

        # Super
        super(Permutation, self).__init__(shape=shape, axes=axes, **kwargs)
    
        # Attributes
        permutation = tf.constant(permutation)
        self._forward_permutation_ = self.add_weight(shape = permutation.shape,
                                                       initializer = tf.keras.initializers.Constant(permutation.numpy()),
                                                       dtype = permutation.dtype,
                                                       trainable = False,
                                                       name="forward_permutation") # name is needed for getting and setting weights
        """(:py:class:`tensorflow.Variable`) - Stores the permutation vector for the forward operation."""
        
        self._inverse_permutation_ = self.add_weight(shape = permutation.shape,
                                                       initializer = tf.keras.initializers.Constant(tf.argsort(permutation).numpy()), 
                                                       dtype = permutation.dtype,
                                                       trainable=False, 
                                                       name="inverse_permutation")
        """(:py:class:`tensorflow.Variable`) - Stores the permutation vector for the inverse operation."""

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        
        # Initialize
        old_shape = list(tf.keras.ops.shape(inputs))

        # Flatten along self._axes_ to fit permutation vector
        inputs = utt.flatten_along_axes(x=inputs, axes=self._axes_)

        # Shuffle
        y_hat = tf.gather(inputs, self._forward_permutation_, axis=self._axes_[0])

        # Unflatten to restore original shape
        y_hat = tf.keras.ops.reshape(y_hat, newshape=old_shape)
        
        # Compute jacobian determinant
        jacobian_determinant = self.compute_jacobian_determinant(x=inputs)

        # Outputs
        return y_hat, jacobian_determinant
    
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        
        # Initialize
        old_shape = tf.keras.ops.shape(y_hat)

        # Flatten along self._axes_ to fit permutation matrix
        y_hat = utt.flatten_along_axes(x=y_hat, axes=self._axes_)

        # Shuffle
        y_hat = tf.gather(y_hat, self._inverse_permutation_, axis=self._axes_[0])

        # Unflatten to restore original shape
        x = tf.keras.ops.reshape(y_hat, newshape=old_shape)
        
        # Outputs
        return x
    
    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:

        # Create vector of zeros with length batch-size
        logarithmic_determinant = 0.0 * tf.keras.ops.sum(x, axis=list(range(1, len(x.shape))))

        # Outputs
        return logarithmic_determinant

class Shuffle(Permutation):
    """Shuffles input :math:`x`. The permutation used for shuffling is randomly chosen once during initialization. 
    Thereafter it is saved as a private attribute. Shuffling is thus deterministic. **IMPORTANT:** The shuffle function is defined on 
    a vector, yet by the requirement of :py:class:`Permutation`, inputs :math:`x` to this layer are allowed to have more than one axis 
    in ``axes``. As described in :py:class:`Permutation`, an input :math:`x` is first flattened along ``axes`` and thus the shuffling can
    be applied. For background information see :py:class:`Permutation`.
    
    :param shape: See base class :py:class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :py:class:`FlowLayer`.
    :type axes: List[int]
    """

    def __init__(self, shape: List[int], axes: List[int], **kwargs):

        # Super
        dimension_count = int(np.prod(shape))
        permutation = list(range(dimension_count)); random.shuffle(permutation)
        super(Shuffle, self).__init__(shape=shape, axes=axes, permutation=permutation, **kwargs)
    
class Heaviside(Permutation):
    """Swops the first and second half of input :math:`x` as inspired by the `Heaviside 
    <https://en.wikipedia.org/wiki/Heaviside_step_function>`_ function.  **IMPORTANT:** The Heaviside function is defined on a vector, 
    yet by the requirement of :py:class:`Permutation`, inputs :math:`x` to this layer are allowed to have more than one axis in ``axes``.
    As described in :py:class:`Permutation`, an input :math:`x` is first flattened along ``axes`` and thus the swopping can be applied. 
    For background information see :py:class:`Permutation`. If the number of dimensions to be permuted is odd-length,
    the permutation will move the first half excluding the middle element. 

    :param shape: See base class :py:class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :py:class:`FlowLayer`.
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
    defined on a matrix, i.e. 2 axes. Yet, here it is possible to specify any number of axes. Note, if the total number
    of elements in the to-be-permuted vector is odd, then the last element will remain in place.

    :param axes: See base class :py:class:`FlowLayer`.
    :type axes: :py:class:`List[int]`
    :param shape: See base class :py:class:`FlowLayer`. 
    :type shape: :py:class:`List[int]`
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
    def generate_rope_indices(shape: List[int]) -> Generator[int, None, None]:
        """Generates indices to traverse a tensor of ``shape``. The traversal follows a rope fitted along the axes by prioritizing
        later axes before earlier axes.

        :param shape: The shape of the tensor to be traversed.
        :type shape: :py:class:`List[int]`
        :yield: current_indices (:py:class:`List[int]`) - The indices pointing to the current cell in the tensor. It provides one index
            along each axis of ``shape``.
        """
        dimension_count = int(np.prod(shape))
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
        dimension_count = np.prod(shape)
        tensor = np.reshape(np.arange(dimension_count), shape)
        rope_values = [None] * dimension_count
        
        # Unravel tensor
        rope_index_generator = CheckerBoard.generate_rope_indices(shape=shape)
        for d in range(dimension_count): rope_values[d] = tensor[tuple(next(rope_index_generator))]

        # Swop every two adjacent values
        for d in range(0, dimension_count - 1, 2): # Skips the last element if odd
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
    
class Coupling(FlowLayer):
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

    :param shape: See base class :py:class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :py:class:`FlowLayer`.
    :type axes: List[int]
    :param compute_coupling_parameters: See the placeholder member :py:meth:`compute_coupling_parameters` for a detailed description 
        of requirements.
    :type compute_coupling_parameters: :py:class:`tensorflow.keras.Layer`
    :param mask: The mask used to select one half of the data while discarding the other half.
    :type mask: :py:class:`gyoza.modelling.masks.Mask`
    
    References:

        - `"NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION" by Laurent Dinh and David Krueger and Yoshua Bengio. <https://arxiv.org/abs/1410.8516>`_
        - `"Density estimation using Real NVP" by Laurent Dinh and Jascha Sohl-Dickstein and Samy Bengio. <https://arxiv.org/abs/1605.08803>`_
    """

    def __init__(self, shape: List[int], axes: List[int], compute_coupling_parameters: tf.keras.Layer, mask: mms.Mask, **kwargs):
        
        # Super
        super(Coupling, self).__init__(shape=shape, axes=axes, **kwargs)

        # Input validity
        shape_message = f"The shape ({shape}) provided to the coupling layer and that provided to the mask ({mask._mask_.shape}) are expected to be the same."
        assert len(shape) == len(mask._shape_), shape_message
        for i in range(len(shape)):
            assert shape[i] == mask._shape_[i], shape_message

        axes_message = f"The axes ({axes}) provided to the coupling layer and that provided to the mask ({mask._axes_}) are expected to be the same."
        assert len(axes) == len(mask._axes_), axes_message
        for i in range(len(axes)):
            assert axes[i] == mask._axes_[i], axes_message

        # Attributes
        self._compute_coupling_parameters_ = compute_coupling_parameters
        """(:py:class:`tensorflow.keras.Layer`) used inside the wrapper :py:meth:`compute_coupling_parameters`"""
        
        self._mask_ = mask
        """(:py:class:`gyoza.modelling.masks.Mask`) - The mask used to select one half of the data while discarding the other half."""
    
    def build(self, input_shape):
        # 1. Ensure the sub-model is built with the incoming shape
        if not self._compute_coupling_parameters_.built:
            self._compute_coupling_parameters_.build(input_shape)
        
        # 2. Call the base build to mark this layer as built
        super().build(input_shape)

    @staticmethod
    def _assert_parameter_validity_(parameters: tf.Tensor | List[tf.Tensor]) -> bool:
        """Determines whether the parameters are valid for coupling.
       
        :param parameters: The parameters to be checked.
        :type parameters: :py:class:`tensorflow.Tensor` or List[:py:class:`tensorflow.Tensor`]
        """

        raise NotImplementedError("All subclasses of Coupling layer should implement _assert_parameter_validity_ to ensure that _compute_coupling_parameters_ provides the correct output, i.e. a single tensor or a list of tensors.")
    
    def compute_coupling_parameters(self, x: tf.Tensor) -> tf.Tensor:
        """This method uses the :py:class:`tensorflow.keras.Layer` ``compute_coupling_parameters` provided during initialization to map ``x`` to coupling 
        parameters used to couple ``x`` with itself. The model may be arbitrarily complicated and **does not have to be invertible** since it only computes
        the coupling parameters, not the overall transformation of this layer. The coupling parameters can be a :py:class:`tensorflow.Tensor` or a list of
        such tensors, whatever is needed by the subclassing Coupling layer.
        
        :param x: The data to be transformed. Shape [batch size, ...] has to allow for masking via 
            :py:attr:`self._mask_`.
        :type x: :py:class:`tensorflow.Tensor`
        :return: y_hat (:py:class:`tensorflow.Tensor`) - The transformed version of ``x``."""
        
        # Propagate
        # Here we can not guarantee that the provided function uses x as name for first input.
        # We thus cannot use keyword input x=x. We have to trust that the first input is correctly interpreted as x.
        y_hat = self._compute_coupling_parameters_(x)

        # Outputs
        return y_hat

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        # Split x
        x_1 = self._mask_.call(x=inputs)

        # Compute parameters
        coupling_parameters = self.compute_coupling_parameters(x_1)
        self._assert_parameter_validity_(parameters=coupling_parameters)

        # Couple
        y_hat_1 = x_1
        y_hat_2 = self._mask_.call(x=self._couple_(x=inputs, parameters=coupling_parameters), is_positive=False)

        # Combine
        y_hat = y_hat_1 + y_hat_2

        # Compute Jacobian determinant
        jacobian_determinant = self.compute_jacobian_determinant(x=inputs)

        # Outputs
        return y_hat, jacobian_determinant
    
    def _couple_(self, x: tf.Tensor, parameters: tf.Tensor | List[tf.Tensor]) -> tf.Tensor:
        """This function implements an invertible coupling for inputs ``x`` and ``parameters``.
        
        :param x: The data to be transformed. Shape assumed to be [batch size, ...] where ... depends on axes of :py:attr:`self._mask_`. 
        :type x: :py:class:`tensorflow.Tensor`
        :param parameters: Constitutes the parameters that shall be used to transform ``x``. It's shape is assumed to support the 
            Hadamard product with ``x``.
        :type parameters: :py:class:`tensorflow.Tensor` or List[:py:class:`tensorflow.Tensor`]
        :return: y_hat (:py:class:`tensorflow.Tensor`) - The coupled tensor of same shape as ``x``."""

        raise NotImplementedError("Subclasses must implement _couple_")
    
    def _decouple_(self, y_hat: tf.Tensor, parameters: tf.Tensor | List[tf.Tensor]) -> tf.Tensor:
        """This function is the inverse of :py:meth:`_couple_`.
        
        :param y_hat: The data to be transformed. Shape assumed to be [batch size, ...] where ... depends on axes :py:attr:`self._mask_`.
        :type y_hat: :py:class:`tensorflow.Tensor`
        :param parameters: Constitutes the parameters that shall be used to transform ``y_hat``. It's shape is assumed to support the 
            Hadamard product with ``x``.
        :type parameters: :py:class:`tensorflow.Tensor` or List[:py:class:`tensorflow.Tensor`]
        :return: y_hat (:py:class:`tensorflow.Tensor`) - The decoupled tensor of same shape as ``y_hat``."""

        raise NotImplementedError("Subclasses must implement _decouple_")
    
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        
        # Split
        y_hat_1 = self._mask_.call(x=y_hat)

        # Compute parameters
        coupling_parameters = self.compute_coupling_parameters(y_hat_1)
        self._assert_parameter_validity_(parameters=coupling_parameters)
        
        # Decouple
        x_1 = y_hat_1
        x_2 = self._mask_.call(x=self._decouple_(y_hat=y_hat, parameters=coupling_parameters), is_positive=False)

        # Combine
        x = x_1 + x_2

        # Outputs
        return x
    
    def get_config(self):
        
        # Super
        config = super(Coupling, self).get_config()
        
        # Update config
        config = {"shape": self._shape_, "axes":self._axes_, "compute_coupling_parameters": tf.keras.layers.serialize(self._compute_coupling_parameters_), "mask": tf.keras.layers.serialize(self._mask_)}
        config.update(self.config)
        
        # Outputs
        return config
    
class AdditiveCoupling(Coupling):
    """This coupling layer implements an additive coupling law of the form :math:`f(x_2, c(x_1) = x_2 + c(x_1)`. For details on the
    encapsulating theory refer to :py:class:`Coupling`. It is important that the ``compute_coupling_parameters`` argument is a :py:class:`tensorflow.keras.Layer`
    that outputs a tensor of same shape as its input, i.e. not a list of tensors.
    
    References:

        - `"Density estimation using Real NVP" by Laurent Dinh and Jascha Sohl-Dickstein and Samy Bengio. <https://arxiv.org/abs/1605.08803>`_
    """

    def __init__(self, shape: List[int], axes: List[int], compute_coupling_parameters: tf.keras.Layer, mask: mms.Mask, **kwargs):
        
        # Super
        super(AdditiveCoupling, self).__init__(shape=shape, axes=axes, compute_coupling_parameters=compute_coupling_parameters, mask=mask, **kwargs)

    @staticmethod
    def _assert_parameter_validity_(parameters: tf.Tensor | List[tf.Tensor]) -> bool:
        # Assertion
        assert isinstance(parameters, tf.Tensor), f"For the AdditiveCoupling layer, the parameters output by the compute_coupling_parameters layer passed to __init__ are assumed to be of type tensorflow.Tensor not {type(parameters)}."
    
    def _couple_(self, x: tf.Tensor, parameters: tf.Tensor | List[tf.Tensor]) -> tf.Tensor:
        
        # Couple
        y_hat = x + parameters

        # Outputs
        return y_hat
    
    def _decouple_(self, y_hat: tf.Tensor, parameters: tf.Tensor | List[tf.Tensor]) -> tf.Tensor:
        
        # Decouple
        x = y_hat - parameters

        # Outputs
        return x
    
    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:
        
        # Create a vector of zeros with length batch-size
        logarithmic_determinant = 0.0 * tf.keras.ops.sum(x, axis=list(range(1, len(x.shape))))

        # Outputs
        return logarithmic_determinant
'''
class AffineCoupling(Coupling):
    """This coupling layer implements an affine coupling law of the form :math:`f(x_2, c(x_1) = e^s x_2 + t`, where :math:`s, t = c(x)`. 
    To prevent division by zero during decoupling, the exponent of :math:`s` is used as scale. For details on the encapsulating 
    theory refer to :py:class:`Coupling`. 
    
    References:

        - `"Density estimation using Real NVP" by Laurent Dinh and Jascha Sohl-Dickstein and Samy Bengio. <https://arxiv.org/abs/1605.08803>`_
    """

    def __init__(self, shape: List[int], axes: List[int], compute_coupling_parameters: tf.keras.Layer, mask: tf.Tensor, **kwargs):
        
        # Super
        super(AffineCoupling, self).__init__(shape=shape, axes=axes, compute_coupling_parameters=compute_coupling_parameters, mask=mask, **kwargs)

    @staticmethod
    def _assert_parameter_validity_(parameters: tf.Tensor | List[tf.Tensor]) -> bool:

        # Assert
        is_valid = type(parameters) == type([]) and len(parameters) == 2
        is_valid = is_valid and isinstance(parameters[0], tf.Tensor) and isinstance(parameters[1], tf.Tensor)
          
        assert is_valid, f"For this coupling layer parameters is assumed to be of type List[tensorflow.Tensor], not {type(parameters)}."
    
    def _couple_(self, x: tf.Tensor, parameters: tf.Tensor | List[tf.Tensor]) -> tf.Tensor:
        
        # Unpack
        scale = tf.exp(parameters[0])
        location = parameters[1]

        # Couple
        y_hat = scale * x + location

        # Outputs
        return y_hat
    
    def _decouple_(self, y_hat: tf.Tensor, parameters: tf.Tensor | List[tf.Tensor]) -> tf.Tensor:
        
        # Unpack
        scale = tf.exp(parameters[0])
        location = parameters[1]

        # Decouple
        x = (y_hat - location) / scale

        # Outputs
        return x
    
    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:
        
        # Split x
        x_1 = self._mask_.call(x=x)

        # Compute parameters
        coupling_parameters = self.compute_coupling_parameters(x_1)

        # Determinant
        logarithmic_scale = coupling_parameters[0]
        logarithmic_determinant = 0.0
        for axis in self._mask_._axes_:
            logarithmic_determinant += tf.keras.ops.reduce_sum(logarithmic_scale, axis=axis)

        # Outputs
        return logarithmic_determinant
'''
class ActivationNormalization(FlowLayer):
    """A trainable location and scale transformation of the data. For each dimension of the specified input shape, a scale and a 
    location parameter is used. That is, if shape == [width, height], then 2 * width * height many parameters are used. Each pair of 
    location and scale is initialized to produce mean equal to 0 and variance equal to 1 for its dimension. To allow for 
    invertibility, the scale parameter is constrained to be non-zero. To simplify computation of the jacobian determinant on 
    logarithmic scale, the scale parameter is here constrained to be positive. Each dimension has the following activation 
    normalization:
    
    - y_hat = (x-l)/s, where s > 0 and l are the scale and location parameters for this dimension, respectively.

    :param shape: See base class :py:class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :py:class:`FlowLayer`.
    :type axes: List[int]
    
    References:

    - `"Glow: Generative Flow with Invertible 1x1 Convolutions" by Diederik P. Kingma and Prafulla Dhariwal. <https://arxiv.org/abs/1807.03039>`_
    - `"A Disentangling Invertible Interpretation Network for Explaining Latent Representations" by Patrick Esser, Robin Rombach and Bjorn Ommer. <https://arxiv.org/abs/2004.13166>`_
    """
    
    class _PositiveConstraint_(tf.keras.constraints.Constraint):

        def __call__(self, w: tf.Variable):
            return tf.clip_by_value(w, clip_value_min=1e-6, clip_value_max=w.value.dtype.max)

    def __init__(self, shape: List[int], axes: List[int], **kwargs):

        # Super
        super(ActivationNormalization, self).__init__(shape=shape, axes=axes)

        # Attributes
        self._location_ = self.add_weight(shape = self._shape_,
                                           initializer = tf.keras.ops.zeros(self._shape_),
                                           dtype = tf.keras.backend.floatx(),
                                           trainable = True,
                                           name="_location_")
        """The value by which each data point shall be translated."""
        
        self._scale_ = self.add_weight(shape = self._shape_,
                                        initializer = tf.keras.ops.ones(self._shape_),
                                        dtype = tf.keras.backend.floatx(), 
                                        trainable=True, 
                                        constraint=ActivationNormalization._PositiveConstraint_(),
                                        name="_scale_")
        """The value by which each data point shall be scaled."""

    def _prepare_variables_for_computation_(self, x:tf.Tensor) -> Tuple[tf.Variable, tf.Variable]:
        """Prepares the variables for computation with data. This ensures variable shapes are compatible with ``x``.
        
        :param x: Data to be passed through :py:meth:`call`. It's shape must agree with input ``x`` of 
            :py:meth:`self.__reshape_variables__`.
        :type x: :py:class:`tensorflow.Tensor`

        :return: 
            - location (tensorflow.Variable) - The :py:attr:`_location_` attribute shaped to fit ``x``. 
            - scale (tensorflow.Variable) - The :py:attr:`_scale_` attribute shaped to fit ``x``."""

        # Shape variables to fit x
        axes = list(range(len(x.shape)))
        for axis in self._axes_: axes.remove(axis)
        location = utt.expand_axes(x=self._location_, axes=axes)
        scale = utt.expand_axes(x=self._scale_, axes=axes)
        
        # Outputs
        return location, scale

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor,tf.Tensor]:
        
        # If this is the first training call
        #if not tf.is_symbolic_tensor(x) and not self._is_initialized_: self._lazy_init_(x=x)

        # Transform
        location, scale = self._prepare_variables_for_computation_(x=inputs)
        y_hat = (inputs - location) / scale # Scale is positive due to constraint
        jacobian_determinant = self.compute_jacobian_determinant(x=inputs)
        # Outputs
        return y_hat, jacobian_determinant
        
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:

        # Transform
        location, scale = self._prepare_variables_for_computation_(x=y_hat)
        x =  y_hat * scale + location

        # Outputs
        return x
    
    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:

        # Count dimensions over remaining axes (for a single instance)
        dimension_count = 1
        for axis in range(1,len(tf.keras.ops.shape(x))):
            if axis not in self._axes_:
                dimension_count *= tf.keras.ops.shape(x)[axis] 
        
        # Compute logarithmic determinant
        # By defintion: sum across dimensions for ln(scale)
        logarithmic_determinant = - dimension_count * tf.keras.ops.sum(tf.keras.ops.log(self._scale_)) # single instance, scale is positive due to constraint, the - sign in front is because the scale is used in the denominator
        logarithmic_determinant = 0.0 * tf.keras.ops.sum(x, axis=list(range(1, len(x.shape)))) + logarithmic_determinant
        
        # Outputs
        return logarithmic_determinant
    
class Reflection(FlowLayer):
    """This layer reflects a data point around ``reflection_count`` learnable normals using the `Householder transform 
    <https://en.wikipedia.org/wiki/Householder_transformation>`_. In this context, the normal is the unit length vector orthogonal to
    the hyperplane of reflection. When ``axes`` contains more than a single entry, the input is first flattened along these axes, 
    then reflected and then unflattened to original shape.

    :param shape: See base class :py:class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :py:class:`FlowLayer`. **IMPORTANT**: These axes are distinct from the learnable reflection axes.
    :type axes: List[int]
    :param reflection_count: The number of successive reflections that shall be executed. Expected to be at least 1.
    :type reflection_count: int

    References:

        - `"Gaussianization Flows" by Chenlin Meng, Yang Song, Jiaming Song and Stefano Ermon <https://arxiv.org/abs/2003.01941>`_
    """

    def __init__(self, shape: List[int], axes: List[int], reflection_count: int, **kwargs):
        # Input validity
        assert 1 <= reflection_count, f'The input reflection_count was expected to be at least 1 but found to be {reflection_count}.'
        
        # Super
        super(Reflection, self).__init__(shape=shape, axes=axes, **kwargs)

        # Attributes
        self._reflection_count_ = reflection_count
        
        self._inverse_mode_ = False
        "(bool) - Indicates whether the reflections shall be executed in reversed order (True) or forward order (False)."

        reflection_normals = tf.math.l2_normalize(-1+2*tf.random.uniform(shape=[self._reflection_count_, np.prod(self._shape_)], dtype=tf.keras.backend.floatx()), axis=1) 
        self._reflection_normals_ = self.add_weight(shape = reflection_normals.shape,
                                                     initializer = reflection_normals,
                                                     dtype = reflection_normals.dtype,
                                                     trainable = True,
                                                     name="reflection_normals",
                                                     constraint=tf.keras.constraints.UnitNorm(axis=1)) # name is needed for getting and setting weights
        """(:py:class:`tensorflow.Tensor`) - These are the axes along which an instance is reflected. Shape == [reflection count, dimension count] where dimension count is the product of the shape of the input instance along :py:attr:`self._axes_`."""
        
    def _reflect_(self, x: tf.Tensor) -> tf.Tensor:
        """This function executes all the reflections of self in a sequence by by applying successive Householder reflections 
        defined by the :py:attr:`_reflection_normals_`. This method provides the backward reflection if 
        :py:attr:`self._inverse_mode` == True and forward otherwise.

        :param x: The flattened data of shape [..., dimension count], where dimension count is the product of the :py:attr:`_shape_` as 
            specified during initialization of self. It is assumed that all axes except for :py:attr:`_axes_` (again, see 
            initialization of self) are moved to ... in the aforementioned shape of ``x``.
        :type x: :py:class:`tensorflow.Tensor`
        :return: x_new (:py:class:`tensorflow.Tensor`) - The rotated version of ``x`` with same shape.
        """

        # Pass x through the sequence of reflections
        x_new = x
        indices = list(range(self._reflection_count_))
        if self._inverse_mode_: 
            # Note: Householder reflections are involutory (their own inverse) https://en.wikipedia.org/wiki/Householder_transformation
            # One can thus invert a sequence of reflections by reversing the order of the individual reflections
            indices.reverse()

        for r in indices:
            v_r = self._reflection_normals_[r][:, tf.newaxis] # Shape == [dimension count, 1]
            dot = tf.keras.ops.sum(x_new * v_r[:, 0], axis=-1, keepdims=True)
            x_new = x_new - 2.0 * dot * v_r[:, 0]

        # Outputs
        return x_new
    
    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        
        # Initialize
        old_shape = list(tf.keras.ops.shape(inputs))

        # Flatten along self._axes_ to fit reflection matrix
        inputs = utt.flatten_along_axes(x=inputs, axes=self._axes_)

        # Move this flat axis to the end for multiplication with reflection matrices
        inputs = utt.move_axis(x=inputs, from_index=self._axes_[0], to_index=-1)

        # Reflect
        y_hat = self._reflect_(x=inputs)

        # Move axis back to where it came from
        y_hat = utt.move_axis(x=y_hat, from_index=-1, to_index=self._axes_[0])

        # Unflatten to restore original shape
        y_hat = tf.keras.ops.reshape(y_hat, newshape=old_shape)

        # Compute Jacobian determinant
        jacobian_determinant = self.compute_jacobian_determinant(x=inputs)
        
        # Outputs
        return y_hat, jacobian_determinant
    
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        
        # Prepare self for inversion
        previous_mode = self._inverse_mode_
        self._inverse_mode_ = True

        # Call forward method (will now function as inverter)
        x, _ = self(inputs=y_hat)

        # Undo the setting of self to restore the method's precondition
        self._inverse_mode_ = previous_mode

        # Outputs
        return x
    
    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:
        
        # It is known that Householder reflections have a determinant of -1 https://math.stackexchange.com/questions/504199/prove-that-the-determinant-of-a-householder-matrix-is-1
        # It is also known that det(AB) = det(A) det(B) https://proofwiki.org/wiki/Determinant_of_Matrix_Product
        # This layer applies succesive reflections as matrix multiplications and thus the determinant of the overall transformation is
        # -1 or 1, depending on whether an even or odd number of reflections are concatenated. Yet on logarithmic scale it is always 0.
        
        # Create vector of zeros with length batch-size
        logarithmic_determinant = 0.0 * tf.keras.ops.sum(x, axis=list(range(1, len(x.shape))))

        # Outputs
        return logarithmic_determinant

'''
class Flatten(FlowLayer):

    def __init__(self, shape: Tuple[int], axes: Tuple[int]) -> "Flatten":
        
        # Super
        super().__init__(shape=shape, axes=axes)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        
        # Initialize
        self.__old_shape__ = cp.copy(tf.keras.ops.shape(x))

        # Flatten along self._axes_ to fit reflection matrix
        y_hat = utt.flatten_along_axes(x=x, axes=self._axes_)

        # Output
        return y_hat
    
    def invert(self, y_hat: tf.Tensor) -> tf.Tensor:
        
        # Unflatten to restore original shape
        y_hat = tf.ops.keras.reshape(y_hat, shape=self.__old_shape__)
        
        # Output
        return y_hat

    def compute_jacobian_determinant(self, x: tf.Tensor) -> tf.Tensor:
        
        logarithmic_determinant = tf.keras.ops.zeros_like(x[:, 0])

        # Outputs
        return logarithmic_determinant
        

class DisentanglingFlowModel(FlowModel):
    """This network is a :py:class:`FlowModel` that can be used to disentangle factors, e.g. to understand representations
    in latent spaces of regular neural networks. It automatically uses the :py:class:`losses.SupervisedFactorLoss` to compute its losses.
    It also overrides the :py:class:`FlowModel`'s implementation for train_step to accomodate for the fact that calibration does not
    simply use single instances but pairs of instances and their similarity.
    
    :param sigma: A measure of how tight clusters in the output space shall be. It is used to set up the factorized loss.
    :type sigma: float

    References:

       - `"A Disentangling Invertible Interpretation Network for Explaining Latent Representations" by Patrick Esser, Robin Rombach and Bjorn Ommer <https://arxiv.org/abs/2004.13166>`_
    """
    
    def __init__(self, layers: List[FlowLayer], **kwargs):
        super().__init__(layers, **kwargs)
        
    
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
        :type Z_ab: :py:class:`numpy.ndarray`
        :param Y_ab: The factor-wise similarity of instances in each pair of ``Z_ab``. **IMPORTANT:** Here, it is assumed that the 
            residual factor is at index 0 AND that the values of ``Y_ab`` are either 0 or 1. Shape == [instance count, factor count].
        :type Y_ab: :py:class:`numpy.ndarray`

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
    
    def train_step(self, data) -> tf.Tensor:
        """Computes the supervised factor-loss for pairs of instances and applies the gradient to the variables.

        :param data: A tuple containg the batch of X and Y, respectively. X is assumed to be a tensorflow.Tensor of shape [batch size,
            2, ...] where 2 indicates the pair x_a, x_b of same factor and ... is the shape of one input instance that has to fit 
            through :py:attr:`self.sequence`. The tensorflow.Tensor Y shall contain the factor indices of shape [batch size].
        :type data: Tuple(tensorflow.Tensor, tensorflow.Tensor)
        :return: loss (:py:class:`tensorflow.Tensor`) - A scalar for the loss observed before applying the train step.
        """
        
        # Unpack inputs
        (z_a, z_b), Y = data
        
        with tf.GradientTape() as tape:
            # First instance
            z_tilde_a = self(z_a, training=True)  # Forward pass
            j_a = self.compute_jacobian_determinant(x=z_a)
            
            # Second instance
            z_tilde_b = self(z_b, training=True)
            j_b = self.compute_jacobian_determinant(x=z_b)
            
            # Compute loss
            loss = self.loss(y_true=Y, y_pred=(z_tilde_a, z_tilde_b, j_a, j_b))

        # Compute and apply gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        #self.loss_tracker.update_state(loss)
        
        # Output
        return {"loss": loss}

'''