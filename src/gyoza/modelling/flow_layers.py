import numpy as np
import tensorflow as tf
from typing import Tuple, List, Union, Generator, Dict, Any, Type, TypeVar

from gyoza.utilities import tensors as utt
import gyoza.modelling.masks as mms
import copy as cp
from gyoza.modelling import losses as mls
import random

FlowModelType = TypeVar("FlowModelType", bound="FlowModel")
FlowLayerType = TypeVar("FlowLayerType", bound="FlowLayer")

@tf.keras.utils.register_keras_serializable()
class FlowModel(tf.keras.Model):
    """A :py:class:`tensorflow.keras.Model` subclass for creating flow models. It relies on a list of ``flow_layers`` as input which will always 
    be executed in the given sequence. Each such :py:class:`~gyoza.modelling.flow_layers.FlowLayer` is expected to implement:
    
    - a call(self, inputs: Union[tensorflow.Tensor, tensorflow.keras.KerasTensor]) -> Tuple[Union[tensorflow.Tensor, tensorflow.keras.KerasTensor], Union[tensorflow.Tensor, tensorflow.keras.KerasTensor]] method that works on eager as well as symbolic :py:class:`tensorflow.Tensor` instances and outputs (outputs, jacobian_determinant), where ``outputs`` is the transformed version of ``inputs`` and ``jacobian_determinant`` is the Jacobian determinant of the transformation on logarithmic scale. 
    - an invert(self, y_hat: tensorflow.Tensor) -> tensorflow.Tensor method that reconstructs the ``inputs`` tensor to the `call` method from its ``outputs``. 
    - a build(self, input_shape: tensorflow.TensorShape) method that ensures all model variables are constructed and layer.built == True. This is only needed if the `__init__` method does not fulfill these criteria already.
    - serializable behavior via :py:meth:`tensorflow.keras.layers.Layer.get_config` and :py:meth:`tensorflow.keras.layers.Layer.from_config`.
    
    It is possible to :py:meth:`~gyoza.modelling.flow_layers.FlowModel.add` layers after construction, but this has to happen **before** the model is built via :py:meth:`~gyoza.modelling.flow_layers.FlowModel.build` and compiled via :py:meth:`tensorflow.keras.Model.compile`.

    :param flow_layers: A list of flow layers.
    :type flow_layers: List[py:class:`~gyoza.modelling.flow_layers.FlowLayer`]"""


    def __init__(self, flow_layers: List["FlowLayer"], **kwargs) -> None:
        
        # Super
        super().__init__(**kwargs)
        
        # Input validity
        if not all([isinstance(layer, FlowLayer) for layer in flow_layers]): raise TypeError(f"The input flow_layers provided to the FlowModel needs to be an array of ~gyoza.modelling.flow_layers.FlowLayer instances but was {[type(layer) for layer in flow_layers]}.")
        self._flow_layers_ = flow_layers
    
    def add(self, flow_layer: "FlowLayer") -> None:
        """
        Adds another flow-layer to the internal list of flow-layers. **Important:** It is only possible to add layers before this model is built via 
        :py:meth:`~gyoza.modelling.flow_layers.FlowModel.build` and compiled via :py:meth:`tensorflow.keras.Model.compile`.

        :param flow_layer: A flow layer with the same requirements as assumed by the constructor of :py:class:`~gyoza.modelling.flow_layers.FlowModel`.
        :type flow_layer: :py:class:`~gyoza.modelling.flow_layers.FlowLayer`
        """

        # State validity
        if self.built: raise Exception(f"Attempted to call ~gyoza.modelling.flow_layers.FlowModel.add on a FlowModel that has already been built. All layers need to be added before calling ~gyoza.modelling.flow_layers.FlowModel.build.")
        
        # Input validity
        if not isinstance(flow_layer, FlowLayer): raise TypeError(f"The input flow_layer provided to ~gyoza.modelling.FlowModel.add needs to be a ~gyoza.modelling.flow_layers.FlowLayer instance but was {type(flow_layer)}.")
        
        # Add layer
        self._flow_layers_.append(flow_layer)

    def build(self, input_shape: Union[tf.TensorShape, Tuple, List]) -> None:
        """
        Builds the model by calling the :py:attr:`~gyoza.modelling.flow_layers.FlowLayer.build` method on all internally managed flow-layers.
        Assumes that each such flow-layer implements a call(self, inputs: tensorflow.keras.KerasTensor) -> Tuple[tensorflow.keras.KerasTensor, Union[tensorflow.keras.KerasTensor, tensorflow.keras.KerasTensor]] 
        method that works on symbolic :py:class:`tensorflow.keras.KerasTensor` instances and outputs (outputs, _), where ``outputs`` is the transformed version of `input`. 
        Also assumes that this `call` method can be called once the layer is built to pass a symbolic input to the next layer in sequence.
    
        :param input_shape: The shape of the input to the model. The input is assumed to be a single :py:class:`tensorflow.Tensor` (i.e. not a collection) and its shape to be provided here shall be structured as [batch-size, ...], where ... is any instance shape consistent with what the :py:class:`~gyoza.modelling.flow_layers.FlowLayer` instances of this model support.
        :type input_shape: Union[:py:class:`tensorflow.TensorShape`, :py:class:`tuple`, :py:class:`list`]
        """

        # Input validity
        if not (isinstance(input_shape, tf.TensorShape) or isinstance(input_shape, Tuple) or isinstance(input_shape, List)): raise TypeError(f"The input_shape passed to ~gyoza.modelling.flow_layers.FlowModel.build is expected to be of type tensorflow.TensorShape, Tuple or List but was {type(input_shape)}.")

        # Call to super
        super().build(input_shape=input_shape)
        
        # Iterate flow layers 
        inputs = tf.keras.Input(input_shape[1:])
        for layer in self._flow_layers_:
            
            # Build layer
            if not layer.built: layer.build(input_shape=tf.keras.ops.shape(inputs))

            # Update input shape for next layer
            inputs, _ = layer(inputs=inputs)

    def call(self, inputs: Union[tf.Tensor, tf.keras.KerasTensor]) -> Tuple[Union[tf.Tensor, tf.keras.KerasTensor], Union[tf.Tensor, tf.keras.KerasTensor]]:
        """Calls the :py:meth:`~gyoza.modelling.flow_layers.FlowLayer.call` method of each of the internally maintained :py:class:`~gyoza.modelling.flow_layers.FlowLayer` instances in the sequence in which they were added to this model.
        Assumes that each flow-layer implements a call(self, inputs: Union[tensorflow.Tensor, tensorflow.keras.KerasTensor]) -> Tuple[Union[tensorflow.Tensor, tensorflow.keras.KerasTensor], Union[tensorflow.Tensor, tensorflow.keras.KerasTensor]] method that works on eager as well as symbolic :py:class:`tensorflow.Tensor` instances and outputs (outputs, jacobian_determinant), where `outputs` is the transformed version of `inputs` and `jacobian_determinant` is the Jacobian determinant of the transformation on logarithmic scale. 
    
        :param inputs: The data to be tranformed. Assumed to be eager numeric or symbolic and of shape [batch-size, ...], where ... is any instance shape with at least 1 axis consistent with what the :py:class:`~gyoza.modelling.flow_layers.FlowLayer` instances of this model support.
        :type inputs: Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`]
        :return: (outputs, jacobian_determinant) (Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`], Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`]]) - The outputs of the transformation of shape [batch-size, ...] and the Jacobian determinant on logarithmic scale of shape [batch-size].
        """
        
        # Input validity
        if not (isinstance(inputs, tf.Tensor) or isinstance(inputs, tf.keras.KerasTensor)): raise TypeError(f"The inputs provided to ~gyoza.modelling.flow_layers.FlowModel.call is assumed to be an eager numeric tensorflow.Tensor or symbolic tensorflow.keras.KerasTensor but was {type(inputs)}.")
        if not (len(tf.keras.ops.shape(inputs)) > 1): raise ValueError(f"The inputs provided to ~gyoza.modelling.flow_layers.FlowModel.call is assumed to have at least two axis, the first of which should be the batch axis, but the given inputs has shape {tf.keras.ops.shape(inputs)}.")

        # Transform
        jacobian_determinant = 0.0 * tf.keras.ops.sum(inputs, axis=list(range(1, len(tf.keras.ops.shape(inputs)))))
        outputs = inputs
        for layer in self._flow_layers_:
            outputs, jacobian_determinant_l = layer(inputs=outputs)
            jacobian_determinant += jacobian_determinant_l

        # Outputs
        return outputs, jacobian_determinant
    
    def invert(self, outputs: tf.Tensor) -> tf.Tensor:
        """Executes the operation of this model in the inverse direction by calling the :py:meth:`~gyoza.modelling.flow_layers.invert` method on all layers 
        in the internally managed list of :py:class:`~gyoza.modelling.flow_layers.FlowLayer` instances in reverse order. This method hence assumes that every 
        :py:class:`~gyoza.modelling.flow_layers.FlowLayer` implements an invert(self, outputs: tf.Tensor) -> tf.Tensor method
        that reconstructs the input to its :py:meth:`~gyoza.modelling.FlowLayer.call` method. 
        This model's invert method is thus the counterpart to its :py:meth:`~gyoza.modelling.FlowModel.call`. **Note:** This method is **only intended for use in eager mode**. 

        :param outputs: The data to be transformed. Assumed to be of the same shape as the output of :py:meth:`~gyoza.modelling.FlowModel.call`.
        :type outputs: :py:class:`tensorflow.Tensor`
        :return: reconstructed_inputs (:py:class:`tensorflow.Tensor`) - The output of the transformation of shape [batch-size, ...]."""        

        # Input validity
        if not isinstance(outputs, tf.Tensor): raise TypeError(f"The outputs argument provided to ~gyoza.modelling.flow_layers.FlowModel.invert is expected to be a tensorflow.Tensor, but was {type(outputs)}.")

        # Transform
        for layer in reversed(self._flow_layers_): outputs = layer.invert(outputs=outputs)
        reconstructed_inputs = outputs

        # Outputs
        return reconstructed_inputs
    
    def get_config(self) -> Dict[str, Any]:
        """
        Converts self to a configuration dictionary that can be used to reconstruct it via the corresponding flow model class's :py:meth:`~gyoza.modelling.flow_layers.FlowModel.from_config`. 

        :return: config (Dict[str, Any]) - The configuration dictionary.
        """

        # Update super config
        config = super().get_config()
        config.update({
            "flow_layers": [tf.keras.layers.serialize(layer) for layer in self._flow_layers_]
        })

        # Outputs
        return config

    @classmethod
    def from_config(cls: FlowModelType, config: Dict[str, Any]) -> FlowModelType:
        """
        Loads a flow model from its configuration dictionary. The dictionary is required to have been obtained via :py:meth:`~gyoza.modelling.flow_layers.FlowModel.get_config`.
        
        :param cls: The specific type of the flow model to be constructed.
        :type cls: FlowModelType
        :param config: The dictionary containing the configuration of the flow model.
        :type config: Dict[str, Any]
        :return: instance (FlowModelType) - The constructed instance.
        """
        # No input checks since this method is called with valid inputs by keras

        # Construct instance
        layers_config = config.pop("flow_layers")
        flow_layers = [tf.keras.layers.deserialize(l) for l in layers_config]
        instance = cls(flow_layers=flow_layers, **config)
        
        # Outputs
        return instance

@tf.keras.utils.register_keras_serializable()
class DisentanglingFlowModel(FlowModel):
    """A :py:class:`~gyoza.modelling.flow_layers.FlowModel` subclass designed to disentangle latent factors in neural network representations.

    This model follows the methodology described in:

        - Esser, P., Rombach, R., & Ommer, B. (2020). 
          "A Disentangling Invertible Interpretation Network for Explaining Latent Representations." 
          `arXiv:2004.13166 <https://arxiv.org/abs/2004.13166>`_

    **Important:** The model expects the output of its
    :py:meth:`~gyoza.modelling.flow_layers.FlowModel.call` to have shape `[batch-size, instance-dimensionality]`. 
    For inputs with more axes (e.g. images), you must either flatten them before passing them to this model or
    include flattening layers in your `flow_layers` sequence to make sure your flow model's output enumerates all its dimensions along one axis.

    **Inference:** Use the regular
    :py:meth:`~gyoza.modelling.flow_layers.FlowModel.call` on a batch of instances.

    **Training:** The model overrides :py:meth:`~tensorflow.keras.Model.train_step` to accept pairs of instances `(Z_a, Z_b)` and a corresponding 
    target tensor ``y_true``. The outputs of both instances, including their Jacobian determinants, are concatenated 
    along axis=-1 and passed as ``y_pred`` to the loss function. When calling
    :py:meth:`tensorflow.keras.Model.compile`, you should provide `~gyoza.modelling.losses.SupervisedFactorLoss` (or compatible) as the loss.

    **Requirements for :py:meth:`tensorflow.keras.Model.fit` inputs:**
    
    - ``x`` must be a tuple `(batch A, batch B)` where each batch is valid input to :py:meth:`~gyoza.modelling.flow_layers.FlowModel.call`.
    - ``y`` must match the `y_pred` expected by the loss function assigned during :py:meth:`tensorflow.keras.Model.compile`.

    **Note:** This model is intended for Keras subclassing workflow (eager or graph mode), not the functional API.

    :param flow_layers: List of flow layers for the model.
    :type flow_layers: List[:py:class:`~gyoza.modelling.flow_layers.FlowLayer`]
    """
    
    def __init__(self, flow_layers: List["FlowLayer"], **kwargs):
        super().__init__(flow_layers, **kwargs)
        
    def train_step(self, data: Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]) -> Dict[str, tf.Tensor]:
        
        # Input validity
        if not (isinstance(data, (tuple, list)) and len(data) == 2 and isinstance(data[0], (tuple, list)) and len(data[0]) == 2):
            raise TypeError(
                f"Expected `data` to be Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor], got {type(data)}."
            )
        if not all(isinstance(x, tf.Tensor) for x in (*data[0], data[1])):
            raise TypeError(
                f"All elements in `data` must be tf.Tensor instances, got {[type(x) for x in (*data[0], data[1])]}"
            )
        
        # Unpack data using Z-notation (equivalent to X-notation, yet in line with the referenced paper)
        (Z_a, Z_b), y_true = data

        with tf.GradientTape() as tape:
            # Forward pass for instances of pair
            y_a, j_a = self(Z_a)
            y_b, j_b = self(Z_b)

            # Concatenate all outputs to have a single tensor as prediction
            y_pred = tf.concat([y_a, y_b, tf.reshape(j_a, (-1, 1)), tf.reshape(j_b, (-1, 1))], axis=-1)

            # Compute loss
            loss = self.compute_loss(x=None, y=y_true, y_pred=y_pred)

        # Compute gradients and update
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Outputs
        return {"loss": loss}

@tf.keras.utils.register_keras_serializable() 
class FlowLayer(tf.keras.layers.Layer):
    """Abstract base class for flow layers. Any input to this layer is assumed to have ``shape`` along ``axes`` as specified during
    initialization.

    **Subclass Instructions:** Any subclasses of this class must

    - override :py:meth:`~gyoza.modelling.flow_layers.FlowLayer.call`
    - override :py:meth:`~gyoza.modelling.flow_layers.FlowLayer.invert`
    - adhere to `serializable`
    
    :param shape: The shape of the input along the specified `axes` that shall be transformed by this layer. If you have e.g. a tensor [batch-size, width, 
        height, color] and you want this layer to transform along width and height, you enter the shape [width, height]. If you 
        want the layer to operate on the color you provide the shape [color] instead.
    :type shape: List[int]
    :param axes: The axes along which the transformation of this layers shall be performed. In the example for `shape` on width and height you would 
        enter [1,2] here. In the example for color you would enter [3] here. Although axes are counted starting from zero, it is assumed that ``axes`` 
        does not contain the axis 0, i.e. the batch axis. Furthermore, axes are assumed to be ascending but they do not have to be contiguous.
    :type axes: List[int]

    References:

        - Dinh, L., Sohl-Dickstein, J. & Bengio, S. (2016).
          "Density estimation using Real NVP"
          `arXiv:1605.08803 <https://arxiv.org/abs/1605.08803>`_

        - Kingma, D. P. & Dhariwal, P. (2018)
          "Glow: Generative Flow with Invertible 1x1 Convolutions"
          `arXiv:1807.03039 <https://arxiv.org/abs/1807.03039>`_

        - Dinh, L., Krueger, D. & Bengio, Y. (2015)
          "NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION" 
          `arXiv:1410.8516 <https://arxiv.org/abs/1410.8516>`_

        - Sankar, A., Keicher, M., Eisawy, R., Parida, A., Pfister, F., Kim, S., T. & Navab, N. (2021)
          "GLOWin: A Flow-based Invertible Generative Framework for Learning Disentangled Feature Representations in Medical Images" 
          `arXiv:2103.10868 <https://arxiv.org/abs/2103.10868>`_

        - Meng, C., Song, Y., Song, J. & Ermon, S. (2020)
          "Gaussianization Flows"
          `arXiv:2003.01941 <https://arxiv.org/abs/2003.01941>`_
        
        - Esser, P., Rombach, R., & Ommer, B. (2020). 
          "A Disentangling Invertible Interpretation Network for Explaining Latent Representations." 
          `arXiv:2004.13166 <https://arxiv.org/abs/2004.13166>`_
    """

    def __init__(self, shape: List[int], axes: List[int], **kwargs) -> None:
        """This constructor shall be used by subclasses only"""

        # Super
        super(FlowLayer, self).__init__(**kwargs)

        # Input validity
        if not len(shape) == len(axes): raise ValueError(f"The input shape ({shape}) is expected to have as many entries as the input axes ({axes}).")
        if not all((isinstance(axes[i], int) and axes[i] > 0) for i in range(len(axes))): raise ValueError(f"The axes in input axes ({axes}) are assumed to be positive int values.")
        for i in range(len(axes)-1):
            if not axes[i] < axes[i+1]: raise ValueError(f"The axes in input axes ({axes}) are assumed to be strictly ascending.")

        if 0 in axes: raise ValueError(f"The input axes ({axes}) must not contain the batch axis, i.e. 0.")

        # Attributes
        self._shape_ = cp.copy(shape)
        """(List[int]) - The shape of the input that shall be transformed by this layer. For detail, see constructor of :py:class:`FlowLayer`"""

        self._axes_ = cp.copy(axes)
        """(List[int]) - The axes of transformation. For detail, see constructor of :py:class:`FlowLayer`"""

    def call(self, inputs: Union[tf.Tensor, tf.keras.KerasTensor]) -> Tuple[Union[tf.Tensor, tf.keras.KerasTensor], Union[tf.Tensor, tf.keras.KerasTensor]]:
        """Executes the operation of this layer in the forward direction.

        :param inputs: The data to be tranformed. Assumed to be of shape [batch-size, ...], where ... is the shape of a single instance assumed to have at least one axis.
        :type inputs: Union[tf.Tensor, tf.keras.KerasTensor]
        :return:
            - y_hat (Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`]): The output of the transformation of shape [batch-size, ...].
            - jacobian_determinant (Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`]): The Jacobian determinant on logarithmic scale of shape [batch-size]."""
        
        raise NotImplementedError()
    
    def invert(self, outputs: tf.Tensor) -> tf.Tensor:
        """Executes the operation of this layer in the inverse direction. It is thus the counterpart to :py:meth:`call`.
        **Note:** This method is **only intended for use in eager mode**. 

        :param outputs: The outputs of :py:meth:`~gyoza.modelling.flow_layers.FlowLayer.call`.
        :type outputs: :py:class:`tensorflow.Tensor`
        :return: reconstructed_inputs (:py:class:`tensorflow.Tensor`) - The reconstructed input to :py:meth:`~gyoza.modelling.flow_layers.FlowLayer.call`."""        
        
        raise NotImplementedError()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Converts self to a configuration dictionary that can be used to reconstruct it via the corresponding flow layer class's :py:meth:`~gyoza.modelling.flow_layers.FlowLayer.from_config`. 

        :return: config (Dict[str, Any]) - The configuration dictionary.
        """

        # Update the super config
        config = super().get_config()
        config.update({
            "shape": self._shape_,
            "axes": self._axes_
        })
        
        # Outputs
        return config

@tf.keras.utils.register_keras_serializable()
class Permutation(FlowLayer):
    """A :py:class:`~gyoza.modelling.flow_layers.FlowLayer` subclass that flattens its input along ``axes``, then reorders the dimensions using ``permutation`` and reshapes 
    it back to its original shape. This transformation is volume-preserving, meaning it has a Jacobian determinant of 1 (or 0 on logarithmic scale).

    :param shape: See base class :py:class:`~gyoza.modelling.flow_layers.FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :py:class:`~gyoza.modelling.flow_layers.FlowLayer`.
    :type axes: List[int]
    :param permutation: A new order of the indices in the interval [0, product(`shape`)).
    :type permutation: List[int]
    """

    def __init__(self, shape: List[int], axes: List[int], permutation: List[int], **kwargs) -> None:

        # Input validity
        dimension_count = int(np.prod(shape))
        if not len(permutation) == dimension_count: raise ValueError(f'The input permutation was expected to have length {dimension_count} based on the number of dimensions in the shape input but it was found to have length {len(permutation)}.')
        if sorted(permutation) != list(range(dimension_count)): raise ValueError(f"The argument permutation provided to ~gyoza.modelling.flow_layers.Permutation.__init__ must be a reordering of [0, ..., product(shape)-1]., but was {permutation}.")

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

    def call(self, inputs: Union[tf.Tensor, tf.keras.KerasTensor]) -> Tuple[Union[tf.Tensor, tf.keras.KerasTensor], Union[tf.Tensor, tf.keras.KerasTensor]]:

        # Input validity
        if not (isinstance(inputs, tf.Tensor) or isinstance(inputs, tf.keras.KerasTensor)): raise TypeError(f"The inputs provided to ~gyoza.modelling.flow_layers.Permutation.call is assumed to be an eager numeric tensorflow.Tensor or symbolic tensorflow.keras.KerasTensor but was {type(inputs)}.")
        if not (len(tf.keras.ops.shape(inputs)) > 1): raise ValueError(f"The inputs provided to ~gyoza.modelling.flow_layers.Permutation.call is assumed to have at least two axis, the first of which should be the batch axis, but the given inputs has shape {tf.keras.ops.shape(inputs)}.")

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
    
    def invert(self, outputs: tf.Tensor) -> tf.Tensor:
        
        # Input validity
        if not isinstance(outputs, tf.Tensor): raise TypeError(f"The outputs argument provided to ~gyoza.modelling.flow_layers.Permutation.invert is expected to be a tensorflow.Tensor, but was {type(outputs)}.")

        # Initialize
        old_shape = list(tf.keras.ops.shape(outputs))

        # Flatten along self._axes_ to fit permutation matrix
        outputs = utt.flatten_along_axes(x=outputs, axes=self._axes_)

        # Shuffle
        outputs = tf.gather(outputs, self._inverse_permutation_, axis=self._axes_[0])

        # Unflatten to restore original shape
        reconstructed_inputs = tf.keras.ops.reshape(outputs, newshape=old_shape)
        
        # Outputs
        return reconstructed_inputs
    
    def compute_jacobian_determinant(self, x: Union[tf.Tensor, tf.keras.KerasTensor]) -> tf.Tensor:
        """Computes the Jacobian determinant of this layer's transformation on logarithmic scale. This is simply zero since permutations are volume-preserving.
        This function supports symbolic execution.

        :param x: The input to this layer.
        :type x: Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`]
        :return: jacobian_determinant (Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`]) - The Jacobian determinant on logarithmic scale of shape [batch-size]."""

        # Create vector of zeros with length batch-size
        jacobian_determinant = 0.0 * tf.keras.ops.sum(x, axis=list(range(1, len(tf.keras.ops.shape(x)))))

        # Outputs
        return jacobian_determinant
    
    def get_config(self) -> Dict[str, Any]:
        
        # Super
        config = super(Permutation, self).get_config()
        
        # Update config
        config.update(
            {"permutation": self._forward_permutation_.numpy().tolist()}
        )
        
        # Outputs
        return config

@tf.keras.utils.register_keras_serializable()
class ShufflePermutation(Permutation):
    """A :py:class:`~gyoza.modelling.flow_layers.Permutation` subclass that shuffles its inputs along the specified ``axes``. The permutation 
    used for shuffling is randomly chosen once during initialization. Thereafter it is saved as a private attribute. Shuffling is thus 
    deterministic. **Important:** The shuffle function is defined on a vector, yet by the requirement of :py:class:`~gyoza.modelling.flow_layers.Permutation`, inputs 
    to this layer are allowed to have more than one axis in ``axes``. As described in :py:class:`~gyoza.modelling.flow_layers.Permutation`, inputs are first 
    flattened along ``axes`` and thus the shuffling can be applied. For background information see :py:class:`~gyoza.modelling.flow_layers.Permutation`.
    
    :param shape: See base class :py:class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :py:class:`FlowLayer`.
    :type axes: List[int]
    """

    def __init__(self, shape: List[int], axes: List[int], **kwargs) -> None:

        # Super
        dimension_count = int(np.prod(shape))
        permutation = list(range(dimension_count)); random.shuffle(permutation)
        super(ShufflePermutation, self).__init__(shape=shape, axes=axes, permutation=permutation, **kwargs)
    
    @classmethod
    def from_config(cls: FlowLayerType, config: Dict[str, Any]) -> "ShufflePermutation":
        """
        Loads a :py:class:`~gyoza.modelling.flow_layers.ShufflePermutation` from its configuration dictionary. The dictionary is required to have been obtained via :py:meth:`~gyoza.modelling.flow_layers.ShufflePermutation.get_config`.
        
        :param cls: The type of this layer (i.e. :py:class:`~gyoza.modelling.flow_layers.ShufflePermutation`).
        :type cls: FlowLayerType
        :param config: The dictionary containing the configuration of the shuffle permutation.
        :type config: Dict[str, Any]
        :return: instance (:py:class:`~gyoza.modelling.flow_layers.ShufflePermutation`) - The constructed instance.
        """

        # Construct instance
        config.pop("permutation") # Permutation is constructed internally
        instance = cls(**config)
        
        # Outputs
        return instance
    
@tf.keras.utils.register_keras_serializable()
class HeavisidePermutation(Permutation):
    """A :py:class:`~gyoza.modelling.flow_layers.Permutation` subclass that swaps the first and second half of its inputs as inspired by the `Heaviside 
    <https://en.wikipedia.org/wiki/Heaviside_step_function>`_ function.  **Important:** The Heaviside function is defined on a vector, 
    yet by the requirement of :py:class:`~gyoza.modelling.flow_layers.Permutation`, inputs to this layer are allowed to have more than one axis in `axes`.
    As described in :py:class:`~gyoza.modelling.flow_layers.Permutation`, an input is first flattened along `axes` and thus the swopping can be applied. 
    For background information see :py:class:`~gyoza.modelling.flow_layers.Permutation`. If the number of dimensions to be permuted is odd-length,
    the permutation will move the first half excluding the middle element. 

    :param shape: See base class :py:class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :py:class:`FlowLayer`.
    :type axes: List[int]
    """

    def __init__(self, shape: List[int], axes: List[int], **kwargs) -> None:

        # Super
        dimension_count = int(np.prod(shape))
        permutation = list(range(dimension_count//2, dimension_count)) + list(range(dimension_count//2))
        super(HeavisidePermutation, self).__init__(shape=shape, axes=axes, permutation=permutation, **kwargs)

    @classmethod
    def from_config(cls: FlowLayerType, config: Dict[str, Any]) -> "HeavisidePermutation":
        """
        Loads a :py:class:`~gyoza.modelling.flow_layers.HeavisidePermutation` from its configuration dictionary. The dictionary is required to have been obtained via :py:meth:`~gyoza.modelling.flow_layers.HeavisidePermutation.get_config`.
        
        :param cls: The type of this layer (i.e. :py:class:`~gyoza.modelling.flow_layers.HeavisidePermutation`).
        :type cls: FlowLayerType
        :param config: The dictionary containing the configuration of the heaviside permutation.
        :type config: Dict[str, Any]
        :return: instance (:py:class:`~gyoza.modelling.flow_layers.HeavisidePermutation`) - The constructed instance.
        """

        # Construct instance
        config.pop("permutation") # Permutation is constructed internally
        instance = cls(**config)
        
        # Outputs
        return instance
    
@tf.keras.utils.register_keras_serializable()
class CheckerBoardPermutation(Permutation):
    """A :py:class:`~gyoza.modelling.flow_layers.Permutation` subclass that swaps the entries of its inputs as inspired by the `checkerboard <https://en.wikipedia.org/wiki/Check_(pattern)>`_
    pattern. Swapping is done in a way that preserves adjacency of among the swapped entries as much as possible. **Important:** The checkerboard pattern is usually
    defined on a matrix, i.e. 2 axes. Yet, here it is possible to specify any number of ``axes``. **Note:** If the total number
    of elements in the to-be-permuted vector is odd, then the last element will remain in place.

    :param axes: See base class :py:class:`FlowLayer`.
    :type axes: List[int]
    :param shape: See base class :py:class:`FlowLayer`. 
    :type shape: List[int]
    """

    @staticmethod
    def is_end_of_axis(index: int, limit: int, direction: int) -> bool:
        """Determines whether an ``index`` iterated in ``direction`` is at the end of a given axis.

        :param index: The index to be checked.
        :type index: int
        :param limit: The number of elements along the axis. An index is considered to be at the end if it is equal to `limit`-1 
            and ``direction`` == 1. 
        :type limit: int
        :param direction: The direction in which the index is iterated. A value of 1 indicates incremental, -1 indicates decremental.
        :type direction: int
        :return: is_end_of_axis (bool) - An indicator for whether the endpoint has been reached.
        """
        if direction == 1: # Incremental
            return index == limit -1
        else: # Decremental
            return index == 0

    @staticmethod
    def generate_rope_indices(shape: List[int]) -> Generator[int, None, None]:
        """Generates indices to traverse a tensor of given ``shape``. The traversal follows a rope fitted along the axes (0 to len(``shape``)-1) by prioritizing
        later axes before earlier axes.

        :param shape: The shape of the tensor to be traversed.
        :type shape: List[int]
        :yield: current_indices (List[int]) - The indices pointing to the current cell, one index per axis.
        """

        dimension_count = int(np.prod(shape))
        current_indices = [0] * len(shape)
        yield current_indices
        directions = [1] * len(shape)
        for d in range(dimension_count):
            # Increment index counter (with carry on to next axis if needed)
            for s in range(len(shape)-1,-1,-1): 
                if CheckerBoardPermutation.is_end_of_axis(index=current_indices[s], limit=shape[s], direction=directions[s]):
                    directions[s] = -directions[s]
                else:
                    current_indices[s] += directions[s]
                    break

            yield list(current_indices)

    def __init__(self, shape: List[int], axes: List[int], **kwargs) -> None:

        # Set up permutation vector
        dimension_count = int(np.prod(shape))
        tensor = np.reshape(np.arange(dimension_count), shape)
        rope_values = [None] * dimension_count
        
        # Unravel tensor
        rope_index_generator = CheckerBoardPermutation.generate_rope_indices(shape=shape)
        for d in range(dimension_count): rope_values[d] = tensor[tuple(next(rope_index_generator))]

        # Swop every two adjacent values
        for d in range(0, dimension_count - 1, 2): # Skips the last element if odd
            tmp = rope_values[d]
            rope_values[d] = rope_values[d+1]
            rope_values[d+1] = tmp

        # Ravel tensor
        rope_index_generator = CheckerBoardPermutation.generate_rope_indices(shape=shape)
        for d in range(dimension_count): tensor[tuple(next(rope_index_generator))] = rope_values[d]

        # Flattened tensor now gives permutation
        permutation = list(np.reshape(tensor, [-1]))

        # Super
        super(CheckerBoardPermutation, self).__init__(shape=shape, axes=axes, permutation=permutation, **kwargs)
    
    @classmethod
    def from_config(cls: FlowLayerType, config: Dict[str, Any]) -> "CheckerBoardPermutation":
        """
        Loads a :py:class:`~gyoza.modelling.flow_layers.CheckerBoardPermutation` from its configuration dictionary. The dictionary is required to have been obtained via :py:meth:`~gyoza.modelling.flow_layers.Permutation.get_config`.
        
        :param cls: The type of this layer (i.e. :py:class:`~gyoza.modelling.flow_layers.CheckerBoardPermutation`).
        :type cls: FlowLayerType
        :param config: The dictionary containing the configuration of the checkerboard permutation.
        :type config: Dict[str, Any]
        :return: instance (:py:class:`~gyoza.modelling.flow_layers.CheckerBoardPermutation`) - The constructed instance. 
        """

        # Construct instance
        config.pop("permutation") # Permutation is constructed internally
        instance = cls(**config)
        
        # Outputs
        return instance
    
@tf.keras.utils.register_keras_serializable()
class Coupling(FlowLayer):
    r"""This subclass of :py:class:`~gyoza.modelling.flow_layers.FlowLayer` couples the input provided to its :py:meth:`~gyoza.modelling.flow_layers.Coupling.call` 
    method with itself.
    
    **Theoretical Background:**

    The :py:meth:`~gyoza.modelling.flow_layers.Coupling.call` method implements the following formulas:
    
    .. math::
        :nowrap:

        \begin{align*}
            x_1 &= w * x \\
            x_2 &= (1-w) * x \\
            y_1 &= x_1 \\
            y_2 &= f(x_2, g(x_1)) \\
            y   &= y_1 + y_2,
        \end{align*}

    where :math:`x`, :math:`x_1`, :math:`x_2`, :math:`w`, :math:`y_1`, :math:`y_2` and :math:`y` all have the same shape. While the :math:`x`'s and the
    :math:`y`'s are all real-valued, the mask :math:`w` is binary. As can be seen from the formulas, :math:`w` is used to separate the inputs
    :math:`x` into two halves which are then saved to :math:`x_1` and :math:`x_2`. While half of the output entries in :math:`y` are directly set to
    :math:`x_1` via :math:`y_1`, the main contribution of this layer is in the computation of the other half of the output via :math:`y_2`. That is,
    the function :math:`g` is used to compute the so-called coupling parameters which then feed into the coupling law :math:`f` to compute :math:`y_2`.
    The computation of coupling parameters can be an arbitrarily complicated and not necessarily invertible mapping, here executed by the :py:class:`tensorflow.keras.Layer`
    or even :py:class:`tensorflow.keras.Model` called ``compute_coupling_parameters``. The coupling parameters can be a tensor of any shape (or even a list of tensors), as long as they are compatible with the coupling law :math:`f`.
    The coupling law :math:`f` is a simple combination of :math:`x_2` and the coupling parameters, e.g. a sum or a product that has to be trivially invertible
    and have a tractable Jacobian determinant.
    
    **Usage Instructions:**
    
    To couple your data, you need to instantiate a subclass of :py:class:`~gyoza.modelling.flow_layers.Coupling` that
    implements your preferred coupling law :math:`f`, e.g. :py:class:`~gyoza.modelling.flow_layers.AdditiveCoupling`. You also need to provide a suitable mask :math:`w` from the :py:mod:`~gyoza.modelling.masks` module
    to allow the coupling layer to filter out half of the entries of your data tensor and couple them with the other half. It is important to specify
    the same ``shape`` and ``axes`` for both the coupling layer and the mask so that they can work together correctly.

    You will also need to provide a suitable ``compute_coupling_parameters`` layer/ model that computes the coupling parameters. The coupling layer's
    :py:meth:`~gyoza.modelling.flow_layers.FlowLayer.call` method will take its ``inputs``, use the ``mask`` to cancel out half of the entries and then
    feed the result (which has the same shape as ``inputs``) to your ``compute_coupling_parameters`` layer/ model. Read the description of your chosen
    coupling layer's subclass to see what requirements it has on the outputs of ``compute_coupling_parameters``. Simple
    :py:class:`~gyoza.modelling.flow_layers.Coupling` subclasses like :py:class:`~gyoza.modelling.flow_layers.AdditiveCoupling` will expect the output of
    ``compute_coupling_parameters`` to have the same shape as ``inputs`` which allows for simple element-wise addition or multiplication with ``inputs``
    before the negative ``mask`` will keep only the entries relevant to form :math:`y_2`. But, more complex coupling layers might have different
    requirements.

    It is important to realize that the halving of the inputs will effectively lead the coupling layer to couple only half of the input dimensions
    while leaving the other half unchanged. To allow all input dimensions to be transformed by coupling layers, it is common practice to succeed the coupling layer
    with a :py:class:`~gyoza.modellling.flow_layers.Permutation` that swaps the positions of the dimensions of the two halves. Importantly, choose a permutation
    that is equivalent to the mask of your choice, e.g. a :py:class:`~gyoza.modelling.flow_layers.CheckerBoardPermutation` for a :py:class:`~gyoza.modelling.masks.CheckerBoardMask`.
    Then, apply another coupling layer with the same mask, followed by another permutation of same kind to move all dimensions back to their original positions. 
    This results in a complete coupling block that can be repeated several times in a normalizing flow to allow for increasingly complex transformations of all input dimensions.

    **Subclass Instructions:** 
    
    Any subclasses of this class must implement

    - _couple_ to implement the usually trivial coupling law :math:`f` used in :py:meth:`~gyoza.modelling.flow_layers.Coupling.call`, e.g. addition or multiplication.
    - _decouple_ to implement the inverse of the coupling law :math:`f` used in :py:meth:`~gyoza.modelling.flow_layers.Coupling.invert`.
    - compute_log_jacobian_determinant to compute the logarithmic Jacobian determinant of the coupling law :math:`f`.
    
    :param shape: See base class :py:class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :py:class:`FlowLayer`.
    :type axes: List[int]
    :param compute_coupling_parameters: A :py:class:`tensorflow.keras.Layer` or even :py:class:`tensorflow.keras.Model` that maps its input tensor to coupling parameters. This model can be arbitrarily complex and does not need to be invertible, but its outputs must be compatible with the coupling law (:math:`f`) implemented by the specific :py:class:`~gyoza.modelling.flow_layers.Coupling` subclass. 
    :type compute_coupling_parameters: Union[:py:class:`tensorflow.keras.Layer`, :py:class:`tensorflow.keras.Model`]
    :param mask: The mask used to keep one half of the data while cancelling the other half. The mask is evaluated in its positive configuration (i.e. 1s select entries, 0s cancel entries, see :py:meth:`~gyoza.modelling.masks.Mask.call`) to select :math:`x_1` and in its negative configuration (i.e. 1s and 0s get flipped) to select :math:`x_2`.
    :type mask: :py:class:`~gyoza.modelling.masks.Mask`
    
    References:

        - Dinh, L., Sohl-Dickstein, J. & Bengio, S. (2016).
          "Density estimation using Real NVP"
          `arXiv:1605.08803 <https://arxiv.org/abs/1605.08803>`_

        - Dinh, L., Krueger, D. & Bengio, Y. (2015)
          "NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION" 
          `arXiv:1410.8516 <https://arxiv.org/abs/1410.8516>`_
    """

    def __init__(self, shape: List[int], axes: List[int], compute_coupling_parameters: tf.keras.Layer, mask: mms.Mask, **kwargs) -> None:
        
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
        """(Union[:py:class:`tensorflow.keras.Layer`, :py:class:`tensorflow.keras.Model`]) A keras model or layer that maps its input tensor to coupling parameters. This layer/ model can be arbitrarily complex and does not need to be invertible, but its outputs must be compatible with the coupling law (:math:`f`) implemented by the specific :py:class:`~gyoza.modelling.flow_layers.Coupling` subclass. """
        
        self._mask_ = mask
        """(:py:class:`~gyoza.modelling.masks.Mask`) - The mask used to keep one half of the data while cancelling out the other half."""
    
    def build(self, input_shape: Union[tf.TensorShape, Tuple, List]) -> None:
        # 1. Ensure the sub-model is built with the incoming shape
        if not self._compute_coupling_parameters_.built:
            self._compute_coupling_parameters_.build(input_shape)
        
        # 2. Call the base build to mark this layer as built
        super().build(input_shape)

    def call(self, inputs: Union[tf.Tensor, tf.keras.KerasTensor]) -> Tuple[Union[tf.Tensor, tf.keras.KerasTensor], Union[tf.Tensor, tf.keras.KerasTensor]]:

        # Input validity
        if not (isinstance(inputs, tf.Tensor) or isinstance(inputs, tf.keras.KerasTensor)): raise TypeError(f"The inputs provided to ~gyoza.modelling.flow_layers.Coupling.call is assumed to be an eager numeric tensorflow.Tensor or symbolic tensorflow.keras.KerasTensor but was {type(inputs)}.")
        if not (len(tf.keras.ops.shape(inputs)) > 1): raise ValueError(f"The inputs provided to ~gyoza.modelling.flow_layers.Coupling.call is assumed to have at least two axis, the first of which should be the batch axis, but the given inputs has shape {tf.keras.ops.shape(inputs)}.")

        # Split x
        x_1 = self._mask_.call(inputs=inputs)

        # Compute parameters
        coupling_parameters = self._compute_coupling_parameters_(x_1)
        
        # Couple
        y_hat_1 = x_1
        y_hat_2 = self._mask_.call(inputs=self._couple_(inputs=inputs, coupling_parameters=coupling_parameters), is_positive=False)

        # Combine
        y_hat = y_hat_1 + y_hat_2

        # Compute Jacobian determinant
        jacobian_determinant = self.compute_jacobian_determinant(x=inputs)

        # Outputs
        return y_hat, jacobian_determinant
    
    def _couple_(self, inputs: Union[tf.Tensor, tf.keras.KerasTensor], coupling_parameters: Union[tf.Tensor, tf.keras.KerasTensor, List[tf.Tensor], List[tf.keras.KerasTensor]]) -> tf.Tensor:
        """This function implements an invertible coupling law :math:`f` for ``inputs`` and ``coupling_parameters``.
        Subclasses must implement this method according to the coupling law they provide. There is no need to apply the negative mask to the ``inputs`` here,
        since this is done afterwards by :py:meth:`~gyoza.modelling.flow_layers.Coupling.call`.
        
        :param inputs: The ``inputs`` to :py:meth:`~gyoza.modelling.flow_layers.Coupling.call`. This can be treated as :math:`x_2` in the formulas given in the :py:class:`~gyoza.modelling.flow_layers.Coupling` description, although the negative mask will only be applied by :py:meth:`~gyoza.modelling.flow_layers.Coupling.call` after :py:meth:`~gyoza.modelling.flow_layers.Coupling._couple_` returned.
        :type inputs: Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`]
        :param coupling_parameters: The coupling parameters that shall be used to transform ``inputs``, corresponding to :math:`g(x_1)` in the formulas given in the :py:class:`~gyoza.modelling.flow_layers.Coupling` description. 
        :type coupling_parameters: Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`, List[:py:class:`tensorflow.Tensor`], List[:py:class:`tensorflow.keras.KerasTensor`]]
        :return: outputs (:py:class:`tensorflow.Tensor`) - The coupled tensor. The negative mask will be applied to this output by :py:meth:`~gyoza.modelling.flow_layers.Coupling.call` to obtain :math:`y_2` in the formulas given in the :py:class:`~gyoza.modelling.flow_layers.Coupling` description."""

        raise NotImplementedError("Subclasses must implement _couple_")
    
    def _decouple_(self, outputs: tf.Tensor, coupling_parameters: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        """This function is the inverse of :py:meth:`~gyoza.modelling.flow_layers.Coupling._couple_`.
        Subclasses must implement this method according to the coupling law they proovide. There is no need to apply the negative mask here,
        since this is done afterwards by :py:meth:`~gyoza.modelling.flow_layers.Coupling.invert`.
        This method only has to work in eager mode, thus not in symbolic mode, since it is only called from :py:meth:`~gyoza.modelling.flow_layers.Coupling.invert`.
        
        :param outputs: The outputs of :py:meth:`~gyoza.modelling.flow_layers.Coupling.call`, i.e. :math:`y`. This can be treated as :math:`y_2` in the formulas given in the :py:class:`~gyoza.modelling.flow_layers.Coupling` description, although the negative mask will only be applied by :py:meth:`~gyoza.modelling.flow_layers.Coupling.invert` after :py:meth:`~gyoza.modelling.flow_layers.Coupling._decouple_` returned. 
        :type outputs: :py:class:`tensorflow.Tensor`
        :param coupling_parameters: The coupling parameters corresponding to :math:`g(x_1)` in the formulas given in the :py:class:`~gyoza.modelling.flow_layers.Coupling` description. 
        :type coupling_parameters: Union[:py:class:`tensorflow.Tensor`, List[:py:class:`tensorflow.Tensor`]]
        :return: reconstructed_inputs (:py:class:`tensorflow.Tensor`) - The decoupled tensor corresponding to :math:`x_2` in the formulas given in the :py:class:`~gyoza.modelling.flow_layers.Coupling` description. The negative mask will be applied to this outout by :py:meth:`~gyoza.modelling.flow_layers.Coupling.invert` to obtain said :math:`x_2`."""

        raise NotImplementedError("Subclasses must implement _decouple_")
    
    def invert(self, outputs: tf.Tensor) -> tf.Tensor:
        
        # Input validity
        if not isinstance(outputs, tf.Tensor): raise TypeError(f"The outputs argument provided to ~gyoza.modelling.flow_layers.Coupling.invert is expected to be a tensorflow.Tensor, but was {type(outputs)}.")
                                      
        # Split
        y_hat_1 = self._mask_.call(inputs=outputs)

        # Compute parameters
        coupling_parameters = self._compute_coupling_parameters_(y_hat_1)
        
        # Decouple
        x_1 = y_hat_1
        x_2 = self._mask_.call(inputs=self._decouple_(outputs=outputs, coupling_parameters=coupling_parameters), is_positive=False)

        # Combine
        reconstructed_inputs = x_1 + x_2

        # Outputs
        return reconstructed_inputs
    
    def get_config(self) -> Dict[str, Any]:
        
        # Super
        config = super(Coupling, self).get_config()
        
        # Update config
        config.update(
            {"compute_coupling_parameters": tf.keras.layers.serialize(self._compute_coupling_parameters_), 
             "mask": tf.keras.layers.serialize(self._mask_)}
        )
        
        # Outputs
        return config
    
    @classmethod
    def from_config(cls: Type[FlowLayerType], config: Dict[str, Any]) -> FlowLayerType:
        """
        Loads a :py:class:`~gyoza.modelling.flow_layers.Coupling` layer from its configuration dictionary. The dictionary is required to have been obtained via :py:meth:`~gyoza.modelling.flow_layers.Coupling.get_config`.
        
        :param cls: The specific type of this layer (i.e. Coupling or one of its subclasses).
        :type cls: FlowLayerType
        :param config: The dictionary containing the configuration of the coupling layer.
        :type config: Dict[str, Any]
        :return: instance (FlowLayerType) - The constructed instance.
        """

        # Construct instance
        compute_coupling_parameters = tf.keras.layers.deserialize(config.pop("compute_coupling_parameters"))
        mask = tf.keras.layers.deserialize(config.pop("mask"))
        instance = cls(compute_coupling_parameters=compute_coupling_parameters, mask=mask, **config)
        
        # Outputs
        return instance
    
@tf.keras.utils.register_keras_serializable()
class AdditiveCoupling(Coupling):
    """This coupling layer implements an additive coupling law of the form :math:`f(x_2, g(x_1)) = x_2 + g(x_1)`, where :math:`g` is the
    ``compute_coupling_parameters`` function provided as argument when constructing an instance of this class. For details on the
    encapsulating theory refer to :py:class:`Coupling` or the paper by Dinh, Sohl-Dickstein and Bengio (2016) referenced below.
    
    References:

        - Dinh, L., Sohl-Dickstein, J. & Bengio, S. (2016).
          "Density estimation using Real NVP"
          `arXiv:1605.08803 <https://arxiv.org/abs/1605.08803>`_

    :param shape: See base class :py:class:`~gyoza.modelling.flow_layers.FlowLayer`.
    :type shape: List[int]
    :param axes: See superclass :py:class:`~gyoza.modelling.flow_layers.FlowLayer`.
    :type axes: List[int]
    :param compute_coupling_parameters: See base class :py:class:`~gyoza.modelling.flow_layers.Coupling`.
    :type compute_coupling_parameters: Union[:py:class:`tensorflow.keras.Layer`, :py:class:`tensorflow.keras.Model`]
    :param mask: See base class :py:class:`~gyoza.modelling.flow_layers.Coupling`.
    :type mask: :py:class:`~gyoza.modelling.masks.Mask`
    """

    def __init__(self, shape: List[int], axes: List[int], compute_coupling_parameters: tf.keras.Layer, mask: mms.Mask, **kwargs) -> None:
        
        # Super
        super(AdditiveCoupling, self).__init__(shape=shape, axes=axes, compute_coupling_parameters=compute_coupling_parameters, mask=mask, **kwargs)

    def _couple_(self, inputs: Union[tf.Tensor, tf.keras.KerasTensor], coupling_parameters: Union[tf.Tensor, tf.keras.KerasTensor]) -> tf.Tensor:
        
        # Couple
        outputs = inputs + coupling_parameters

        # Outputs
        return outputs
    
    def _decouple_(self, outputs: tf.Tensor, coupling_parameters: tf.Tensor) -> tf.Tensor:
        
        # Decouple
        reconstructed_inputs = outputs - coupling_parameters

        # Outputs
        return reconstructed_inputs
        
    def compute_jacobian_determinant(self, x: Union[tf.Tensor, tf.keras.KerasTensor]) -> tf.Tensor:
        """Computes the Jacobian determinant of this layer's transformation on logarithmic scale. This is simply zero for each instance since additive couplings are volume-preserving.

        :param x: The input to this layer.
        :type x: Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`]
        :return: jacobian_determinant (Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`]) - The Jacobian determinant on logarithmic scale of shape [batch-size]."""
        
        # Create a vector of zeros with length batch-size
        jacobian_determinant = 0.0 * tf.keras.ops.sum(x, axis=list(range(1, len(tf.keras.ops.shape(x)))))

        # Outputs
        return jacobian_determinant
    
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
        jacobian_determinant = 0.0
        for axis in self._mask_._axes_:
            jacobian_determinant += tf.keras.ops.reduce_sum(logarithmic_scale, axis=axis)

        # Outputs
        return jacobian_determinant
'''

@tf.keras.utils.register_keras_serializable()
class ActivationNormalization(FlowLayer):
    r"""Applies a trainable location- and scale-transformation of the following form to its inputs:

    .. math::
        \hat{y} = \frac{x - l}{s}, \quad l \in \mathbb{R}, \quad s > 0,

    where :math:`l` and :math:`s` are the `location` and `scale` parameters. For each dimension covered
    by the ``axes`` argument, a separate location and scale parameter is learned. For example, suppose you
    construct this layer with ``axes`` == [1,2] and ``shape`` == [width, height]. If you then provide
    ``inputs`` of shape [batch-size, width, height, channel-count] to this layer's :py:meth:`~~gyoza.modelling.flow_layers.ActivationNormalization.call` 
    method, one location and one scale parameter will be learned for each pixel position (i.e. 2 * width * height many parameters).
    The above location- and scale-transformation will then sweep across all other axes of the input
    (here: batch-size and channel-count) during the call routine.
    
    :param shape: See base class :py:class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :py:class:`FlowLayer`.
    :type axes: List[int]
    
    References:

        - Sankar, A., Keicher, M., Eisawy, R., Parida, A., Pfister, F., Kim, S., T. & Navab, N. (2021)
          "GLOWin: A Flow-based Invertible Generative Framework for Learning Disentangled Feature Representations in Medical Images" 
          `arXiv:2103.10868 <https://arxiv.org/abs/2103.10868>`_
        
        - Esser, P., Rombach, R., & Ommer, B. (2020). 
          "A Disentangling Invertible Interpretation Network for Explaining Latent Representations." 
          `arXiv:2004.13166 <https://arxiv.org/abs/2004.13166>`_
    """
    
    class _PositiveConstraint_(tf.keras.constraints.Constraint):
        """Constraint enforcing strictly positive scale parameters."""
        def __call__(self, w: tf.Variable):
            return tf.clip_by_value(w, clip_value_min=1e-6, clip_value_max=w.value.dtype.max)

    def __init__(self, shape: List[int], axes: List[int], **kwargs) -> None:

        # Super
        super(ActivationNormalization, self).__init__(shape=shape, axes=axes, **kwargs)

        # Attributes
        self._location_ = self.add_weight(shape = self._shape_,
                                           initializer = "zeros",
                                           dtype = tf.keras.backend.floatx(),
                                           trainable = True,
                                           name="_location_")
        """The value by which each data point shall be translated."""
        
        self._scale_ = self.add_weight(shape = self._shape_,
                                        initializer = "ones",
                                        dtype = tf.keras.backend.floatx(), 
                                        trainable=True, 
                                        constraint=ActivationNormalization._PositiveConstraint_(),
                                        name="_scale_")
        """The value by which each data point shall be scaled. This variable is constrained to be positive."""

    def _prepare_variables_for_computation_(self, x: Union[tf.Tensor, tf.keras.KerasTensor]) -> Tuple[tf.Variable, tf.Variable]:
        """Prepares the variables for computation with data. This ensures variable shapes are compatible with ``x``.
        
        :param x: The input to be passed through :py:meth:`call`. 
        :type x: Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`]

        :return: 
            - location (tensorflow.Variable): The :py:attr:`_location_` attribute shaped to fit ``x``. 
            - scale (tensorflow.Variable): The :py:attr:`_scale_` attribute shaped to fit ``x``."""

        # Shape variables to fit x
        axes = list(range(len(tf.keras.ops.shape(x))))
        for axis in self._axes_: axes.remove(axis)
        location = utt.expand_axes(x=self._location_, axes=axes)
        scale = utt.expand_axes(x=self._scale_, axes=axes)
        
        # Outputs
        return location, scale

    def call(self, inputs: Union[tf.Tensor, tf.keras.KerasTensor]) -> Tuple[Union[tf.Tensor, tf.keras.KerasTensor], Union[tf.Tensor, tf.keras.KerasTensor]]:
        
        # Input validity
        if not (isinstance(inputs, (tf.Tensor, tf.keras.KerasTensor))): raise TypeError(f"The inputs provided to ~gyoza.modelling.flow_layers.ActivationNormalization.call is assumed to be an eager numeric tensorflow.Tensor or symbolic tensorflow.keras.KerasTensor but was {type(inputs)}.")
        if not (len(tf.keras.ops.shape(inputs)) > 1): raise ValueError(f"The inputs provided to ~gyoza.modelling.flow_layers.ActivationNormalization.call is assumed to have at least two axis, the first of which should be the batch axis, but the given inputs has shape {tf.keras.ops.shape(inputs)}.")

        # Transform
        location, scale = self._prepare_variables_for_computation_(x=inputs)
        y_hat = (inputs - location) / scale # Scale is positive due to constraint
        jacobian_determinant = self.compute_jacobian_determinant(x=inputs)
        # Outputs
        return y_hat, jacobian_determinant
        
    def invert(self, outputs: tf.Tensor) -> tf.Tensor:

        # Input validity
        if not isinstance(outputs, tf.Tensor): raise TypeError(f"The outputs argument provided to ~gyoza.modelling.flow_layers.ActivationNormalization.invert is expected to be a tensorflow.Tensor, but was {type(outputs)}.")

        # Transform
        location, scale = self._prepare_variables_for_computation_(x=outputs)
        reconstructed_inputs =  outputs * scale + location

        # Outputs
        return reconstructed_inputs
    
    def compute_jacobian_determinant(self, x: Union[tf.Tensor, tf.keras.KerasTensor]) -> tf.Tensor:
        """Computes the Jacobian determinant of this layer's transformation on logarithmic scale. This is computed as the negative sum of ln(`scale`) over
        all `scale` parameters held by this layer, multiplied by the number of remaining dimensions located on the other axes of ``x``.
        This function supports symbolic execution.

        :param x: The input to this layer.
        :type x: Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`]
        :return: jacobian_determinant (Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`]) - The Jacobian determinant on logarithmic scale of shape [batch-size]."""
        
        # Count dimensions over remaining axes (for a single instance)
        dimension_count = 1
        for axis in range(1,len(tf.keras.ops.shape(x))):
            if axis not in self._axes_:
                dimension_count *= tf.keras.ops.shape(x)[axis] 
        
        # Compute logarithmic determinant
        # By defintion: sum across dimensions for ln(scale)
        jacobian_determinant = - dimension_count * tf.keras.ops.sum(tf.keras.ops.log(self._scale_)) # single instance, scale is positive due to constraint, the - sign in front is because the scale is used in the denominator
        jacobian_determinant = 0.0 * tf.keras.ops.sum(x, axis=list(range(1, len(tf.keras.ops.shape(x))))) + jacobian_determinant
        
        # Outputs
        return jacobian_determinant
    
@tf.keras.utils.register_keras_serializable()
class Reflection(FlowLayer):
    """This layer reflects a data point around `reflection_count` learnable normals using the `Householder transform 
    <https://en.wikipedia.org/wiki/Householder_transformation>`_. In this context, the normal is the unit-length vector orthogonal to
    the hyperplane of reflection. When ``axes`` contains more than a single entry, the input is first flattened along these axes, 
    then reflected and then unflattened to original shape.

    :param shape: See base class :py:class:`FlowLayer`.
    :type shape: List[int]
    :param axes: See base class :py:class:`FlowLayer`. **Important**: These axes are the axes of a tensor shape, e.g. axes 1 and 2 in a tensor of shape [batch-size, width, height] and hence distinct from the learnable reflection axes. The reflection axes are the normals to the hyperplanes of reflection and are learned internally as part of this layer.
    :type axes: List[int]
    :param reflection_count: The number of successive reflections that shall be executed. Expected to be at least 1.
    :type reflection_count: int

    References:
        
        - Meng, C., Song, Y., Song, J. & Ermon, S. (2020)
          "Gaussianization Flows"
          `arXiv:2003.01941 <https://arxiv.org/abs/2003.01941>`_
    """

    def __init__(self, shape: List[int], axes: List[int], reflection_count: int, **kwargs) -> None:
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
                                                     initializer = lambda shape, dtype=None: reflection_normals,
                                                     dtype = reflection_normals.dtype,
                                                     trainable = True,
                                                     name="reflection_normals",
                                                     constraint=tf.keras.constraints.UnitNorm(axis=1)) # name is needed for getting and setting weights
        """(:py:class:`tensorflow.Tensor`) - These are the axes along which an instance is reflected. Shape == [reflection count, dimension count] where dimension count is the product of the shape of the input instance along :py:attr:`self._axes_`."""
        
    def _reflect_(self, x: tf.Tensor) -> tf.Tensor:
        """This function executes all the reflections of self in a sequence by applying successive Householder reflections 
        defined by the :py:attr:`_reflection_normals_`. This method provides the backward reflection if 
        :py:attr:`self._inverse_mode` == True and forward otherwise.

        :param x: The flattened data of shape [..., dimension count], where dimension count is the product of the :py:attr:`_shape_` as 
            specified during initialization of self. It is assumed that all axes except for :py:attr:`_axes_` (again, see 
            initialization of self) are moved to ... in the aforementioned shape of `x`.
        :type x: :py:class:`tensorflow.Tensor`
        :return: x_new (:py:class:`tensorflow.Tensor`) - The rotated version of `x` with same shape.
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
    
    def call(self, inputs: Union[tf.Tensor, tf.keras.KerasTensor]) -> Tuple[Union[tf.Tensor, tf.keras.KerasTensor], Union[tf.Tensor, tf.keras.KerasTensor]]:
        
        # Input validity
        if not (isinstance(inputs, tf.Tensor) or isinstance(inputs, tf.keras.KerasTensor)): raise TypeError(f"The inputs provided to ~gyoza.modelling.flow_layers.Reflection.call is assumed to be an eager numeric tensorflow.Tensor or symbolic tensorflow.keras.KerasTensor but was {type(inputs)}.")
        if not (len(tf.keras.ops.shape(inputs)) > 1): raise ValueError(f"The inputs provided to ~gyoza.modelling.flow_layers.Reflection.call is assumed to have at least two axis, the first of which should be the batch axis, but the given inputs has shape {tf.keras.ops.shape(inputs)}.")

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
    
    def invert(self, outputs: tf.Tensor) -> tf.Tensor:
        
        # Input validity
        if not isinstance(outputs, tf.Tensor): raise TypeError(f"The outputs argument provided to ~gyoza.modelling.flow_layers.Reflection.invert is expected to be a tensorflow.Tensor, but was {type(outputs)}.")

        # Prepare self for inversion
        previous_mode = self._inverse_mode_
        self._inverse_mode_ = True

        # Call forward method (will now function as inverter)
        reconstructed_inputs, _ = self(inputs=outputs)

        # Undo the setting of self to restore the method's precondition
        self._inverse_mode_ = previous_mode

        # Outputs
        return reconstructed_inputs
    
    def compute_jacobian_determinant(self, x: Union[tf.Tensor, tf.keras.KerasTensor]) -> tf.Tensor:
        """Computes the Jacobian determinant of this layer's transformation on logarithmic scale. This is simply zero since reflections are volume-preserving.
        This function supports symbolic execution.

        :param x: The input to this layer.
        :type x: Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`]
        :return: jacobian_determinant (Union[:py:class:`tensorflow.Tensor`, :py:class:`tensorflow.keras.KerasTensor`]) - The Jacobian determinant on logarithmic scale of shape [batch-size]."""
        
        # It is known that Householder reflections have a determinant of -1 https://math.stackexchange.com/questions/504199/prove-that-the-determinant-of-a-householder-matrix-is-1
        # It is also known that det(AB) = det(A) det(B) https://proofwiki.org/wiki/Determinant_of_Matrix_Product
        # This layer applies succesive reflections as matrix multiplications and thus the determinant of the overall transformation is
        # -1 or 1, depending on whether an even or odd number of reflections are concatenated. Yet on logarithmic scale it is always 0.
        
        # Create vector of zeros with length batch-size
        jacobian_determinant = 0.0 * tf.keras.ops.sum(x, axis=list(range(1, len(tf.keras.ops.shape(x)))))

        # Outputs
        return jacobian_determinant
    
    def get_config(self) -> Dict[str, Any]:
        # Update the super config
        config = super().get_config()
        config.update({
            "reflection_count": self._reflection_count_
        })
        
        # Outputs
        return config
    

'''
How to extend this module:
- When adding a new layer, ensure that it is registered as keras serializable by using the decorator
  @tf.keras.utils.register_keras_serializable()
- Ensure that the new layer subclasses FlowLayer
- Ensure that the new layer implements the methods:
    - __init__
    - call
    - invert
    - compute_jacobian_determinant
    - get_config
    - from_config (as a class method)
    - If the new layer has sub-layers, ensure that they are properly serialized and deserialized in get_config and from_config
    - if the new layer has trainable parameters, ensure that they are properly initialized in __init__ and built in build (if needed). 
    - If the layer changes the output shape, e.g. as a Flatten layer would do, override the compute_output_shape method.
    - Ensure that all input validity checks are in place.
    - Add docstrings for the class and all methods.
    - Add references to relevant literature if applicable.
    - Add type hints for all methods.

Important:
- Remember that the automatic graph computation of tensorflow requires all operations to be done inside call or build methods.
- Ensure that any random operations are properly seeded if reproducibility is required.
- Use keras.ops instead of tensorflow operations where possible for better compatibility with keras layers and models.
- Ensure that the layer works with both eager execution and graph mode.
- Add unit tests for the new layer in the appropriate test module.


'''