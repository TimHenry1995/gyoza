import tensorflow as tf, numpy as np
import gyoza.utilities.tensors as utt
from abc import ABC
from typing import List
import copy as cp
from typing import TypeVar, Type, Dict, Any

MaskType = TypeVar("MaskType", bound="Mask")

@tf.keras.utils.register_keras_serializable()
class Mask(tf.keras.layers.Layer):
    """This class can be used to let half of the elements of a tensor :math:`x` pass through while cancelling out the other half.
    An input :math:`x` to this layer is expected to have ``shape`` along ``axes``. It is then flattened along these ``axes`` and 
    the ``mask`` is applied. Thereafter, :math:`x` is unflattened and returned. 
    
    :param axes: The axes along which the selection shall be applied.
    :type axes: :class:`List[int]`
    :param shape: The shape that input :math:`x` to this layer has along `axes`.
    :type shape: :class:`List[int]`
    :param mask: The mask to be applied to data passing through this layer. Its shape is expected to be a vector with 
        product(`shape`) many entries. 
    :type mask: :class:`tensorflow.Tensor`
    """

    def __init__(self, axes: List[int], shape: List[int], mask: tf.Tensor, **kwargs) -> None:

        # Super
        super(Mask, self).__init__(**kwargs)

        # Input validity
        if not (isinstance(axes, list) and all([isinstance(a, int) for a in axes])):
            raise TypeError(f"The input `axes` to gyoza.modelling.masks.Mask is expected to be a list of ints but was {type(axes)}")
        
        if not (isinstance(shape, list) and all([isinstance(s, int) for s in shape])):
            raise TypeError(f"The input `shape` to gyoza.modelling.masks.Mask is expected to be a list of ints but was {type(axes)}")
        if 0 in shape: raise ValueError(f"The input `shape` to gyoza.modelling.masks.Mask is expected to have all positive entries, thus skipping the batch-axis at index 0 but was {shape}")
        
        self._axes_ = list(axes)
        """(:class:`List[int]`) - The axes along which the selection shall be applied."""

        self._shape_ = list(shape)
        """(:class:`List[int]`) - The shape that input :math:`x` to this layer has along :py:attr:`gyoza.modelling.masks.Mask.axes`."""

        mask = tf.cast(mask, dtype=tf.keras.backend.floatx())
        mask = tf.keras.ops.reshape(mask, newshape=[-1]) # Ensure it is flattened
        self._mask_ = mask
        """(:class:`tensorflow.Tensor) - The mask to be applied to input :math:`x` to this layer."""

    def call(self, inputs: tf.Tensor, is_positive: bool = True) -> tf.Tensor:
        """Applies the binary mask of self to ``inputs``.

        :param inputs: The data to be masked. The expected shape of ``inputs`` depends on the axis and shape specified during initialization.
        :type inputs: :class:`tensorflow.Tensor`
        :param is_positive: Indicates whether the positive or negative mask version shall be applied. For the positive configuration, 1s let entries pass while 0 cancel them out. For the negative configuration, roles are switched.
        :type is_positive: bool, optional, default == True
        :return: x_masked (:class:`tensorflow.Tensor`) - The masked data of same shape as ``inputs``.
        """

        # Set parity of mask
        if is_positive: mask = self._mask_
        else: mask = 1 - self._mask_
        
        # Flatten x along self._axes_ 
        x_old_shape = tf.keras.ops.shape(inputs)
        inputs = utt.flatten_along_axes(x=inputs, axes=self._axes_)

        # Reshape mask
        axes = list(range(len(tf.keras.ops.shape(inputs))))
        axes.remove(self._axes_[0])
        mask = utt.expand_axes(x=mask, axes=axes) # Now has same shape as flat x along self._axes_[0] and singleton everywhere else
        
        # Mask
        x_masked = inputs * mask

        # Unflatten to restore original shape
        x_masked = tf.keras.ops.reshape(x_masked, newshape=x_old_shape)
        
        # Outputs
        return x_masked
    
    def arrange(self, x: tf.Tensor) -> tf.Tensor:
        """Arranges `x` into a vector such that all elements set to 0 by :py:meth:`gyoza.modelling.masks.Mask.mask` are enumerated first and all elements 
        that passed through the mask are enumerated last.

        :param x: The data to be arranged. The shape is assumed to be compatible with :py:meth:`gyoza.modelling.masks.Mask.mask`.
        :type x: :class:`tensorflow.Tensor`
        :return: x_flat (:class:`tensorflow.Tensor`) - The arranged version of `x` whose shape is flattened along the first axis
            of attribute :py:attr:`_axes_`.
        """
        
        # Flatten x along self._axes_ to fit from_to 
        x = utt.flatten_along_axes(x=x, axes=self._axes_)

        # Move self._axes_[0] to end
        x = utt.swop_axes(x=x, from_axis=self._axes_[0], to_axis=-1)

        # Matrix multiply
        from_to = Mask.__compute_from_to__(mask=self._mask_)
        x_new = tf.linalg.matvec(tf.transpose(from_to, perm=[1,0]), x)

        # Move final axis to self.__axis__[0]
        x_new = utt.swop_axes(x=x_new, from_axis=-1, to_axis=self._axes_[0])

        # Output
        return x_new
    
    def re_arrange(self, x_new: tf.Tensor) -> tf.Tensor:
        """This function is the inverse of :py:meth:`gyoza.modelling.masks.Mask.arrange`.
        
        :param x_new: The output of :py:meth:`gyoza.modelling.masks.Mask.arrange`.
        :type x: :class:`tensorflow.Tensor`
        
        :return: x (tensorflow.Tensor) - The input to :py:meth:`gyoza.modelling.masks.Mask.arrange`."""

        # Move self._axes_[0] to end
        x_new = utt.swop_axes(x=x_new, from_axis=self._axes_[0], to_axis=-1)
        
        # Matrix multiply
        from_to = Mask.__compute_from_to__(mask=self._mask_)
        x = tf.linalg.matvec(from_to, x_new)
        
        # Move final axis to self.__axis__[0]
        x = utt.swop_axes(x=x, from_axis=-1, to_axis=self._axes_[0])

        # Unflatten along self._axes_
        old_shape = x.shape[:self._axes_[0]] + self._shape_ + x.shape[self._axes_[0]+1:]
        x = tf.keras.ops.reshape(x, newshape=old_shape)

        # Outputs
        return x
    

    @staticmethod
    def __compute_from_to__(mask: tf.Tensor) -> tf.Tensor:
        """Sets up a matrix that can be used to arrange all elements of an input x (after flattening) such the ones marked with a 1 
        by the mask appear first while the ones marked with a zero occur last.
        
        :param mask: The mask that defines the mapping. It can be of arbitrary shape since it will be flattened internally.
        :type mask: tensorflow.Tensor.
        :return: from_to (tensorflow.Tensor) - The matrix that determines the mapping on flattened inputs. Note: to arrange elements
            of an input x one has to flatten x along the mask dimension first, then broadcast `from_to` to fit the new shape of x.
            After matrix multiplication of the two one needs to undo the flattening to get the arrange x."""

        # Determine indices
        from_indices = tf.concat([tf.where(mask), tf.where(1-mask)],0).numpy()[:,0].tolist()
        to_indices = list(range(len(from_indices)))
        
        # Set up matrix
        from_to = np.zeros(shape=[mask.shape[0], mask.shape[0]]) # Square matrix
        from_to[from_indices, to_indices] = 1
        from_to = tf.constant(from_to, dtype=tf.keras.backend.floatx())

        # Outputs
        return from_to
    
    def get_config(self) -> Dict[str, Any]:
        """
        Converts self to a configuration dictionary that can be used to reconstruct it via the corresponding mask class's :py:meth:`gyoza.modelling.masks.Mask.from_config`. 

        :return: config (Dict[str, Any]) - The configuration dictionary.
        """

        # Super
        config = super(Mask, self).get_config()
        
        # Update config
        config.update(
            {"shape": self._shape_, 
             "axes":self._axes_, 
             "mask": self._mask_.numpy().tolist()
            }
        )
        
        # Outputs
        return config
    
    @classmethod
    def from_config(cls: MaskType, config: Dict[str, Any]) -> MaskType:
        """
        Loads a :py:class:`gyoza.modelling.masks.Mask` from its configuration dictionary. The dictionary is required to have been obtained via the corresponding mask's :py:meth:`gyoza.modelling.masks.Mask.get_config`.
        
        :param cls: The specific type of the mask to be constructed.
        :type cls: MaskType
        :param config: The dictionary containing the configuration of the mask.
        :type config: Dict[str, Any]
        :return: instance (MaskType) - The constructed mask instance.
        """

        # No input checks since this method is called with valid inputs by keras

        # Construct instance
        mask = config.pop("mask")
        mask = tf.constant(mask, dtype=tf.keras.backend.floatx())
        instance = cls(mask=mask, **config)
        
        # Outputs
        return instance

@tf.keras.utils.register_keras_serializable()       
class HeavisideMask(Mask):
    """Applies a `Heaviside <https://en.wikipedia.org/wiki/Heaviside_step_function>`_ function to its input :math:`x`, e.g. 0001111. 
    **Important:** The Heaviside function is defined on a vector, yet by the requirement of :class:`~gyoza.modeling.masks.Mask`, inputs :math:`x` to this 
    layer are allowed to have more than one axis in ``axes``. As described in :class:`~gyoza.modeling.masks.Mask`, an input :math:`x` is first flattened 
    along ``axes`` and thus Heaviside can be applied. For background information see :class:`~gyoza.modeling.masks.Mask`.
    
    :param shape: See base class :class:`~gyoza.modeling.masks.Mask`.
    :type shape: List[int]
    :param axes: See base class :class:`~gyoza.modeling.masks.Mask`.
    :type axes: List[int]
    """

    def __init__(self, axes: List[int], shape: List[int], **kwargs) -> "HeavisideMask":
        
        # Set up mask
        mask = np.ones(shape=shape)
        mask = np.reshape(mask, [-1]) # Flattening
        mask[:mask.shape[0] // 2] = 0
        mask = tf.constant(mask) 

        # Super
        super(HeavisideMask, self).__init__(axes=axes, shape=shape, mask=mask, **kwargs)

    @classmethod
    def from_config(cls: MaskType, config: Dict[str, Any]) -> MaskType:
        
        # Construct instance
        config.pop("mask") # The mask is constructed internally and thus not needed from config
        instance = cls(**config)
        
        # Outputs
        return instance
    
@tf.keras.utils.register_keras_serializable()
class CheckerBoardMask(Mask):
    """A mask of ones and zeros arranged in a `checkerboard <https://en.wikipedia.org/wiki/Check_(pattern)>`_ . 
        
    :param axes: The axes along which the checkerboard pattern shall be applied. Assumed to be consecutive indices, e.g. 
        [2,3] or [3,4].
    :type axes: :class:`List[int]`
    :param shape: The shape of the mask along `axes`, e.g. 64*32 if an input :math:`x` has shape [10,3,64,32] and `axes` == [2,3].
    :type shape: :class:`List[int]`
    """

    def __init__(self, axes: List[int], shape: List[int], **kwargs) -> "CheckerBoardMask":
        
        # Set up mask
        mask = np.zeros(shape) 
        dimension_count = np.prod(shape)
        current_indices = [0] * len(shape)
        mask[tuple(current_indices)] = np.sum(current_indices) % 2
        for d in range(dimension_count):
            # Increment index counter (with carry on to next axes if needed)
            for s in range(len(shape)-1,-1,-1): 
                if current_indices[s] == shape[s] - 1:
                    current_indices[s] = 0
                else:
                    current_indices[s] += 1
                    break

            mask[tuple(current_indices)] = np.sum(current_indices) % 2

        mask = np.reshape(mask, [-1]) # Flatten
        mask = tf.constant(mask) 
        
        # Super
        super(CheckerBoardMask, self).__init__(axes=axes, shape=shape, mask=mask, **kwargs)

    @classmethod
    def from_config(cls, config) -> MaskType:
        
        # Construct instance
        config.pop("mask") # The mask is constructed internally and thus not needed from config
        instance = cls(**config)
        
        # Outputs
        return instance