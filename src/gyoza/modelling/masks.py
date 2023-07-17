import tensorflow as tf, numpy as np
import gyoza.utilities.tensors as utt
from abc import ABC
from typing import List
import copy as cp

class Mask(tf.keras.Model, ABC):
    """This class can be used to curate elements of a tensor x. As suggested by the name semi, half of x is selected
    while the other half is not."""

    def __init__(self, axes: List[int], mask: tf.Tensor):
        """Constructor for this class. Subclasses can use it to store attributes.
        
        :param axes: The axes along which the selection shall be applied.
        :type axes: :class:`List[int]`
        :param mask: The mask to be applied to data passing through this layer. 
        :type mask: :class:`tensorflow.Tensor`
        """

        # Super
        super(Mask, self).__init__()

        self.__axes__ = axes
        """(:class:List[int]`) - The axes along which the selection shall be applied."""

        self.__mask__ = tf.cast(mask, dtype=tf.keras.backend.floatx())
        """(:class:`tensorflow.Tensor) - The mask to be applied to data passing through this layer."""

        self.__from_to__ = Mask.__compute_from_to__(mask=mask)
        """(:class:`tensorflow.Tensor) - A matrix that defines the mapping during :py:meth:`arrange` and :py:meth:`re_arrange`."""

        self.built = True # This is set to prevent a warning saying that serialzation for mask is skipped becuase mask is not built

    @staticmethod
    def __compute_from_to__(mask: tf.Tensor) -> tf.Tensor:
        """Sets up a matrix that can be used to arrange all elements of an input x (after flattening) such the ones marked with a 1 
        by the mask appear first while the ones marked with a zero occur last.
        
        :param mask: The mask that defines the mapping. It can be of arbitrary shape since it will be flattened internally.
        :type mask: tensorflow.Tensor.
        :return: from_to (tensorflow.Tensor) - The matrix that determines the mapping on flattened inputs. Note: to arrange elements
            of an input x one has to flatten x along the mask dimension first, then broadcast ``from_to`` to fit the new shape of x.
            After matrix multiplication of the two one needs to undo the flattening to get the arrange x."""

        # Flatten mask
        mask = tf.reshape(mask, [-1]) 

        # Determine indices
        from_indices = tf.concat([tf.where(mask), tf.where(1-mask)],0).numpy()[:,0].tolist()
        to_indices = list(range(len(from_indices)))
        
        # Set up matrix
        from_to = np.zeros(shape=[mask.shape[0],mask.shape[0]]) # Square matrix
        from_to[from_indices, to_indices] = 1
        from_to = tf.constant(from_to, dtype=tf.keras.backend.floatx())

        # Outputs
        return from_to

    def call(self, x: tf.Tensor, is_positive: bool = True) -> tf.Tensor:
        """Applies the binary mask of self to ``x``.

        :param x: The data to be masked. The expected shape of ``x`` depends on the axis and shape specified during initialization.
        :type x: :class:`tensorflow.Tensor`
        :param is_positive: Indicates whether the positive or negative mask version shall be applied, where negative == 1 - positive.
            Default is True.
        :type is_positive: bool, optional
        :return: x_masked (:class:`tensorflow.Tensor`) - The masked data of same shape as ``x``.
        """
        
        # Parity
        if is_positive: mask = self.__mask__
        else: mask = 1 - self.__mask__
        
        # Reshape mask to fit x
        axes = list(range(len(x.shape)))
        for axis in self.__axes__: axes.remove(axis) 
        mask = utt.expand_axes(x=mask, axes=axes)

        # Apply mask to x
        x_new = x * mask

        # Outputs
        return x_new

    def arrange(self, x: tf.Tensor) -> tf.Tensor:
        """Arranges ``x`` into a vector such that all elements set to 0 by :py:meth:`mask` are enumerated first and all elements 
        that passed the mask are enumerated last.

        :param x: The data to be arranged. The shape is assumed to be compatible with :py:meth:`mask`.
        :type x: :class:`tensorflow.Tensor`
        :return: x_flat (:class:`tensorflow.Tensor`) - The arranged version of ``x`` whose shape is flattened along the first axis
            of attribute :py:attr:`__axes__`.
        """
        
        # Flatten x along self.__axes__ to fit from_to 
        x = utt.flatten_along_axes(x=x, axes=self.__axes__)

        # Move self.__axes__[0] to end
        x = utt.swop_axes(x=x, from_axis=self.__axes__[0], to_axis=-1)

        # Matrix multiply
        x_new = tf.linalg.matvec(tf.transpose(self.__from_to__, perm=[1,0]), x)

        # Move final axis to self.__axis__[0]
        x_new = utt.swop_axes(x=x_new, from_axis=-1, to_axis=self.__axes__[0])

        # Output
        return x_new
    
    def re_arrange(self, x_new: tf.Tensor) -> tf.Tensor:
        """This function is the inverse of :py:meth:`arrange`.
        
        :param x_new: The output of :py:meth:`arrange`.
        :type x: :class:`tensorflow.Tensor`
        
        :return: x (tensorflow.Tensor) - The input to :py:meth:`arrange`."""

        # Move self.__axes__[0] to end
        x_new = utt.swop_axes(x=x_new, from_axis=self.__axes__[0], to_axis=-1)
        
        # Matrix multiply
        x = tf.linalg.matvec(self.__from_to__, x_new)
        
        # Move final axis to self.__axis__[0]
        x = utt.swop_axes(x=x, from_axis=-1, to_axis=self.__axes__[0])

        # Unflatten along self.__axes__
        old_shape = x.shape[:self.__axes__[0]] + self.__mask__.shape + x.shape[self.__axes__[0]+1:]
        x = tf.reshape(x, shape=old_shape)

        # Outputs
        return x
       
class Heaviside(Mask):
    """Applies a one-dimensional `Heaviside <https://en.wikipedia.org/wiki/Heaviside_step_function>`_ function of the shape 000111 to its input. Inputs are expected to have 1 spatial axes 
    located at ``axes`` with ``shape`` many elements.
    
    :param axes: The axes (here only one axis) along which the Heaviside mask shall be applied.
    :type axes: :class:`List[int]`
    :param shape: The number of units along ``axes``, e.g. [5] if an input x has shape [3,5,2] and ``axes`` == [1].
    :type shape: :class:`List[int]`
    """

    def __init__(self, axes: int, shape: int):
        
        # Input validity
        assert len(axes) == 1, f"There must be one axis instead of {len(axes)} along which the Heaviside shall be applied."
        assert len(shape) == 1, f"The shape input is equal to {shape}, but it must have one axis."

        # Set up mask
        mask = np.ones(shape=shape)
        mask[:shape[0] // 2] = 0
        mask = tf.constant(mask) 

        # Super
        super(Heaviside, self).__init__(axes=axes, mask=mask)

class SquareWaveSingleAxis(Mask):
    """Applies a one-dimensional `square wave <https://en.wikipedia.org/wiki/Square_wave>`_ of the shape 010101 to its input. Inputs are expected to have 1 spatial axis located at
    ``axes`` with ``shape`` many elements.
    
    :param axes: The axes (here only one axis) along which the square wave shall be applied.
    :type axes: :class:`List[int]`
    :param shape: The number of units along ``axes``, e.g. [5] if an input x has shape [3,5,2] and ``axes`` == [1].
    :type shape: :class:`List[int]`
    """

    def __init__(self, axes: int, shape: int):
        
        # Input validity
        assert len(axes) == 1, f"There must be one axis instead of {len(axes)} along which the square-wave shall be applied."
        assert len(shape) == 1, f"The shape input is equal to {shape}, but it must have one axis."

        # Set up mask
        mask = np.ones(shape=shape)
        mask[::2] = 0
        mask = tf.constant(mask) 

        # Super
        super(SquareWaveSingleAxis, self).__init__(axes=axes, mask=mask)

class SquareWaveTwoAxes(Mask):
    """Applies a two-dimensional square wave, also known as `checkerboard pattern <https://en.wikipedia.org/wiki/Check_(pattern)>`_ to its input. Inputs are expected to have 2 spatial
    axes located at ``axes`` with ``shape`` units along those axes.
        
    :param axes: The two axes along which the square-wave pattern shall be applied. Assumed to be two consecutive indices.
    :type axes: :class:`List[int]`
    :param shape: The shape of the mask, e.g. 64*32 if an input x has shape [10,3,64,32] and ``axes`` == [2,3].
    :type shape: :class:`List[int]`
    """

    def __init__(self, axes: List[int], shape: List[int]) -> None:
        # Input validity
        assert len(axes) == 2, f"There must be two axes instead of {len(axes)} along which the square-wave shall be applied."
        assert axes[1] == axes[0] + 1, f"The axes {axes} have to be two consecutive indices."
        assert len(shape) == 2, f"The shape input is equal to {shape}, but it must have two axes."

        # Set up mask
        mask = np.ones(shape) 
        mask[1::2,1::2] = 0
        mask[::2,::2] = 0
        mask = tf.constant(mask) 
        
        # Super
        super(SquareWaveTwoAxes, self).__init__(axes=axes, mask=mask)