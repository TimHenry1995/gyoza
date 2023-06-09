import tensorflow as tf, numpy as np
import gyoza.utilities.tensors as utt
from abc import ABC
from typing import List

class SemiSelector(tf.keras.Model, ABC):
    """This class can be used to curate elements of a tensor x. As suggested by the name semi, half of x is selected
    while the other half is not."""

    def __init__(self, axes: List[int], mask: tf.Variable):
        """Constructor for this class. Subclasses can use it to store attributes.
        
        :param axes: The axes along which the selection shall be applied.
        :type axes: :class:`List[int]`
        :param mask: The mask to be applied to data passing through this layer. It should be an untrainable tensorflow Variable.
        :type mask: :class:`tensorflow.Variable`
        """

        # Super
        super(SemiSelector, self).__init__()

        self.__axes__ = axes
        """The axes along which the selection shall be applied."""

        self.__mask__ = mask
        """The mask to be applied to data passing through this layer."""

    def mask(self, x: tf.Tensor) -> tf.Tensor:
        """Applies a binary mask to ``x`` such that half of its entries are set to zero and the other half passes.

        :param x: The data to be masked. The expected shape of ``x`` depends on the axis and shape specified during initialization.
        :type x: :class:`tensorflow.Tensor`
        :return: x_masked (:class:`tensorflow.Tensor`) - The masked data of same shape as ``x``.
        """

        # Reshape mask to fit x
        axes = list(range(len(x.shape)))
        for axis in self.__axes__: axes.remove(axis) 
        mask = utt.expand_axes(x=self.__mask__, axes=axes)

        # Mask
        x_new = x*mask

        # Outputs
        return x_new

    def arrange(self, x: tf.Tensor) -> tf.Tensor:
        """Arranges ``x`` into a vector such that all elements set to 0 by :py:meth:`mask` are enumerated first and all elements 
        that passed the mask are enumerated last.

        :param x: The data to be arranged. The shape is assumed to be compatible with :py:meth:`mask`.
        :type x: :class:`tensorflow.Tensor`
        :return: x_flat (:class:`tensorflow.Tensor`) - The arranged version of ``x`` whose shape is arranged along the axes of 
            attribute :py:attr:`__axes__`.
        """
        
        # Select
        x_1 = tf.boolean_mask(tensor=x, mask=1-self.__mask__, axis=self.__axes__[0])
        x_2 = tf.boolean_mask(tensor=x, mask=  self.__mask__, axis=self.__axes__[0])

        # Concatenate
        x_new = tf.concat([x_1, x_2], axis=self.__axes__[0])

        # Output
        return x_new

class HeaviSide(SemiSelector):
    """Applies a one-dimensional Heaviside function of the shape 000111 to its input. Inputs are expected to have 1 spatial axes 
    located at ``axes`` with ``shape`` many elements.
    
    :param axes: The axes (here only one axis) along which the Heaviside mask shall be applied.
    :type axes: :class:`List[int]`
    :param shape: The number of units along ``axes``, e.g. [5] if an input x has shape [3,5,2] and ``axes`` == [1].
    :type shape: :class:`List[int]`
    :param is_positive: If set to True, then the first ``shape`` // 2 units are set to zero while the remainig ones pass. If set to 
        False, the mask is flipped.
    :type is_positive: bool, optional"""

    def __init__(self, axes: int, shape: int, is_positive: bool = True):
        
        # Input validity
        assert len(axes) == 1, f"There must be one axis instead of {len(axes)} along which the Heaviside shall be applied."
        assert len(shape) == 1, f"The shape input is equal to {shape}, but it must have one axis."

        # Set up mask
        mask = np.ones(shape, dtype=np.float32)
        mask[:shape[0] // 2] = 0
        if not is_positive: mask = 1 - mask
        mask = tf.Variable(initial_value=mask, trainable=False, dtype=tf.float32) 

        # Super
        super(HeaviSide, self).__init__(axes=axes, mask=mask)

class SquareWave1D(SemiSelector):
    """Applies a one-dimensional square wave of the shape 010101 to its input. Inputs are expected to have 1 spatial axis located at
    ``axes`` with ``shape`` many elements.
    
    :param axes: The axes (here only one axis) along which the square wave shall be applied.
    :type axes: :class:`List[int]`
    :param shape: The number of units along ``axes``, e.g. [5] if an input x has shape [3,5,2] and ``axes`` == [1].
    :type shape: :class:`List[int]`
    :param is_positive: If set to True, then units at odd indices pass while units at even indices are set to zero.
        If set to False, the mask is flipped.
    :type is_positive: bool, optional"""

    def __init__(self, axes: int, shape: int, is_positive: bool = True):
        
        # Input validity
        assert len(axes) == 1, f"There must be one axis instead of {len(axes)} along which the square-wave shall be applied."
        assert len(shape) == 1, f"The shape input is equal to {shape}, but it must have one axis."

        # Set up mask
        mask = np.ones(shape)
        mask[::2] = 0
        if not is_positive: mask = 1 - mask
        mask = tf.Variable(initial_value=mask, trainable=False, dtype=tf.float32) 

        # Super
        super(SquareWave1D, self).__init__(axes=axes, mask=mask)

class SquareWave2D(SemiSelector):
    """Applies a two-dimensional square wave, also known as checkerboard pattern to its input. Inputs are expected to have 2 spatial
    axes located at ``axes`` with ``shape`` units along those axes.
        
    :param axes: The two axes along which the square-wave pattern shall be applied. Assumed to be two consecutive indices.
    :type axes: :class:`List[int]`
    :param shape: The shape of the mask, e.g. 64*32 if an input x has shape [10,3,64,32] and ``axes`` == [2,3].
    :type shape: :class:`List[int]`
    :param is_positive: If set to True, the mask is equal to 1 where the sum of indices is odd and 0 otherwise. 
        If set to False, the mask is flipped.
    :type is_positive: bool, optional"""

    def __init__(self, axes: List[int], shape: List[int], is_positive: bool = True) -> None:
        # Input validity
        assert len(axes) == 2, f"There must be two axes instead of {len(axes)} along which the square-wave shall be applied."
        assert axes[1] == axes[0] + 1, f"The axes {axes} have to be two consecutive indices."
        assert len(shape) == 2, f"The shape input is equal to {shape}, but it must have two axes."

        # Set up mask
        mask = np.ones(shape) 
        mask[1::2,1::2] = 0
        mask[::2,::2] = 0
        if not is_positive: mask = 1 - mask
        mask = tf.Variable(initial_value=mask, trainable=False, dtype=tf.float32) 
        
        # Super
        super(SquareWave2D, self).__init__(axes=axes, mask=mask)