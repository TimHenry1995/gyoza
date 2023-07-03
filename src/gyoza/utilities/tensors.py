import tensorflow as tf
from typing import List
import copy as cp

def move_axis(x: tf.Tensor, from_index: int, to_index: int) -> tf.Tensor:
    """Moves an axis from from_index to to_index.
    
    :param x: A tensor of shape [..., k, ...] where k is at from_index.
    :type x: :class:`tensorflow.Tensor`
    :param from_index: The index of the axis before transposition.
    :type from_index: int
    :param to_index: The index of the axis after transposition.
    :type to_index: int
    :return: x_new (:class:`tensorflow.Tensor`): The tensor x transposed such that shape [..., k, ...] now has k at to_index."""
 
    # Input validity
    if from_index == -1: from_index = len(x.shape)-1
    if to_index == -1: to_index = len(x.shape)-1

    # Move axis
    new_order = list(range(len(x.shape)))
    del new_order[from_index]
    new_order.insert(to_index, from_index)
    x_new = tf.transpose(a=x, perm=new_order)

    # Outputs
    return x_new

def expand_axes(x: tf.Tensor, axes) -> tf.Tensor:
    """Expands x with singleton axes.
    
    :param x: The tensor to be expanded.
    :type x: :class:`tensorflow.Tensor`
    :param axes: The axes along which to expand. Their indices are assumed to be valid in the shape of ``x_new``. This means if, 
        e.g. ``x`` has two axes then ``axes`` may be, e.g. [0,1,3,5,6,7] where axes 2 and 4 are filled in order by ``x`` but 
        ``axes`` must not be, e.g. [0,1,3,5,6,10] because of the gap between 6 and 10 that would be introduced in the shape of
        ``x_new``.
    :type axes: :class:`List[tensorflow.Tensor]`
    
    :return: x_new (:class:`tensorflow.Tensor`) - The reshaped version of x with singletons along ''axes''."""

    # Initialize
    new_axis_count = len(x.shape) + len(axes)
    
    # Compatibility of new and old axes
    old_axes = list(range(new_axis_count))
    for axis in axes:
        # Input validity
        assert axis < new_axis_count, f"""The axis {axis} must be in the interval [0,{new_axis_count})."""
    
        # Exclude new axis from old axes
        old_axes.remove(axis)

    # Set new shape
    o = 0 # Iterates old axes
    new_shape = [1] * new_axis_count
    for axis in old_axes:
        new_shape[axis] = x.shape[o]
        o += 1

    x_new = tf.reshape(x, new_shape)

    # Outputs
    return x_new

def flatten_along_axes(x: tf.Tensor, axes: List[int]) -> tf.Tensor:
    """Flattens an input ``x`` along axes ``axes``.
    
    :param x: The input to be flattened. Assumed to have at least as many axes as indicated by ``axes``.
    :type x: :class:`tensorflow.Tensor`
    :param axes: The axes along which the input shall be flattened.
    :type axes: :class:`List[int]`
    :return: x_new (:class:`tensorflow.Tensor`) - The reshaped tensor ``x`` flattened along ``axes``."""

    # Exception handling
    if len(axes) == 0: return x

    # Reshape
    new_shape = list(x.shape)

    new_shape[axes[0]] = 1
    for a in axes: new_shape[axes[0]] *= x.shape[a]
    axes = cp.copy(axes); axes.reverse()
    for a in axes[:-1]: del new_shape[a]

    x_new = tf.reshape(x, new_shape) # Now has original shape except for axes which have been flattened

    # Outputs
    return x_new

def swop_axes(x: tf.Tensor, from_axis: int, to_axis: int) -> tf.Tensor:
    """Swops axes of ``x``.
    
    :param x: The input whose axes shall be swopped. Assumed to have at least as many axes as indicated by 
        ``from_axis`` and ``to_axis``.
    :type x: :class:`tensorflow.Tensor`
    :param from_axis: The axes to be swopped with ``to_axis``.
    :type from_axis: int
    :param to_axis: The axes to be swopped with ``from_axis``.
    :type to_axis: int
    :return: x_new (:class:`tensorflow.Tensor`) - The input which ``from_axis`` and ``to_axis`` swopped."""

    # Reshape
    axes = list(range(len(x.shape)))
    tmp = axes[to_axis]
    axes[to_axis] = axes[from_axis]
    axes[from_axis] = tmp
    x_new = tf.transpose(x, perm=axes)

    # Outputs
    return x_new 
        