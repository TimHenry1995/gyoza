import tensorflow as tf
import copy as cp

def move_axis(x: tf.Tensor, from_index: int, to_index: int) -> tf.Tensor:
    """Moves an axis from from_index to to_index.
    
    Inputs:
    - x: A tensor of shape [..., k, ...] where k is at from_index.
    - from_index: The index of the axis before transposition.
    - to_index: The index of the axis after transposition.
    
    Outputs:
    - x_new: The tensor x transposed such that shape [..., k, ...] is now at to_index."""
 
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
    :param axes: The axes along which to expand. Their indices are assumed to be valid in the output shape. They must not 
        introduce gaps in the output shape. This means if, e.g. x has two axes then ``axes`` may be, e.g. [0,1,3,5,6,7] where
        axes 2 and 4 are filled in orde by x but ``axes`` must not be, e.g. [0,1,3,5,6,10] because of the gap between 6 and 10 
        that would be introduced in the output shape.
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