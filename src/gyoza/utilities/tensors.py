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
    new_order[to_index] = from_index
    new_order[from_index] = to_index
    x_new = tf.transpose(a=x, perm=new_order)

    # Outputs
    return x_new