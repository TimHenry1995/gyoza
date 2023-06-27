import tensorflow as tf
from typing import Dict, Any

class BasicFullyConnectedNet(tf.keras.Model):
    """This class provides a basic fully connected network. It passes data through several :class:`tensorflow.keras.layers.Dense` 
    layers and applies optional batch normalization. 
    
    :param int latent_channel_count: The number of channels maintained between intermediate layers. 
    :param int output_channel_count: The number of channels of the final layer.
    :param int depth: The number of layers to be used in between the input and output. If set to 0, there will only be a single 
        layer mapping from input to output. If set to 1, then there will be 1 intermediate layer, etc. 
    :param bool, optional use_tanh: Indicates whether each layer shall use the hyperbolic tangent activaction function. If set to False, 
        then a leaky relu is used. Defaults to False.
    :param bool, optional use_batch_normalization: Indicates whether each layer shall use batch normalization or not. Defaults to False.
    """

    def __init__(self, latent_channel_count:int, output_channel_count:int, depth: int):
        
        # Super
        super(BasicFullyConnectedNet, self).__init__()
        
        # Compile list of layers
        channel_counts = [latent_channel_count] * depth + [output_channel_count]
        layers = []
        for d in range(depth + 1):
            layers.append(tf.keras.layers.Dense(units=channel_counts[d]))
            if d < depth: layers.append(tf.keras.layers.LeakyReLU())
        
        # Attributes
        self.sequential = tf.keras.Sequential(layers)
        """Attribute that refers to the :class:`tensorflow.keras.Sequential` model collecting all layers of self."""

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """"""
        
        # Predict
        y_hat = self.sequential(x)

        # Outputs:
        return y_hat
    
class ChannelWiseConvolution2D(tf.keras.Model):
    """This class provides a sequential convolutional neural network that applies the same spatial filters to each channel.

    :param layer_count: The number of layers.
    :type layer_count: int
    :param conv2D_kwargs: The kew-word arguments for the :class:`tensorflow.keras.layers.Conv2D` layers that are used here.
        Important: The channel_axis is assumed to be the default, i.e. the last axis.
    """

    def __init__(self, layer_count:int = 3, conv2D_kwargs: Dict[str, Any] = {}):

        # Super
        super(ChannelWiseConvolution2D, self).__init__()

        # Create layers
        layers = [None] * (layer_count + 2)
        layers[0] = tf.keras.layers.Lambda(lambda x: tf.transpose(x[:,tf.newaxis,:,:,:], [0,4,2,3,1]))
        for l in range(layer_count): layers[l+1] = tf.keras.layers.Conv2D(**conv2D_kwargs)
        layers[-1] = tf.keras.layers.Lambda(lambda x: tf.squeeze(tf.transpose(x, [0,4,2,3,1])))

        # Attributes
        self.sequential = tf.keras.models.Sequential(layers=layers)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """"""
        
        # Predict
        y_hat = self.sequential(x)

        # Outputs:
        return y_hat