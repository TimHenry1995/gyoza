import tensorflow as tf

class BasicFullyConnectedNet(tf.keras.Model):
    """This class provides a basic fully connected network. It essentially passes data through several
    :class:`tensorflow.keras.layers.Dense` layers and applies optional batch normalization. 
    
    :param int latent_channel_count: The number of channels maintained between intermediate layers. 
    :param int output_channel_count: The number of channels of the final layer.
    :param int depth: The number of layers to be used in between the input and output. If set to 0, there will only be a single 
        layer mapping from input to output. If set to 1, then there will be 1 intermediate layer, etc. 
    :param bool, optional use_tanh: Indicates whether each layer shall use the hyperbolic tangent activaction function. If set to False, 
        then a leaky relu is used. Defaults to False.
    :param bool, optional use_batch_normalization: Indicates whether each layer shall use batch normalization or not. Defaults to False."""

    def __init__(self, latent_channel_count:int, output_channel_count:int, depth: int, use_tanh:bool=False, use_batch_normalization:bool=False):
        
        # Super
        super(BasicFullyConnectedNet, self).__init__()
        
        # Compile list of layers
        channel_counts = [latent_channel_count] * depth + [output_channel_count]
        layers = []
        for d in range(depth + 1):
            layers.append(tf.keras.layers.Dense(units=channel_counts[d]))
            if use_batch_normalization: layers.append(tf.keras.layers.BatchNormalization(channel_counts[d]), axis=-1)
            if d < depth or not use_tanh: layers.append(tf.keras.layers.LeakyReLU())
        
        if use_tanh: layers.append(tf.keras.layers.Tanh())
        
        # Attributes
        self.sequential = tf.keras.Sequential(layers)
        """Attribute that refers to the :class:`tensorflow.keras.Sequential` model collecting all layers of self."""

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Applies the forward operation to ``x``.
        
        :param x: The data tensor that should be passed through the network.
        :type x: :class:`tensorflow.Tensor` 
        :return: y_hat (:class:`tensorflow.Tensor`) - The prediction."""
        
        # Predict
        y_hat = self.sequential(x)

        # Outputs:
        return y_hat
