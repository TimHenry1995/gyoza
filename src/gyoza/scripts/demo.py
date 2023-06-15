import tensorflow as tf
from gyoza.modelling import flow_layers as gmf
from gyoza.modelling import masks as gmm
from gyoza.modelling import standard_layers as msl
import os

if __name__ == "__main__":
    channel_count = 5
    batch_size = 4
    compute_coupling_parameters = tf.keras.layers.Dense(units=channel_count)
    mask = gmm.HeaviSide(axes=[1], shape=[channel_count])
    mask.build(input_shape=[batch_size,channel_count])
    #coupling_layer =  msl.BasicFullyConnectedNet(latent_channel_count=channel_count, output_channel_count=channel_count, depth=3)
    coupling_layer = gmf.AdditiveCouplingLayer(compute_coupling_parameters=compute_coupling_parameters, mask=mask)

    x = tf.reshape(tf.range(batch_size*channel_count, dtype=tf.float32), [batch_size, channel_count])
    y = coupling_layer(x)
    print(x)
    print(y)

    # Saving and Loading
    path = os.path.join(os.getcwd(), "example_model")
    coupling_layer.save(path)
    del coupling_layer
    loaded_coupling_layer = tf.keras.models.load_model(path)
    y_prime = loaded_coupling_layer(x)
    print(y_prime)
    k=3
