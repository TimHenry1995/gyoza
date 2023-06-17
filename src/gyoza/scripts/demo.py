import tensorflow as tf
from gyoza.modelling import flow_layers as gmf
from gyoza.modelling import masks as gmm
from gyoza.modelling import standard_layers as msl
import os

if __name__ == "__main__":
    # TODO Swirl example with two spatial axes and 1 channel axis
    def create_model(channel_count: int = 5) -> msl.FlowLayer:

        compute_coupling_parameters = tf.keras.layers.Dense(units=channel_count)
        mask = gmm.HeaviSide(axes=[1], shape=[channel_count])
        
        network = gmf.SequentialFlowNetwork(sequence=[
            gmf.AdditiveCoupling(axes=[1], shape=[channel_count], compute_coupling_parameters=compute_coupling_parameters, mask=mask), 
            gmf.Shuffle(axes=[1], shape=[channel_count])
            ])

        return network

    channel_count = 5
    batch_size = 4

    network = create_model(channel_count=channel_count)

    x = tf.reshape(tf.range(batch_size*channel_count, dtype=tf.float32), [batch_size, channel_count])
    y = network(x)
    print(x)
    print(y)

    # Saving and Loading
    path = os.path.join(os.getcwd(), "example_model.h5")
    network.save_weights(path)
    del network
    loaded_network = create_model()
    loaded_network.build(input_shape=x.shape)
    loaded_network.load_weights(path)
    y_prime = loaded_network(x)
    print(y_prime)
    k=3
