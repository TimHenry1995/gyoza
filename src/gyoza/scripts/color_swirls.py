import tensorflow as tf
from gyoza.modelling import flow_layers as mfl
from gyoza.modelling import masks as gmm
from gyoza.modelling import standard_layers as msl
from gyoza.utilities import math as gum
import os
import matplotlib.pyplot as plt
import numpy as np

    
# Generate some data

my_dpi = 192 # https://www.infobyip.com/detectmonitordpi.php
image=gum.make_color_wheel(pixels_per_inch=my_dpi, pixel_count=512, swirl_strength=3, gaussian_variance=1)
plt.imshow(image)
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.tight_layout()
plt.axis('off')

path = os.path.join(os.getcwd(), "image.png")
plt.savefig(path, bbox_inches='tight', dpi=my_dpi)
plt.close()
k=9
# Further transformation
#shuffle = Shuffle(channel_count=2)
#basic_fully_connected = BasicFullyConnectedNet(latent_channel_count=2, output_channel_count=2, depth=2)

#tmp = shuffle(x=x)
#y_hat = basic_fully_connected(x=tmp)

# Visualization


def create_model(channel_count: int = 5) -> msl.FlowLayer:

    compute_coupling_parameters = tf.keras.layers.Dense(units=channel_count)
    mask = gmm.HeaviSide(axes=[1], shape=[channel_count])
    
    network = mfl.SequentialFlowNetwork(sequence=[
        mfl.AdditiveCoupling(axes=[1], shape=[channel_count], compute_coupling_parameters=compute_coupling_parameters, mask=mask), 
        mfl.Shuffle(axes=[1], shape=[channel_count])
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
