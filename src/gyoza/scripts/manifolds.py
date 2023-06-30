"""This case study demonstrates how factor modelling with flow neural networks can be used to represent the position along one-
dimensional manifolds. In particular, several such manifolds placed in the real plane are constructed and for each of them a
flow neural network is calibrated to disentagle the position along the manifold from its morphology. The network then allows to
move along the manifold by simply adjusting its corresponding factor.
"""

import tensorflow as tf
from gyoza.modelling import flow_layers as mfl
from gyoza.modelling import masks as gmm
from gyoza.modelling import standard_layers as msl
from gyoza.utilities import math as gum
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from typing import List
from gyoza.modelling import data_iterators as mdis
import shutil
import random

# configuration
tf.keras.backend.set_floatx('float32') # If reconstruction precision is too low, increase the precision here
initial_xs = np.arange(0,1+1/999,1/999, dtype=tf.keras.backend.floatx()) # These x points shall be mapped onto manifolds
is_plotting = True 

# 1. Generate manifolds
# 1.1 Create mathematical functions
f1 = lambda x: (x, 0*x)
f2 = lambda x: gum.rotate(xs=x, ys=0*x, theta=np.pi/4)
f3 = lambda x: (x, x**6)
f4 = lambda x: (x, 0.1*(3*x-1.5)**7)
f5 = lambda x: (x, np.sin(6*np.pi*x))
f6 = lambda x: gum.archimedian_spiral(xs=x*5*np.pi, alpha=1)

if is_plotting:
    plt.figure(); plt.suptitle("One-Dimensional Manifolds")
    plt.subplot(2,3,1)
    plt.scatter(*f1(initial_xs)); plt.title('f1'); plt.ylim(-1,1)

    plt.subplot(2,3,2)
    plt.scatter(*f2(initial_xs)); plt.title('f2')

    plt.subplot(2,3,3)
    plt.scatter(*f3(initial_xs)); plt.title('f3')

    plt.subplot(2,3,4)
    plt.scatter(*f4(initial_xs)); plt.title('f4')

    plt.subplot(2,3,5)
    plt.scatter(*f5(initial_xs)); plt.title('f5')

    plt.subplot(2,3,6)
    plt.scatter(*f6(initial_xs)); plt.title('f6')
    plt.tight_layout()
    plt.show()

# 1.2 Select a manifold
f_raw = f3

# 1.3 Generate points along the manifold
noise_function = lambda x, y: (x+ 0.05 * np.random.standard_normal(size=y.shape), y + 0.05 * np.random.standard_normal(size=y.shape))
f = lambda x: noise_function(*f_raw(x))
Y = initial_xs # Here the 'labels' are the position along the unit line
xs, ys = f(initial_xs) # Map onto manifold in 2D plane
X = np.concatenate([xs[:,np.newaxis], ys[:,np.newaxis]], axis=1)

# 1.4 Create an instance pair geneator
# The flow model needs pairs of instances and their similarity rating to learn 
# underlying factors, Here, the factor is chosen to be proximity along the manifold
def pair_iterator(X,Y, batch_size):
    
    # Convenience variables
    y_range = np.max(Y) - np.min(Y)
    
    # Initialize batches
    X_a = np.empty(shape=[batch_size, X.shape[1]], dtype=tf.keras.backend.floatx()) # Instance a of pair
    X_b = np.empty(shape=[batch_size, X.shape[1]], dtype=tf.keras.backend.floatx()) # Instance b of pair
    Y_a_b   = np.empty(shape=[batch_size,], dtype=tf.keras.backend.floatx()) # Similarity of their Y values
    k = 0 # Counter for elements in batch
    
    # Loop over pairs
    for n in range(X.shape[0]):
        i = random.randint(0, X.shape[0]-1)
        j = random.randint(0, X.shape[0]-1)

        X_a[k,:] = X[i,:]
        X_b[k,:] = X[j,:]
        Y_a_b[k] = 1 - np.abs(Y[i]-Y[j])/y_range # Manifold proximity in range [0,1]
        k += 1
        if (k == (batch_size) or n == (X.shape[0]-1)):
            
            yield tf.concat([X_a[:k,:][:,np.newaxis,:], X_b[:k,:][:,np.newaxis,:]], axis=1), tf.concat([tf.zeros([k,1], dtype=tf.keras.backend.floatx()), Y_a_b[:k][:,np.newaxis]], axis=1)
            k = 0

    return

iterator = pair_iterator(X=X, Y=Y, batch_size=8)
X_a_b, Y_a_b = next(iterator)
print("A batch has shapes:")
print("X_a_b: (instances, pair, coordinates)", X_a_b.shape)
print("Y_a_b: (instances)                   ", Y_a_b.shape)

if is_plotting:
    plt.figure(); plt.title("Points And Their Manifold Proximities")
    
    plt.scatter(X_a_b[:,0,0], X_a_b[:,0,1])
    plt.scatter(X_a_b[:,1,0], X_a_b[:,1,1])

    for i in range(xs.shape[0]-1): plt.plot(*f_raw(initial_xs[i:i+2]), color='gray')

    plt.legend(['X_a','X_b','f'])
    for i in range(X_a_b.shape[0]):
        plt.plot([X_a_b[i,0,0], X_a_b[i,1,0]], [X_a_b[i,0,1], X_a_b[i,1,1]], '--', color='black')
        plt.text((X_a_b[i,0,0] +X_a_b[i,1,0])/2, (X_a_b[i,0,1]+X_a_b[i,1,1])/2, np.round(Y_a_b[i], 2))
    plt.show()

# 2. Build a model

def create_model() -> mfl.FlowLayer:
    dimensionality = 2
    compute_coupling_parameters_1 = msl.BasicFullyConnectedNet(latent_channel_count=4*dimensionality, output_channel_count=dimensionality, depth=3)
    mask_1 = gmm.SquareWave1D(axes=[1], shape=[dimensionality])
    compute_coupling_parameters_2 = msl.BasicFullyConnectedNet(latent_channel_count=4*dimensionality, output_channel_count=dimensionality, depth=3)
    mask_2 = gmm.SquareWave1D(axes=[1], shape=[dimensionality])
    compute_coupling_parameters_3 = msl.BasicFullyConnectedNet(latent_channel_count=4*dimensionality, output_channel_count=dimensionality, depth=3)
    mask_3 = gmm.SquareWave1D(axes=[1], shape=[dimensionality])
    compute_coupling_parameters_4 = msl.BasicFullyConnectedNet(latent_channel_count=4*dimensionality, output_channel_count=dimensionality, depth=3)
    mask_4 = gmm.SquareWave1D(axes=[1], shape=[dimensionality])
    
    network = mfl.SupervisedFactorNetwork(sequence=[
        mfl.AdditiveCoupling(axes=[1], shape=[dimensionality], compute_coupling_parameters=compute_coupling_parameters_1, mask=mask_1), 
        mfl.Shuffle(axes=[1], shape=[dimensionality]),
        mfl.AdditiveCoupling(axes=[1], shape=[dimensionality], compute_coupling_parameters=compute_coupling_parameters_2, mask=mask_2), 
        mfl.Shuffle(axes=[1], shape=[dimensionality]),
        mfl.AdditiveCoupling(axes=[1], shape=[dimensionality], compute_coupling_parameters=compute_coupling_parameters_3, mask=mask_3), 
        mfl.Shuffle(axes=[1], shape=[dimensionality]),
        mfl.AdditiveCoupling(axes=[1], shape=[dimensionality], compute_coupling_parameters=compute_coupling_parameters_4, mask=mask_4), 
        ],
        factor_channel_count=[1,1]) # One channel for the residual factor (index 0) and one for the manifold proximity (index 1)

    return network

network = create_model()
#network.build(input_shape=X_train.shape)

print("For the following 3 (non-paired) input instances ... ")
print(X_a_b[:3,0,:])
print("... the model provides this decomposition ")
print(network(X_a_b[:3,0,:]))
print("... and this is the reconstruction error (should be almost 0)")
print(network.invert(network(X_a_b[:3,0,:])) - X_a_b[:3,0,:])

# 4. Train the model
if is_plotting:
    tmp_xs = np.arange(np.min(initial_xs), np.max(initial_xs), (np.max(initial_xs) - np.min(initial_xs)) / 55)
    tmp_xs, tmp_ys = f(tmp_xs)
    plt.figure(); plt.suptitle("Before training")
    plt.subplot(1,2,1); plt.title("X")
    plt.scatter(tmp_xs, tmp_ys, c=gum.color_palette/255.0); plt.xlabel("First Dimension"); plt.ylabel("Second Dimension")
    plt.subplot(1,2,2); plt.title("Z")
    Z = network(tf.concat([tmp_xs[:,np.newaxis], tmp_ys[:,np.newaxis]], axis=1))
    plt.scatter(Z[:,0], Z[:,1], c=gum.color_palette/255.0); plt.xlabel('Residual Factor'); plt.ylabel('Manifold Proximity Factor')
    plt.tight_layout()
    plt.show() 

network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
epoch_count = 10
for e in range(epoch_count):
    iterator = pair_iterator(X=X, Y=Y, batch_size=128)
    for batch in iterator:
        loss = network.train_step(data=batch)
        print(loss.numpy())

if is_plotting:
    tmp_xs = np.arange(np.min(initial_xs), np.max(initial_xs), (np.max(initial_xs) - np.min(initial_xs)) / 55)
    tmp_xs, tmp_ys = f(tmp_xs)
    plt.figure(); plt.suptitle("After training")
    plt.subplot(1,2,1); plt.title("X")
    plt.scatter(tmp_xs, tmp_ys, c=gum.color_palette/255.0); plt.xlabel("First Dimension"); plt.ylabel("Second Dimension")
    plt.subplot(1,2,2); plt.title("Z")
    Z = network(tf.concat([tmp_xs[:,np.newaxis], tmp_ys[:,np.newaxis]], axis=1)); plt.xlabel('Residual Factor'); plt.ylabel('Manifold Proximity Factor')
    plt.scatter(Z[:,0], Z[:,1], c=gum.color_palette/255.0)
    plt.tight_layout()
    plt.show() 

# Saving and Loading
path = os.path.join(os.getcwd(), "example_model.h5")
network.save_weights(path)
del network
loaded_network = create_model()
loaded_network.build(input_shape=X_train.shape)
loaded_network.load_weights(path)
print(np.round(loaded_network(X_train[:3]).numpy(), 2))
