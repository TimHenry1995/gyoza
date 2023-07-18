"""This case study demonstrates how factor modelling with flow neural networks can be used to represent one-dimensional manifolds. 
In particular, several such manifolds placed in the real plane are constructed and for each of them a flow neural network is 
calibrated to disentangle the position along the manifold from its morphology. The network then allows to move along the manifold 
by simply adjusting the value of the position factor. This illustrates the non-linearity of flow neural networks.
"""
if True:
    import tensorflow as tf
    from gyoza.modelling import flow_layers as mfl
    from gyoza.modelling import masks as gmm
    from gyoza.modelling import standard_layers as msl
    from gyoza.utilities import math as gum
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import OrderedDict
    from typing import List, Callable
    from gyoza.modelling import data_iterators as mdis
    import shutil
    import random
    plt.rcParams["font.family"] = "Times New Roman"

# Configuration
tf.keras.backend.set_floatx('float64') # If reconstruction precision is too low, increase the precision here
is_plotting = True 
np.random.seed(14)
tf.random.set_seed(21)
random.seed(380)

# 1. Generate manifolds
# 1.1 Generate equally spaced points along a line of length S
golden_ratio = (1 + 5 ** 0.5) / 2
beta = np.log(golden_ratio)/(np.pi/2) # https://en.wikipedia.org/wiki/Golden_spiral see section on mathematics
S = (1*np.sqrt(1+beta**2)/beta)*(np.exp(beta*2*np.pi)-1) # https://www.quora.com/What-is-the-arc-length-of-the-logarithmic-spiral-from-t0-to-t the arc length of the golden spiral for one full rotation
s = np.linspace(-S/2,S/2,10, dtype=tf.keras.backend.floatx()) # These points are the stride along the manifolds

# 1.2 Create mathematical functions that map stride s along an arc to x and y coordinates
f1 = lambda s: (np.arcsinh(s), np.sign(np.arcsinh(s)) * np.cosh(np.arcsinh(s)) - np.sign(s))
f2 = lambda s: gum.rotate(xs=s, ys=0*s, theta=np.arccos((f1(S/2)[1]/ f1(S/2)[0]) / np.sqrt(f1(S/2)[0]**2 + f1(S/2)[1]**2))) # this theta leads to same slope as the cosh function on s

f3 = lambda s: (s, S*np.sign(s)/4)
def f4(s):
    half_1 = s[np.where(s <  0)]
    half_2 = s[np.where(0 <= s)]

    x = np.concatenate([half_1, half_2 - S/4]) + 1*S/8
    y = np.array([1*S/4]* len(half_1) + [3*S/4] * len(half_2)) - S/2

    return x, y

f5 = lambda s: ((S/(2*np.pi))* np.sin((s+S/2)*(2*np.pi)/S), (S/(2*np.pi))*np.cos((s+S/2)*(2*np.pi)/S))
f6 = lambda s: gum.logarithmic_spiral(xs=np.log((s+S/2)/(1*np.sqrt(1+beta**2)/beta)+1)/beta, alpha=1, beta=beta) # https://www.quora.com/What-is-the-arc-length-of-the-logarithmic-spiral-from-t0-to-t

#
if False:
    plt.figure(); plt.suptitle(r"One-Dimensional Manifolds in $\mathbb{R}^2$")
    plt.subplot(2,3,1)
    plt.scatter(*f2(s)); plt.title(r'$f_{1a}$'); plt.ylim(-S/1.8,S/1.8); plt.xlim(-S/1.8,S/1.8); plt.ylabel(r"$x_2$")

    plt.subplot(2,3,4)
    plt.scatter(*f1(s)); plt.title(r'$f_{1b}$'); plt.ylim(-S/1.8,S/1.8); plt.xlim(-S/1.8,S/1.8); plt.ylabel(r"$x_2$"); plt.xlabel(r"$x_1$")

    plt.subplot(2,3,2)
    plt.scatter(*f3(s)); plt.title(r'$f_{2a}$'); plt.ylim(-S/1.8,S/1.8); plt.xlim(-S/1.8,S/1.8)

    plt.subplot(2,3,5)
    plt.scatter(*f4(s)); plt.title(r'$f_{2b}$'); plt.ylim(-S/1.8,S/1.8); plt.xlim(-S/1.8,S/1.8); plt.xlabel(r"$x_1$")

    plt.subplot(2,3,3)
    plt.scatter(*f5(s)); plt.title(r'$f_{3a}$'); plt.ylim(-S/1.8,S/1.8); plt.xlim(-S/1.8,S/1.8)

    plt.subplot(2,3,6)
    plt.scatter(*f6(s)); plt.title(r'$f_{3b}$'); plt.ylim(-S/1.8,S/1.8); plt.xlim(-S/1.8,S/1.8); plt.xlabel(r"$x_1$")
    plt.tight_layout()
    plt.show()

# 1.3 Select a manifold
f_raw = f1
noise_strength = 0.05

# 1.4 Generate points along the manifold
noise_function = lambda x, y: (x+ noise_strength * np.random.standard_normal(size=y.shape), y + noise_strength * np.random.standard_normal(size=y.shape))
f = lambda x: noise_function(*f_raw(x))

Y = np.concatenate([np.zeros([len(s),1]), s[:, np.newaxis]], axis=1) # Here the 'labels' are the position along the unit line
xs, ys = f(s) # Map onto manifold in 2D plane
X = np.concatenate([xs[:,np.newaxis], ys[:,np.newaxis]], axis=1)

# 1.5 Create an instance pair geneator
# The flow model needs pairs of instances and their similarity rating to learn 
# underlying factors, Here, the factor is chosen to be proximity along the manifold
similarity_function = lambda y_a, y_b: np.concatenate([np.zeros([y_a.shape[0],1]), 1 - np.abs(y_a[:,1:2]-y_b[:,1:2])/S], axis=1)
batch_size = 3
iterator = mdis.volatile_factorized_pair_iterator(X=X, Y=Y, batch_size=batch_size, similarity_function=similarity_function)
X_a_b, Y_a_b = next(iterator)
print("A batch has shapes:")
print("X_a_b: (instances, pair, coordinates)", X_a_b.shape)
print("Y_a_b: (instances)                   ", Y_a_b.shape)

if True:
    plt.figure(); plt.title("Pairs of instances And Their Factorized Similarities")
    
    plt.scatter(X_a_b[:,0,0], X_a_b[:,0,1]) # Instances a
    plt.scatter(X_a_b[:,1,0], X_a_b[:,1,1]) # Instances b

    for i in range(xs.shape[0]-1): plt.plot(*f_raw(s[i:i+2]), color='gray')

    plt.legend(['X_a','X_b','f'])
    for i in range(X_a_b.shape[0]):
        plt.plot([X_a_b[i,0,0], X_a_b[i,1,0]], [X_a_b[i,0,1], X_a_b[i,1,1]], '--', color='black')
        plt.text((X_a_b[i,0,0] +X_a_b[i,1,0])/2, (X_a_b[i,0,1]+X_a_b[i,1,1])/2, np.round(Y_a_b[i], 2))
    plt.show()

# 2. Build a model

def create_model() -> mfl.FlowLayer:
    dimensionality = 2
    compute_coupling_parameters_1 = msl.BasicFullyConnectedNet(latent_dimension_count=4*dimensionality, output_dimension_count=dimensionality, depth=3)
    mask_1 = gmm.SquareWave(axes=[1], shape=[dimensionality])
    compute_coupling_parameters_2 = msl.BasicFullyConnectedNet(latent_dimension_count=4*dimensionality, output_dimension_count=dimensionality, depth=3)
    mask_2 = gmm.SquareWave(axes=[1], shape=[dimensionality])
    compute_coupling_parameters_3 = msl.BasicFullyConnectedNet(latent_dimension_count=4*dimensionality, output_dimension_count=dimensionality, depth=3)
    mask_3 = gmm.SquareWave(axes=[1], shape=[dimensionality])
    compute_coupling_parameters_4 = msl.BasicFullyConnectedNet(latent_dimension_count=4*dimensionality, output_dimension_count=dimensionality, depth=3)
    mask_4 = gmm.SquareWave(axes=[1], shape=[dimensionality])
    
    network = mfl.SupervisedFactorNetwork(sequence=[
        #mfl.AdditiveCoupling(axes=[1], shape=[dimensionality], compute_coupling_parameters=compute_coupling_parameters_1, mask=mask_1), 
        #mfl.Shuffle(axes=[1], shape=[dimensionality]),
        #mfl.AdditiveCoupling(axes=[1], shape=[dimensionality], compute_coupling_parameters=compute_coupling_parameters_2, mask=mask_2), 
        #mfl.Shuffle(axes=[1], shape=[dimensionality]),
        #mfl.AdditiveCoupling(axes=[1], shape=[dimensionality], compute_coupling_parameters=compute_coupling_parameters_3, mask=mask_3), 
        #mfl.Shuffle(axes=[1], shape=[dimensionality]),
        #mfl.AdditiveCoupling(axes=[1], shape=[dimensionality], compute_coupling_parameters=compute_coupling_parameters_4, mask=mask_4), 
        #mfl.Shuffle(axes=[1], shape=[dimensionality]),
        #mfl.Reflection(axes=[1], shape=[dimensionality], reflection_count=2),
        mfl.ActivationNormalization(axes=[1], shape=[dimensionality])
        ],
        factor_dimension_count=[1,1]) # One dimension for the residual factor (index 0) and one for the manifold proximity (index 1)

    return network

network = create_model()

network(X_a_b[:,0,:]) # Pass through one representative sample
print("For the following 3 X_a input instances ... ")
print(X_a_b[:3,0,:]) # 0 selects X_a
print("... the model provides this decomposition ")
print(network(X_a_b[:3,0,:])) # 0 selects X_a
print("... and this is the reconstruction error (should be almost 0)")
print(network.invert(network(X_a_b[:3,0,:])) - X_a_b[:3,0,:])  # 0 selects X_a

def plot_input_output(network, x_range, f, title):
    # Sample from manifold to illustrate distortion of data
    xs = np.linspace(x_range[0], x_range[1], len(gum.color_palette)) # Each point will receive its own color
    manifold_xs, manifold_ys = f(xs)
    
    # Create gridlines to illustrate distortion of surrounding space
    points_per_line = 10
    min_x = np.min(manifold_xs); max_x = np.max(manifold_xs); mean_x = np.abs(np.mean(manifold_xs))
    xs = np.linspace(min_x - np.abs(mean_x-min_x), max_x + np.abs(mean_x-max_x), points_per_line)
    min_y = np.min(manifold_ys); max_y = np.max(manifold_ys); mean_y = np.abs(np.mean(manifold_ys))
    ys = np.linspace(min_y - np.abs(mean_y-min_y), max_y + np.abs(mean_y-max_y), points_per_line)
    h_xs, h_ys = np.meshgrid(xs, ys) # horizontal line coordinates
    v_ys, v_xs = np.meshgrid(ys, xs) # vertical line coordinates

    # Plot
    fig, axs = plt.subplots(2,4,figsize=(9,4.5), gridspec_kw={'height_ratios': [4, 0.5], 'width_ratios':[0.5,4,0.5,4]})
    # X
    # Plot joint distributions
    plt.suptitle(title)
    plt.subplot(2,4,2); plt.title("X")
    # Gridlines
    for l in range(points_per_line): plt.plot(h_xs[l,:], h_ys[l,:], color='#C5C9C7', linewidth=0.75)
    for l in range(points_per_line): plt.plot(v_xs[l,:], v_ys[l,:], color='#C5C9C7', linewidth=0.75)
    # Data
    plt.scatter(manifold_xs, manifold_ys, c=gum.color_palette/255.0, zorder=3); plt.xlabel("First Dimension"); plt.ylabel("Second Dimension")
    X_x_lim = plt.xlim(); X_y_lim = plt.ylim() # Use these for marginal distributions

    # Z
    plt.subplot(2,4,4); plt.title("Z")
    Z = network(tf.concat([manifold_xs[:,np.newaxis], manifold_ys[:,np.newaxis]], axis=1))
    H = network(tf.concat([np.reshape(h_xs, [-1])[:,np.newaxis], np.reshape(h_ys, [-1])[:,np.newaxis]], axis=1))
    V = network(tf.concat([np.reshape(v_xs, [-1])[:,np.newaxis], np.reshape(v_ys, [-1])[:,np.newaxis]], axis=1))
    
    # Gridlines
    for l in range(points_per_line): plt.plot(H[l*points_per_line:(l+1)*points_per_line,0], H[l*points_per_line:(l+1)*points_per_line,1], color='#C5C9C7', linewidth=0.75)
    for l in range(points_per_line): plt.plot(V[l*points_per_line:(l+1)*points_per_line,0], V[l*points_per_line:(l+1)*points_per_line,1], color='#C5C9C7', linewidth=0.75)
    # Data
    plt.scatter(Z[:,0], Z[:,1], c=gum.color_palette/255.0, zorder=3); plt.xlabel('Residual Factor'); plt.ylabel('Manifold Proximity Factor')
    Z_x_lim = plt.xlim(); Z_y_lim = plt.ylim()
    
    # Plot marginal distributions
    # X
    plt.subplot(2,4,6)
    plt.hist(manifold_xs, histtype='step'); plt.gca().invert_yaxis(); plt.xlim(X_x_lim); plt.axis('off')
    plt.subplot(2,4,1)
    plt.hist(manifold_ys, orientation='horizontal', histtype='step'); plt.ylim(X_y_lim); plt.gca().invert_xaxis(); plt.axis('off')
    
    # Z
    plt.subplot(2,4,8)
    plt.hist(Z[:,0], histtype='step'); plt.gca().invert_yaxis(); plt.xlim(Z_x_lim); plt.axis('off')
    plt.subplot(2,4,3)
    plt.hist(Z[:,1], orientation='horizontal', histtype='step'); plt.ylim(Z_y_lim); plt.gca().invert_xaxis(); plt.axis('off')
    
    # Make other subplots invisible
    plt.subplot(2,4,5); plt.axis('off')
    plt.subplot(2,4,7); plt.axis('off')

    plt.tight_layout()
    plt.show() 
    
# 4. Train the model
if is_plotting:
    plot_input_output(network, x_range = [np.min(s), np.max(s)], f=f, title="X And Z Before Model Calibration")

network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05))
epoch_count = 10
for e in range(epoch_count):
    iterator = pair_iterator(X=X, Y=Y, batch_size=batch_size)
    for batch in iterator:
        loss = network.train_step(data=batch)
        print(loss.numpy())
        #print(network.sequence[0].variables[0])

if is_plotting:
    plot_input_output(network, x_range = [np.min(s), np.max(s)], f=f, title="X And Z ")

# Evaluate
# take factor 1 and measure its correlation with initial_x

# Saving and Loading
path = os.path.join(os.getcwd(), "example_model.h5")
network.save_weights(path)
del network
loaded_network = create_model()
loaded_network.build(input_shape=X_train.shape)
loaded_network.load_weights(path)
print(np.round(loaded_network(X_train[:3]).numpy(), 2))
