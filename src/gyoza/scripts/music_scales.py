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

is_plotting = False    
# 1. Generate diatonic scale data set
# 1.1 Inspect musical notes and their frequencies
if is_plotting:
    plt.figure(); plt.title("Frequency Distribuiton of Musical Notes")
    plt.bar(x=gum.note_to_frequency.keys(), height=gum.note_to_frequency.values())
    plt.xticks(list(gum.note_to_frequency.keys())[::12]); plt.ylabel('Frequency'); plt.xlabel("Note"); plt.show()

# 1.2 Generate scales

major_scales = OrderedDict()
minor_scales = OrderedDict()
frequencies = list(gum.note_to_frequency.values())
for n, note in enumerate(list(gum.note_to_frequency.keys())[:-12]):
    major_scales[note] = [frequencies[n+i] for i in gum.major_steps]
    minor_scales[note] = [frequencies[n+i] for i in gum.minor_steps]

# 1.3 Split into train and test sets
scale_count = len(major_scales)
X_major = np.array(list(major_scales.values()))
X_minor = np.array(list(minor_scales.values()))
X_train = np.concatenate([X_major[:scale_count//2,:], X_minor[scale_count//2:,:]], axis=0)
X_test =  np.concatenate([X_minor[:scale_count//2,:], X_major[scale_count//2:,:]], axis=0)
Y_train = np.array([0] * (scale_count//2) + [1] * (scale_count - scale_count//2))
Y_test =  np.array([1] * (scale_count//2) + [0] * (scale_count - scale_count//2))
print('X_train shape ', X_train.shape)
print('Y_train shape', Y_train.shape)
print('X_test shape ', X_test.shape)
print('Y_test shape', Y_test.shape)

# Need to make pairs such that the elements x_a and x_b in the pair share one concept while all other concepts
# are uniformly at random similar or dissimilar. Here we have pairs for scale being for instance both major
# And we we have pairs for root being for instance both note F5

# 1.4 Plot the two set halves
def plot_scales(X: np.ndarray, Y: np.ndarray, root_names: List[str], title_suffix: str) -> None:
    """Plots diatonic scales as scatter plots.

    :param X: The frequencies of scales. Shape == [n, 8] where n is the number of scales and 8 the number of notes per scale.
        Their order is assumed to be synchronous with ``root_name``.
    :type X: np.ndarray
    :param Y: Labels for the n scales in ``X``, whereby 0 indicates major and 1 indicates minor scale. It is assumed to have 
        synchronized indexing with ``X``. These labels are used to color the scatter.
    :type Y: np.ndarray
    :param root_names: A list of the root note names that shall be enumerated along the horizontal axis, e.g. ['C0','C#0','D0'].
        These are assumed to be sorted according to musical convention.
    :type root_names: List[str]
    :param title_suffix: A suffix that shall be added to the figure title.
    :type title_suffix: str
    """

    # Preparations
    root_indices = np.array(list(range(len(root_names))))
    x_0_root_indices = np.reshape(np.repeat(root_indices[np.where(Y==0)][:,np.newaxis], axis=1, repeats=X.shape[1]), [-1])
    x_1_root_indices = np.reshape(np.repeat(root_indices[np.where(Y==1)][:,np.newaxis], axis=1, repeats=X.shape[1]), [-1])
    
    # Plotting
    plt.scatter(x_0_root_indices, X[np.where(Y==0),:], marker='.')
    plt.scatter(x_1_root_indices, X[np.where(Y==1),:], marker='.')
    plt.xticks(root_indices[::12], root_names[::12])
    plt.ylabel('Frequency'); plt.xlabel("Root Note"); plt.legend(['Major','Minor'])
   
if is_plotting:
    plt.figure(); plt.suptitle("Diatonic Scale Dataset")
    plt.subplot(1,2,1); plt.title("Train")
    plot_scales(X=X_train, Y=Y_train, root_names=list(gum.note_to_frequency.keys()), title_suffix='Train')
    plt.subplot(1,2,2); plt.title("Test")
    plot_scales(X=X_test, Y=Y_test, root_names=list(gum.note_to_frequency.keys()), title_suffix='Test')
    plt.yticks([]); plt.ylabel('')
    plt.show()

# 2. Build data iterator

# 2.1 Save the data to disk
data_folder = os.path.join(os.getcwd(), "Music Scales")
train_data_folder = os.path.join(data_folder, "Train")
test_data_folder = os.path.join(data_folder, "Test")

if not os.path.exists(train_data_folder): os.makedirs(train_data_folder)
if not os.path.exists(test_data_folder): os.makedirs(test_data_folder)

for i, instance in enumerate(X_train): np.save(os.path.join(train_data_folder, f"{i}.npy"))
for i, instance in enumerate(X_test): np.save(os.path.join(test_data_folder, f"{i}.npy"))

# 2.2 Construct iterator
mdis.PairIterator(data_path=train_data_folder, x_file_names=os.listdir(train_data_folder), labels=)

# 3. Build a model
def create_model(channel_count: int) -> mfl.FlowLayer:

    compute_coupling_parameters = msl.BasicFullyConnectedNet(latent_channel_count=channel_count, output_channel_count=channel_count, depth=3) #tf.keras.layers.Dense(units=channel_count)
    mask = gmm.SquareWave1D(axes=[1], shape=[channel_count])
    
    network = mfl.SupervisedFactorNetwork(sequence=[
        mfl.AdditiveCoupling(axes=[1], shape=[channel_count], compute_coupling_parameters=compute_coupling_parameters, mask=mask), 
        mfl.Shuffle(axes=[1], shape=[channel_count])
        ])

    return network

channel_count = X_train.shape[1] # The number of notes in a scale
batch_size = 4

network = create_model(channel_count=channel_count)
network.compile()
#y_hat = network(X_train[:batch_size,:])
network.fit_generator(generator=pair_iterator)
mdis.PairIterator(data_path=)

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
