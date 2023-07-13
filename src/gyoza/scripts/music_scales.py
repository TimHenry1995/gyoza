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



# This case study demonstrates how factor modelling with flow neural networks can be used to represent factors underlying musical


# notes. In particular, the model operates on octets of notes represented by their frequencies. It shall disentangle the diatonic 


# scale (major versus minor) of the octet and thus allow to change it, while keeping residual aspetcs of the octet intact. 



is_plotting = True    


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


X_major = np.array(list(major_scales.values())) / gum.note_to_frequency['B8']


X_minor = np.array(list(minor_scales.values())) / gum.note_to_frequency['B8']


X_train = np.concatenate([X_major[:scale_count//2,:], X_minor[scale_count//2:,:]], axis=0)


X_test =  np.concatenate([X_minor[:scale_count//2,:], X_major[scale_count//2:,:]], axis=0)


Y_train = np.zeros([scale_count, 2]) # The column at index 0 is for residual factor, the column at index 1 is for scale


Y_train[:,1] = np.array([0] * (scale_count//2) + [1] * (scale_count - scale_count//2))


Y_test = np.zeros([scale_count, 2]) # Same setup as for Y_train


Y_test[:,1] =  np.array([1] * (scale_count//2) + [0] * (scale_count - scale_count//2))


print("The generated data has the following shapes:")


print('\tX_train (instance count, note count) = ', X_train.shape)


print('\tY_train (instance count, note count) = ', Y_train.shape)


print('\tX_test  (instance count, note count) = ', X_test.shape)


print('\tY_test  (instance count, note count) = ', Y_test.shape)



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


    plot_scales(X=X_train, Y=Y_train[:,1], root_names=list(gum.note_to_frequency.keys()), title_suffix='Train')


    plt.subplot(1,2,2); plt.title("Test")


    plot_scales(X=X_test, Y=Y_test[:,1], root_names=list(gum.note_to_frequency.keys()), title_suffix='Test')


    plt.yticks([]); plt.ylabel('')


    plt.show()



# 2. Build data iterator


# Need to make pairs such that the elements x_a and x_b in the pair share one concept while all other concepts


# are uniformly at random similar or dissimilar. Here we have pairs for scale being for instance both major


# And we we have pairs for root being for instance both note F5



# 2.1 Save the data to disk


data_folder = os.path.join(os.getcwd(), "Music Scales")


train_data_folder = os.path.join(data_folder, "Train")


test_data_folder = os.path.join(data_folder, "Test")



# Ensure folder exists and are empty


if os.path.exists(train_data_folder): shutil.rmtree(train_data_folder)


os.makedirs(train_data_folder)


if os.path.exists(test_data_folder): shutil.rmtree(test_data_folder)


os.makedirs(test_data_folder)



# Save data


def save_data(data: np.ndarray, data_folder: str, name_prefix: str) -> str:


    file_names = [None] * data.shape[0]


    for i, instance in enumerate(data): 


        file_names[i] = f"{name_prefix}_{i}.npy"


        np.save(os.path.join(data_folder, file_names[i]), instance)

    return file_names



x_files_names_train = save_data(data=X_train, data_folder=train_data_folder, name_prefix="X")


x_files_names_test = save_data(data=X_test, data_folder=test_data_folder, name_prefix="X")


y_files_names_train = save_data(data=Y_train, data_folder=train_data_folder, name_prefix="Y")


y_files_names_test = save_data(data=Y_test, data_folder=test_data_folder, name_prefix="Y")



# 2.2 Construct iterator


train_pair_iterator = mdis.PersistentFactorizedPairIterator(data_path=train_data_folder,


                                  x_file_names=x_files_names_train, 


                                  y_file_names=y_files_names_train, 


                                  x_shape=X_train.shape[1:],


                                  batch_size=96)


test_pair_iterator = mdis.PersistentFactorizedPairIterator(data_path=test_data_folder,


                                  x_file_names=x_files_names_test, 


                                  y_file_names=y_files_names_test, 


                                  x_shape=X_test.shape[1:],


                                  batch_size=96)


'''


print("The iterator provides the data in batches of the folowing shapes:")


print("Train:")


for batch in train_pair_iterator:


    X_b, Y_b = batch


    print("\t X_b.shape: (instance count, pair, note count) = ", X_b.shape, "Y_b.shape: (instance count, factor count) = ", Y_b.shape)


print("Test:")


for batch in test_pair_iterator:


    X_b, Y_b = batch


    print("\t X_b.shape: (instance count, pair, note count) = ",X_b.shape, "Y_b.shape: (instance count, factor count) = ", Y_b.shape)
train_pair_iterator.re


'''


# 3. Create a model


def create_model() -> mfl.FlowLayer:


    note_count = 8


    compute_coupling_parameters_1 = msl.BasicFullyConnectedNet(latent_channel_count=note_count, output_channel_count=note_count, depth=3)


    mask_1 = gmm.SquareWaveSingleAxis(axes=[1], shape=[note_count])


    compute_coupling_parameters_2 = msl.BasicFullyConnectedNet(latent_channel_count=note_count, output_channel_count=note_count, depth=3)


    mask_2 = gmm.SquareWaveSingleAxis(axes=[1], shape=[note_count])


    compute_coupling_parameters_3 = msl.BasicFullyConnectedNet(latent_channel_count=note_count, output_channel_count=note_count, depth=3)


    mask_3 = gmm.SquareWaveSingleAxis(axes=[1], shape=[note_count])
    


    network = mfl.SupervisedFactorNetwork(sequence=[


        mfl.AdditiveCoupling(axes=[1], shape=[note_count], compute_coupling_parameters=compute_coupling_parameters_1, mask=mask_1), 


        mfl.Shuffle(axes=[1], shape=[note_count]),


        mfl.AdditiveCoupling(axes=[1], shape=[note_count], compute_coupling_parameters=compute_coupling_parameters_2, mask=mask_2), 


        mfl.Shuffle(axes=[1], shape=[note_count]),


        mfl.AdditiveCoupling(axes=[1], shape=[note_count], compute_coupling_parameters=compute_coupling_parameters_3, mask=mask_3), 


        ],


        factor_channel_count=[7,1])



    return network



network = create_model()


#network.build(input_shape=X_train.shape)



print("For the following 3 input instances ... ")


print(X_train[:3])


print("... the model provides this output ")


print(np.round(network(X_train[:3]).numpy(), 2))



# 4. Train the model


network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))


epoch_count = 100


for e in range(epoch_count):


    for batch in train_pair_iterator:


        loss = network.train_step(data=batch)


        print(loss.numpy())



# Saving and Loading


path = os.path.join(os.getcwd(), "example_model.h5")


network.save_weights(path)


del network


loaded_network = create_model()


loaded_network.build(input_shape=X_train.shape)


loaded_network.load_weights(path)


print(np.round(loaded_network(X_train[:3]).numpy(), 2))


k=3


# Harmonics

