import numpy as np
from typing import List
import tensorflow as tf
import copy as cp
import os
import random as rd
from typing import Tuple

class PairIterator(tf.keras.utils.Sequence):
    """Iterates pairs x_a, x_b of data that have the a label in common. An instance x_a is chosen arbitrarily from ``x_file_names`` 
    and among the list of instances with same label as x_a, a second instance is chosen arbitrarily. This pair is placed in a batch
    along with batch_size-1 other pairs. The output will contain the pairs X [``batch_size``, 2, *``shape``] and the labels y of 
    shape [``batch_size``, factor count].
    
    :param data_path: Path to the folder that contains the inputs that shall be paired based on ``label``.
    :type data_path: str
    :param x_file_names: File names that identify input instances stored at ``data_path`` in .npy files (including file extension).
    :type x_file_names: List[str]
    :param labels: The labels that correspond to the instances listed in ``x_file_names``. For each label, there shall be factor count
        many entries whose integer value indicates the corresponding label along that factor. Shape == [instance count, factor count].   
    :type labels: :class:``Tensor``
    :param shape: The shape of one x instance, e.g. [128,128,3] for an image.
    :type shape: List[int]
    :param batch_size: The number of pairs that shall be put inside a batch.
    :type batch_size: int, optional
    """

    def __init__(self, data_path: str, x_file_names: List[str], labels: List[List[int]], shape: List[int], batch_size: int = 32):
        'Constructor for this class'
        
        # Input validity
        assert len(x_file_names) == len(labels), f"The inputs x_file_names and labels were expected to have the same length but were found to have length {len(x_file_names)}, {len(labels)}, respectively."
        
        # Attributes
        self.data_path = data_path
        self.__x_file_names__ = cp.copy(x_file_names)
        self.__labels__ = cp.deepcopy(labels)
        self.shape = cp.copy(shape)
        self.batch_size = batch_size

        # Set starting conditions
        self.on_epoch_end()

    def __len__(self) -> int:
        """Computes the number of batches per epoch"""
        return int(np.floor(len(self.__labels__) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Iterates a batch of data. For details see ``PairIterator`` description.
        
        :param index: The index of the first data point x_a in the current batch hat shall be selected from the entire data set.
        :type index: int

        :return:
            - X (:class:`tensorflow.Tensor`) - A pair of instances. The first instance is always taken from :py:attr:`self.__indices__` (starting at ``index``, in order), while the second one is drawn uniformly at random from :py:attr:`self.__indices__`. Shape == [len(indices), 2, :py:attr:`self.shape`].
            - Y (:class`tensorflow.Tensor`) - Boolean tensor of shape == [``batch_size``, factor count], indicating for each factor wether the two instances are similar or not.
        """
        
        # Generate indices of the batch
        current_indices = self.__indices__[index*self.batch_size:(index+1)*self.batch_size]

        # Iterate data
        X, Y = self.__load_batch__(indices=current_indices)

        return X, Y

    def on_epoch_end(self) -> None:
        """Updates indexes after each epoch"""
        self.__indices__ = np.arange(self.__labels__.shape[0])
        np.random.shuffle(self.__indices__)

    def __load_batch__(self, indices: List[int]) -> Tuple[tf.Tensor, tf.Tensor]:
        """Loads pairs of data points. For details see ``PairIterator`` description.
        
        :param indices: The indices of the x_a instances that shall be loaded from self.__x_file_names__.
        :type indices: List[int]
        :return:
            - X (:class:`tensorflow.Tensor`) - A pair of instances. The first instance is always taken from ``indices`` (in order), while the second one is drawn uniformly at random from :py:attr:`self.__indices__`. Shape == [len(indices), 2, :py:attr:`self.shape`].
            - Y (:class`tensorflow.Tensor`) - Boolean tensor of shape == [``batch_size``, factor count], indicating for each factor wether the two instances are similar or not.
        ."""

        # Initialization
        X = np.empty((self.batch_size, 2, *self.shape))
        Y = np.empty((self.batch_size, self.__labels__.shape[1]), dtype=int)

        # Load data
        for i, x_a_index in enumerate(indices):
            x_a = np.load(os.path.join(self.data_path, self.__x_file_names__[x_a_index]))
            x_b_index = rd.choice(self.__indices__)
            x_b = np.load(os.path.join(self.data_path, x_b_path))
            X[i,:] = np.concatenate([x_a[np.newaxis,:], x_b[np.newaxis,:]], axis=0)
            Y[i,:] = tf.cast(self.__labels__[x_a_index,:] == self.__labels__[x_b_index], tf.float32)

        # Outputs
        return tf.constant(X, dtype=tf.float32), tf.constant(Y, dtype=tf.float32)
