import numpy as np
from typing import List
import tensorflow as tf
import copy as cp
import os
import random as rd
from typing import Tuple

class PairIterator(tf.keras.utils.Sequence):
    """This class provides functionality to iterates instances x_a of the data and finds for each of them an arbitrarily selected x_b 
    such that they form a pair. The X outputs will be batches of such pairs while the Y outputs will be the corresponding batches of
    factor-wise label equality. That is, the Y_i for pair X_i will be a vector of length factor count indicating label equality with 1
    and inequality with zero. X thus has shape [``batch_size``, 2, *``x_shape``] and Y has shape [``batch_size``, factor count].
    
    :param data_path: Path to the folder that contains the inputs that shall be paired based on ``label``.
    :type data_path: str
    :param x_file_names: File names that identify input instances stored at ``data_path`` in .npy files (including file extension).
    :type x_file_names: List[str]
    :param y_file_names: Files names to label vectors that correspond to the instances listed in ``x_file_names``. The vector shall
        have factor count many entries whose integer value indicates the corresponding label along that factor. 
    :type y_file_names: List[str]
    :param x_shape: The shape of one x instance, e.g. [128,128,3] for an image.
    :type x_shape: List[int]
    :param batch_size: The number of pairs that shall be put inside a batch.
    :type batch_size: int, optional
    """

    def __init__(self, data_path: str, x_file_names: List[str], y_file_names: List[str], x_shape: List[int], batch_size: int = 32):
        'Constructor for this class'
        
        # Input validity
        assert len(x_file_names) == len(y_file_names), f"The inputs x_file_names and y_file_names were expected to have the same length but were found to have length {len(x_file_names)}, {len(y_file_names)}, respectively."
        
        # Attributes
        self.__data_path__ = data_path
        self.__x_file_names__ = cp.copy(x_file_names)
        self.__y_file_names__ = cp.copy(y_file_names)
        self.__x_shape__ = cp.copy(x_shape)
        self.batch_size = batch_size

        # Set starting conditions
        self.on_epoch_end()

    def __len__(self) -> int:
        """Computes the number of batches per epoch"""
        return int(np.floor(len(self.__x_file_names__) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Iterates a batch of data. For details see ``PairIterator`` description.
        
        :param index: The index of the first data point x_a in the current batch hat shall be selected from the entire data set.
        :type index: int

        :return:
            - X (:class:`tensorflow.Tensor`) - A pairs of instances x_a, x_b. The x_a instance is always taken from the batch of instances in range [``index``, ``index``+:py:attr:`self.batch_size`]` (in order), while x_b is drawn uniformly at random from :py:attr:`self.__indices__`. Shape == [len(indices), 2, :py:attr:`self.__shape__`].
            - Y (:class`tensorflow.Tensor`) - A tensor of ones and zeros with shape == [:py:attr:`batch_size`, factor count], indicating for each factor wether the two instances have the same label or not.
        """
        
        # Generate indices of the batch
        current_indices = self.__indices__[index*self.batch_size:(index+1)*self.batch_size]

        # Iterate data
        X, Y = self.__load_batch__(indices=current_indices)

        return X, Y

    def on_epoch_end(self) -> None:
        """Updates indexes after each epoch"""
        self.__indices__ = np.arange(len(self.__x_file_names__))
        np.random.shuffle(self.__indices__)

    def __load_batch__(self, indices: List[int]) -> Tuple[tf.Tensor, tf.Tensor]:
        """Loads pairs of data points. For details see ``PairIterator`` description.
        
        :param indices: The indices of the x_a instances that shall be loaded from self.__x_file_names__.
        :type indices: List[int]
        :return:
            - X (:class:`tensorflow.Tensor`) - A pairs of instances x_a, x_b. The x_a instance is always taken from ``indices`` (in order), while x_b is drawn uniformly at random from :py:attr:`self.__indices__`. Shape == [len(indices), 2, :py:attr:`self.__shape__`].
            - Y (:class`tensorflow.Tensor`) - A tensor of ones and zeros with shape == [:py:attr:`batch_size`, factor count], indicating for each factor wether the two instances have the same label or not.
        """

        # Initialization
        X = np.empty((self.batch_size, 2, *self.__x_shape__))
        factor_count = np.load(os.path.join(self.__data_path__, self.__y_file_names__[0])).shape[0]
        Y = np.empty((self.batch_size, factor_count), dtype=int)

        # Load data
        for i, x_a_index in enumerate(indices):
            # X_i
            x_a = np.load(os.path.join(self.__data_path__, self.__x_file_names__[x_a_index]))
            x_b_index = rd.choice(self.__indices__)
            x_b = np.load(os.path.join(self.__data_path__, self.__x_file_names__[x_b_index]))
            X[i,:] = np.concatenate([x_a[np.newaxis,:], x_b[np.newaxis,:]], axis=0)
            
            # Y_i
            y_a = np.load(os.path.join(self.__data_path__, self.__y_file_names__[x_a_index]))
            y_b = np.load(os.path.join(self.__data_path__, self.__y_file_names__[x_b_index]))
            Y[i,:] = tf.cast(y_a == y_b, tf.float32)

        # Outputs
        return tf.constant(X, dtype=tf.float32), tf.constant(Y, dtype=tf.float32)
