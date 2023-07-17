import numpy as np
from typing import List
import tensorflow as tf
import copy as cp
import os
import random as rd
from typing import Tuple, Callable

class PersistentFactorizedPairIterator(tf.keras.utils.Sequence):
    """This class provides functionality to iterates instances x_a of the data and finds for each of them an arbitrarily selected x_b 
    such that they form a pair. The X outputs will be batches of such pairs while the Y outputs will be the corresponding batches of
    factor-wise label equality. That is, the Y_i for pair X_i will be a vector of length factor count indicating label equality with 1
    and inequality with zero. X thus has shape [``batch_size``, 2, * ``x_shape`` ] and Y has shape [``batch_size``, factor count].
    
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
            - X (:class:`tensorflow.Tensor`) - A pairs of instances x_a, x_b. The x_a instance is always taken from the batch of instances in range [``index``, ``index``+:py:attr:`self.batch_size`] (in order), while x_b is drawn uniformly at random from :py:attr:`self.__indices__`. Shape == [len(indices), 2, :py:attr:`self.__shape__`].
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
            Y[i,:] = tf.cast(y_a == y_b, tf.keras.backend.floatx())

        # Outputs
        return tf.constant(X, dtype=tf.keras.backend.floatx()), tf.constant(Y, dtype=tf.keras.backend.floatx())


def volatile_factorized_pair_iterator(X: np.ndarray, Y: np.ndarray, similarity_function: Callable, batch_size: int):
    """This iterator yields pairs of instances X_a and X_b along with their corresponding factorized similarity Y. Pairs are obtained 
    by shuffling X once for X_a and once for X_b. It is thus possible that pair i has the same instance in X_a and X_b, yet unlikely 
    for large instance counts in X. The iterator produces a instance count many pairs, split into batches. The last batch may be 
    smaller than ``batch_size`` to reach that pair count.

    :param X: Input data of shape [instance count, ...], where ... is any shape convenient for the caller.
    :type X: :class:`numpy.ndarray` or :class:`tensorflow.Tensor`
    :param Y: Scores of factors of shape [instance count, factor count].
    :type Y: :class:`numpy.ndarray` or :class:`tensorflow.Tensor`
    :param similarity_function: A callable that takes as input Y_a (:class:`numpy.nparray` or :class:`tensorflow.Tensor`), Y_b 
        (:class:`numpy.nparray` or :class:`tensorflow.Tensor`) which are each Y representation of shape [``batch_size``, 
        factor_count]. It then calculates the similarity for each factor and outputs a :class:`tensorflow.Tensor` of shape
        [``batch_size``, factor count].
    :type similarity_function: :class:`Callable`
    :param batch_size: Desired number of instances per batch
    :type batch_size: int

    :return: 
        - X_a_b (:class:`tensorflow.Tensor`) - A batch of instance pairs of shape [<=`batch_size`,
            2, ...], where 2 is due to the concatenation of X_a and X_b and ... is the same instance-wise shape as for ``X``. 
        - Y_a_b (:class:`tensorflow.Tensor`) - The corresponding batch of similarities as obtained by feeding the instances of X_a 
            and X_b into ``similarity_function``. It has shape [``batch_size``, factor count].
        
    """

    # Input validity
    assert len(X.shape) > 0 and Y.shape[0] > 0 and X.shape[0] == Y.shape[0], f"The inputs X and Y were expected to have the same number of instances along the initial axis, yet X has shape {X.shape} and Y has shape {Y.shape}."
    assert len(Y.shape) == 2, f"The shape of Y should be [instance count, factor count], but found {Y.shape}."
    assert Y.shape[0] >= 2 and Y.shape[1] == similarity_function(Y[0:1,:], Y[1:2,:]).shape[1], f"The similarity function is expected to provide factor count many outputs but this could not be verified when feeding the first two instances in."

    # Convenience variables
    instance_count = Y.shape[0]
    
    # Loop over batches
    for batch_size in [batch_size] * (instance_count // batch_size) + [instance_count - batch_size * (instance_count // batch_size)]:
        a = np.random.randint(low=0, high=instance_count, size=batch_size)
        b = np.random.randint(low=0, high=instance_count, size=batch_size)

        X_a = tf.cast(X[a,:], tf.keras.backend.floatx())[:, tf.newaxis, :]
        X_b = tf.cast(X[b,:], tf.keras.backend.floatx())[:, tf.newaxis, :]
        X_a_b = tf.concat([X_a, X_b], axis=1)
        Y_a_b = tf.cast(similarity_function(Y[a,:], Y[b,:]), tf.keras.backend.floatx())

        yield X_a_b, Y_a_b

    return