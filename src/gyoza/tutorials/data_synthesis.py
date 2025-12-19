import numpy as np
import tensorflow as tf
import random
from typing import Callable, Tuple
import numpy as np
from typing import List
import tensorflow as tf
from typing import Tuple, Callable, Generator

# For reproducability
def reset_random_number_generators(seed: int):
    """This function resets the random number generators of python, numpy and tensorflow.

    :param seed: The new seed that shall be provided to each random number generator.
    :type seed: int
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def create_data_set(S: np.array, manifold_function: Callable, noise_standard_deviation: Tuple[float, float]) -> Tuple[np.array, np.array]:
    """Creates a data set by passing the position ``S`` along the manifold through the ``manifold_function`` and adding Gaussian noise to
    each dimension. That noise is centered at zero and has standard deviation ``noise_standard_deviation``.

    :param S: The position along the manifold. Shape == [:math:`M`], where :math:`M` is the number of instances
    :type S: np.array
    :param manifold_function: A function that maps from ``S`` to 2-dimensional coordinates on the manifold in the real two dimensional plane.
    :type manifold_function: :class:`Callable`
    :param noise_standard_deviation: Tuple with standard deviations for random normal noise added to the two respective z-dimensions.
    :type noise_standard_deviation: Tuple[float, float]
    :return:
        - Z (:class:`numpy.ndarray`) - The collection of points around the manifold. Shape == [:math:`M`, :math:`N`], where :math:`M` is the
            instance count and :math:`N` the dimensionality.
        - Y (:class:`numpy.ndarray`) - The target for factorization. Here, it is simply a matrix of shape == [:math:`M`, :math:`F`], where :math:`M` is the instance count and :math:`F=2` the factor count. The first column consists of standard normally distributed random numbers that correspond to the residual factor. The second column is the standardized S for the position factor. 
    """

    # Add noise
    data_function = lambda x: add_noise(*manifold_function(x), noise_standard_deviation=noise_standard_deviation)

    # Generate data
    Y = np.concat([np.random.standard_normal(size=[len(S),1]), (S[:, np.newaxis]-np.mean(S))/np.std(S)], axis=1)
    Z_1, Z_2 = data_function(x=S) # Map line s onto manifold in 2D plane
    Z = np.concatenate([Z_1[:,np.newaxis], Z_2[:,np.newaxis]], axis=1)

    # Outputs
    return Z, Y

add_noise = lambda x_1, x_2, noise_standard_deviation: (x_1 + np.random.normal(scale=noise_standard_deviation[0], size=x_1.shape), x_2 + np.random.normal(scale=noise_standard_deviation[1], size=x_2.shape))
    
def factorized_pair_iterator(X: np.ndarray, Y: np.ndarray, batch_size: int, target_correlations: List[float]) -> Generator[Tuple[tf.Tensor, tf.Tensor], None, None]:
    """This infinite iterator yields pairs of instances X_a and X_b along with their corresponding factorized correlation Y_ab. Pairs are 
    obtained by first drawing an x_a instance uniformly at random from X. Then, given this x_a instance, the `target_correlations` and a standard normally
    distributed helper random random variable, a hypothetical x_b instance is computed. Then, the closest (in the L-2 norm sense) X instance to this hypothetical
    x_b instance is chosen as the actual x_b instance. The two instances x_a and x_b form a pair. If the `batch_size` is too large for the given total number of 
    instances in `X`, it is possible that pairs occur more than once in a batch.
    
    :param X: Input data of shape [instance count, ...], where ... is any shape convenient for the caller.
    :type X: :class:`numpy.ndarray`
    :param Y: Scores of factors of shape [instance count, factor count], **including the residual factor at index 0**. These y-values are assumed to be normally distributed and **uncorrelated**.
    :type Y: :class:`numpy.ndarray`
    :param batch_size: Desired number of instances per batch
    :type batch_size: int
    :target_correlations: The desired correlations that x_a and x_b shall have along the different factors. Also the residual factor with its correlation of zero shall be included. The order of entries in `target_correlations` should be aligned with the order of factors in `Y`.
    :type target_correlations: List[float]

    :yield: 
        - X_ab (:class:`numpy.ndarray`) - A batch of instance pairs of shape [batch_size`, 2, ...], where 2 is due to the 
            concatenation of X_a and X_b and ... is the same instance-wise shape as for ``X``. 
        - Y_ab (:class:`numpy.ndarray`) - The `target_similarities` prepended with a zero and casted to shape [``batch_size``, factor count], including the residual factor.
    """

    # Input validity
    assert len(X.shape) > 0 and Y.shape[0] > 0 and X.shape[0] == Y.shape[0], f"The inputs X and Y were expected to have the same number of instances along the initial axis, yet X has shape {X.shape} and Y has shape {Y.shape}."
    assert len(Y.shape) == 2, f"The shape of Y should be [instance count, factor count], but found {Y.shape}."
    assert len(Y.shape) >= 2, f"The input Y needs to have at least two instances."
    
    # Convenience variables
    factor_count = len(target_correlations) # Includes residual factor
    instance_count = Y.shape[0]
    muh_a, sigma_a = np.mean(Y, axis=0), np.std(Y, axis=0) # Each shape == [factor count], excluding the residual factor
    muh_b, sigma_b = muh_a, sigma_a
    Y_ab = tf.repeat(np.array(target_correlations)[np.newaxis,:], repeats=batch_size, axis=0) # Concatenate along instance axis
    target_correlations = np.array(target_correlations)

    #sigma_b_given_a = np.sqrt(sigma_b * (1- target_correlations**2)) # Shape == [factor_count]

    #y_sorted_indices = [np.argsort(a=Y[f]) for f in range(factor_count)] # Shape [factor count][instance count]

    # Prepare a neighbor look-up table. Rows are instances of y, columns are neighbors at increasing distances. The largest distance corresponds to +- 3 standard deviations away from the instance's y value
    # Shape == [instance count, factor count, 2], where 2 is for the lower and upper confidence interval bounds located at +- 3 standard deviations away from the mean
    #confidence_intervals = (muh_b + target_correlations * sigma_b/ sigma_a * (Y - muh_a)) + sigma_b * np.sqrt(1-target_correlations**2) * np.array([-3.0, + 3.0] * factor_count)
                           
    #for i in range (instance_count):

    #    # Find neighbor located in confidence interval
    #    standardized_Y = (Y - Y[i])/sigma_b_given_a # Shape == [factor count]
    #    neighbor_indices = np.where(np.pow(standardized_Y - Y[i], 2) < 3**2)


    # Loop over batches
    while True:
        
        # Choose random X_a instances
        a_indices = np.random.randint(low=0, high=instance_count, size=batch_size)
        X_a = tf.cast(X[a_indices,:], tf.keras.backend.floatx())[:, tf.newaxis, :]
        Y_a = Y[a_indices]

        # Compute hypothetical y_b instances
        hypothetical_Y_b = (muh_b + target_correlations * sigma_b/ sigma_a * (Y_a - muh_a)) + sigma_b * np.sqrt(1-target_correlations**2) * np.random.multivariate_normal(mean=np.zeros(shape=[factor_count]), cov=np.eye(N=factor_count), size=batch_size)
        
        # For each hypothetical y_b in the batch, find closest match from real data
        X_b = np.empty_like(X_a)
        for i in range(len(hypothetical_Y_b)): # Iterate batch
            b_index = np.argmin(np.sqrt(np.sum(np.pow(Y-hypothetical_Y_b[i], 2), axis=1)))
            try:
                X_b[i] = X[b_index]
            except Exception as error:
                print(i, b_index, error)
        X_ab = tf.concat([X_a, X_b], axis=1) # Concatenate along pair axis
        
        yield X_ab, Y_ab
