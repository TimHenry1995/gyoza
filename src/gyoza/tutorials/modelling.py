import numpy as np
from typing import List
import random

def cross_validate(Z: np.ndarray, Y: np.ndarray, target_correlations: List[float], networks: List[mfl.SupervisedFactorModel], batch_size: int, epoch_count: int, manifold_name: str, plot_losses: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Performs cross-validation on the provided networks. It first shuffles the data ``Z`` and ``Y``, then partitions it into fold-count many
    equally sized subsets and then calibrates each network on the data set-minus one subset of the partition. The held out subsets are returned
    for model evaluationl. The fold-count is inferred from the length of ``network``. This implementation assumes that the networks are
    :class:`SupervisedFactorNetworks` that are calibrated using the :py:func:`volatile_factorized_pair_iterator`.

    :param Z: The input to the networks (when in inference mode) of shape [:math:`M`,:math:`N`], where :math:`M` is the total instance count and
        :math:`N` the dimensionality.
    :type Z: :class:`numpy.ndarray`
    :param Y: The factorized targets per instance of shape [:math:`M`. :math:`F`], where :math:`M` is the total instance count and :math:`F` is
        the factor count.
    :type Y: :class:`numpy.ndarray`
    :target_correlations: The desired correlations that x_a and x_b shall have along the different factors. Here, only the actual factors should be included in the list, while the residual factor will be assigned a correlation of zero. The order of entries in `target_correlations` should be aligned with the order of factors in `Y`.
    :type target_correlations: List[float]
    :param networks: The networks to be calibrated. They are assumed to be compiled with an optimizer.
    :type networks: List[:class:`SupervisedFactorNetwork`]
    :param batch_size: The size of batches used during calibration.
    :type batch_size: int
    :param epoch_count: The number of times the iterator shall cycle through the data during calibration of each network.
    :type epoch_count: int
    :param minimum_similarity: A float typically non-negative, that indicates the minimum similarity that instance in a pair need to have in order to be output by the iterator. The lower the value, the more variety there will be among pairs.
    :type minimum_similarity: float, optional, defaults to 0.0
    :param manifold_name: The name of the manifold on which the networks are fitted. Used for the title of the plot.
    :type manifold_name: str
    :param minimum_similarity: The minimum similarity that instances should have to be listed as a pair by the ``volatile_factorized_pair_iterator``.
    :param plot_losses: Indicates whether the loss per epoch shall be plotted for each network.
    :type plot_losses: bool, default False

    :return:
        - Z_test (List[:class:`numpy.ndarray`]) - The test subsets for the model input, one for each instance in ``networks``, having
            shape [:math:`M^*`,:math:`N`], where :math:`M^*` is the test set size equal to the total instance count in ``Z`` divided by the
            number of instances in ``networks`` and :math:`N` is the dimensionality.
        - Y_test (List[:class:`numpy.ndarray`]) - The test subsets for the models factor targets, one for each instance in ``networks``, having
            shape [:math:`M^*`,:math:`F`], where :math:`M^*` is the test set size equal to the total instance count in ``Z`` divided by the
            number of instances in ``networks`` and :math:`F` is the number of factors.
    """

    # Shuffle indices
    M = Z.shape[0]
    indices = list(range(M)); random.shuffle(indices)

    # Prepare figure
    fold_count = len(networks)

    # Cross validate
    Z_test = [None] * fold_count; Y_test = [None] * fold_count
    fold_size = M // fold_count
    iterators = [None] * fold_count
    for k in range(fold_count):

        # Calibrate
        train_indices = indices[:k*fold_size] + indices[(k+1)*fold_size:]
        M = len(train_indices)
        iterator = mdis.volatile_factorized_pair_iterator(X=Z[train_indices,:], Y=Y[train_indices,:], batch_size=batch_size, target_correlations=target_correlations)
        epoch_loss_means, epoch_loss_standard_deviations = networks[k].fit(iterator=iterator, epoch_count=epoch_count, batch_count=M//batch_size)

        if plot_losses: plot_loss_trajectory(epoch_loss_means=epoch_loss_means, epoch_loss_standard_deviations=epoch_loss_standard_deviations, manifold_name=manifold_name)

        # Save test indices
        test_indices = indices[k*fold_size:(k+1)*fold_size]
        Z_test[k] = Z[test_indices,:]; Y_test[k] = Y[test_indices,:]

    # Outputs
    return Z_test, Y_test