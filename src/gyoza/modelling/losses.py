import tensorflow as tf 
import numpy as np
from typing import List, Union
import copy as cp


class SupervisedFactorLoss(tf.keras.losses.Loss):
    r"""
    This loss can be used to incentivize the entries of the output vector of a flow network to be arranged according to semantic 
    factors of the data with multivariate normal distribution. It implements the following formula:

    .. math:: 
        \mathcal{L} = \sum_{F=1}^{K} \mathbb{E}_{(z^a,z^b) \sim p(z^a, z^b | F) } l(z^a, z^b | F)

    .. math:: 
        :nowrap:

        \begin{eqnarray}
            l(z^a,z^b | F) &= \frac{1}{2} \sum_{k=0}^{K} ||T(z^a)_k||^2 - log|T'(z^a)| \\
                &+ \frac{1}{2} \sum_{k \neq F} ||T(z^b)_k||^2 - log|T'(z^b)| \\
                &+ \frac{1}{2} \frac{||T'(z^b)_F - \sigma_{ab} T(z^a)_F||^2}{1-\sigma_{ab}^2},
        \end{eqnarray}
        
    where :math:`T(z)` is the model whose loss shall be computed, :math:`z^a`, :math:`z^b` are instances passed trough :math:`T`,
    :math:`T'(z^a)` is the Jacobian of :math:`T` and :math:`\sigma_{ab}` is the clustering strength of instances (see below). 
    The factors can be thought of as independent components. A factor :math:`k` spreading across :math:`N_k` entries of the 
    output vector is incentivised by this loss to represent the similarity of two inputs :math:`z^a` and :math:`z^b` along one and 
    only one concept. For instance, factors can represent color, roughness, size, animal species, or material. The loss expects 
    training instances to come in pairs :math:`z^a` and :math:`z^b` for each such factor. A pair should have strong positive 
    association :math:`\sigma` such that the corresponding factor can capture the underlying concept of similarity. Yet, the 
    association shall be (on average) close to zero for all other factors. See also :class:`UnsupervisedFactorLoss`.

    :param dimensions_per_factor: A list of integers that enumerates the number of dimensions (entries in a vector) of the factors thought to underly
        the representation of :math:`z^{\sim}`. These shall include the residual factor at index 0 which collect all variation not captured by the 
        true factors. The sum of all entries is assumed to be equal to the number of dimensions in :math:`z^{\sim}`.
    :type dimensions_per_factor: List[int] 

    References:

        - Esser, P., Rombach, R., & Ommer, B. (2020). 
          "A Disentangling Invertible Interpretation Network for Explaining Latent Representations." 
          `arXiv:2004.13166 <https://arxiv.org/abs/2004.13166>`_
    """
 
    def __init__(self, dimensions_per_factor: List[int], *args) -> None:
        
        # Super
        super(SupervisedFactorLoss, self).__init__(*args)

        # Attributes
        factor_count = len(dimensions_per_factor)
        factor_masks = np.zeros(shape=[factor_count, np.sum(dimensions_per_factor)])
        total = 0
        for u, dimension_count in enumerate(dimensions_per_factor): 
            factor_masks[u, total:total+dimension_count] = 1
            total += dimension_count
        self.__factor_masks__ = tf.constant(factor_masks, dtype=tf.keras.backend.floatx()) 
        """Collects masks (one per factor) that are 1 for each factor's dimensions and zero elsewhere. Shape == [factor count, dimension count]"""

        self.__dimensions_per_factor__ = cp.copy(dimensions_per_factor)
        """(:class:`List[int]`) - The number of dimensions per factor. Length equals factor count."""
    
    def call(self, y_true: Union[tf.Tensor, tf.keras.KerasTensor], y_pred: Union[tf.Tensor, tf.keras.KerasTensor]) -> Union[tf.Tensor, tf.keras.KerasTensor]:
        """Computes the loss.
        
        :param y_true: A matrix of shape [batch-size, factor-count], that indicates for each pair in the batch and each factor, to 
            what extent the two instances from ``z_tilde_a`` and ``z_tilde_b`` are correlated along this factor. This correlation 
            is assumed to be in the range [0,1]. The residual factor (located at index 0) is typically not the same for any two 
            instances and thus usually stores a zero in this array. 
        :type y_true: :class:`tensorflow.Tensor`
        :param y_pred: A concatenation along the dimension axis for the tensors [z_tilde_a, z_tilde_b, j_a, j_b]. Shape == [batch-size, 2*dimensionality of z-tilde + 2]
        :type y_pred: Union[:class:`tensorflow.Tensor`, :class:`tensorflow.keras.KerasTensor`]

        - z_tilde_a: The output of model T on the first input of the pair (z^a, z^b). Shape == [batch size, dimension count] where dimension count is the number of dimensions in the flattened output of T.
        - z_tilde_b: The output of model T on the second input of the pair (z^a, z^b). Shape == [batch size, dimension count] where dimension count is the number of dimensions in the flattened output of T.
        - j_a: The jacobian determinant on logarithmic scale of T at z^a. Shape == [batch size,1]
        - j_b: The jacobian determinant on logarithmic scale of T at z^b. Shape == [batch size,1]

        :return: loss (Union[:class:`tensorflow.Tensor`, :class:`tensorflow.keras.KerasTensor`]) - A single value indicating the amount of error the model makes in factoring its inputs.
        """

        # Unpack predictions
        d = int(np.sum(self.__dimensions_per_factor__))

        z_tilde_a = y_pred[:, :d] # Shape == [batch-size, dimensionality]
        z_tilde_b = y_pred[:, d:2*d] # Shape == [batch-size, dimensionality]
        j_a = y_pred[:, 2*d] # Shape == [batch-size]
        j_b = y_pred[:, 2*d + 1] # Shape == [batch-size]
        
        # Input validity
        if not len(z_tilde_a.shape) == 2: raise ValueError(f"z_tilde_a has shape {z_tilde_a.shape} but was expected to have shape [batch size, dimension count].")
        if not len(z_tilde_b.shape) == 2: raise ValueError(f"z_tilde_b has shape {z_tilde_b.shape} but was expected to have shape [batch size, dimension count].")
        if not z_tilde_a.shape == z_tilde_b.shape: raise ValueError(f"The inputs z_tilde_a and z_tilde_b where expected to have the same shape [batch size, dimension count] but found {z_tilde_a.shape} and {z_tilde_b.shape}, respectively.")
        if not (z_tilde_a.shape[1] == self.__factor_masks__.shape[1]): raise ValueError(f"z_tilde_a was expected to have as many dimensions along axis 1 as the sum of dimensions in dimensions_per_factor specified during initialization ({self.__factor_masks__.shape[1]}) but it has {z_tilde_a.shape[1]}.")
        if not (z_tilde_b.shape[1] == self.__factor_masks__.shape[1]): raise ValueError(f"z_tilde_b was expected to have as many dimensions along axis 1 as the sum of dimensions in dimensions_per_factor specified during initialization ({self.__factor_masks__.shape[1]}) but it has {z_tilde_b.shape[1]}.")
    
        if not len(list(j_a.shape)) == 1: raise ValueError(f"The input j_a was expected to have shape [batch size] but found {j_a.shape}.")
        if not len(list(j_b.shape)) == 1: raise ValueError(f"The input j_b was expected to have shape [batch size] but found {j_b.shape}.")
        if not j_a.shape == j_b.shape: raise ValueError(f"The inputs j_a and j_b where expected to have the same shape [batch size] but have {j_a.shape} and {j_b.shape}, respectively.")
        if not j_a.shape[0] == z_tilde_a.shape[0]: raise ValueError(f"The inputs z_tilde and j are expected to have the same number of instances along the batch axis (axis 0).")
        
        if not len(y_true.shape) == 2: raise ValueError(f"The input y_true is expected to have shape [batch size, factor count], but has shape {y_true.shape}.")
        if not y_true.shape[0] == z_tilde_a.shape[0]: raise ValueError(f"The inputs y_true and z_tilde are assumed to have the same number of instances in the batch. Found {y_true.shape[0]} and {z_tilde_a.shape[0]}, respectively.")
        
        # Expand y_true to per-dimension correlations
        # y_true: Shape == [batch-size, factor-count]
        # factor_masks: Shape == [factor-count, dimensionality]
        # result: [batch-size, dimensionality]
        #factor_count = y_true.shape[1]
        #y_true_N = tf.zeros_like(z_tilde_a)
        #for f in range(factor_count):
        #    y_true_N += self.__factor_masks__[f:f+1] * y_true[:, f:f+1]
        y_true_N = tf.keras.ops.matmul(y_true, self.__factor_masks__)

        # Clamp to avoid zero variance        
        eps = 1e-6
        y_true_N = tf.clip_by_value(y_true_N, -1 + eps, 1 - eps)
        
        # Term 1: log p(z_a)

        # Term 1: Marginal Prior for z_a
        # Must be 0.5 * z^2.
        term_1 = 0.5 * tf.reduce_sum(tf.square(z_tilde_a), axis=1)

        # Term 2: Conditional Prior for z_b | z_a
        # This term MUST include the log(var) component from the Gaussian density
        var = 1.0 - tf.square(y_true_N)
        diff = z_tilde_b - y_true_N * z_tilde_a
        # Density of N(rho*za, 1-rho^2) is: 
        # 0.5 * [ (zb - rho*za)^2 / var + log(var) ]
        term_2 = 0.5 * tf.reduce_sum(tf.square(diff) / var + tf.math.log(var), axis=1)

        # Total NLL: Subtract the model's Jacobian determinants
        # IMPORTANT: Do NOT multiply (j_a + j_b) by 0.5. 
        # The quadratic terms (0.5 * z^2) already account for the 0.5 in the density.
        loss = term_1 + term_2 - (j_a + j_b) # j_a/j_b already contain the negative log


        return loss

        
