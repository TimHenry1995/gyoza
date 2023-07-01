import tensorflow as tf 
import numpy as np
from typing import List, Tuple

class UnsupervisedFactorLoss():
    pass

class SupervisedFactorLoss():
    """This loss can be used to incentivize the entries of the output vector of a normalizing flow network to be arranged according to
    semantic factors of the data. Take the network z^~ = T(z) which accepts data z as input (e.g. a latent representation of a 
    classical auto-encoder) and shall output and a multivariate normal distribution z^~ that arranges its entries as such factors.
    The factors can be thought of as independent components. A factor k spreading across N_k entries of the output vector is 
    incentivised by this loss to represent the similarity of two inputs z^a and z^b along one and only one concept. For instance, 
    factors can represent color, roughness, size, animal species, or material. The loss expects training instances to come in 
    pairs z^a and z^b for each such factor. A pair should have strong positive association ``sigma`` such that the corresponding 
    factor can capture the underlying concept of similarity. Yet, the association shall be close to zero for all other concepts
    (on average), i.e. all other factors. If the labelling of the data is too coarse to provide such pairs, one can use style 
    transfer models built externally to construct pairs with similar style. If it is impossible to construct pairs, one can also 
    use the :class:`UnsupervisedFactorLoss` instead.

    :param sigma: This hyperparameter refletcs the category resemblance of instances. It is chosen to be in the interval (0,1) and 
        should be clos to 1 for cohesive categories and close to 0 for categories that only loosely apply to their instances. One can 
        choose, for instance, its default value of 0,975 if training instances in the form of animal pictures are based on clearly 
        recognizable species. It should be chosen equal to e.g. 0.5 if there is a lot of fluctuation in e.g furr length, size or shape.  
    :type sigma: float, optional
    :param channels_per_factor: A list of integers that enumerates the number of channels (entries in a vector) of the factors thought to underly
        the representation of z_tilde. These shall include the residual factor at index 0 which collect all variation not captured by the 
        true factors. The sum of all entries is assumed to be equal to the number of channels in z_tilde.
    :type channels_per_factor: List[int] 

    References:
        - "A Disentangling Invertible Interpretation Network for Explaining Latent Representations" by Patrick Esser, Robin Rombach and Bjorn Ommer
    """

    def __init__(self, factor_channel_counts: List[int], sigma: float = 0.975, *args):
        
        # Super
        super(SupervisedFactorLoss, self).__init__(*args)

        # Attributes
        factor_masks = np.zeros(shape=[len(factor_channel_counts), np.sum(factor_channel_counts)])
        total = 0
        for u, channel_count in enumerate(factor_channel_counts): 
            factor_masks[u, total:total+channel_count] = 1
            total += channel_count
        self.__factor_masks__ = tf.constant(factor_masks, dtype=tf.keras.backend.floatx()) 
        """Collects masks (one per factor) that are 1 for each factor's channels and zero elsewhere. Shape == [factor count, channel count]"""

        self.__sigma__ = sigma
        """Hyperparameter in (0,1) indicating association strength between pairs of instances."""

    def compute(self,y_true: tf.Tensor, y_pred: Tuple[tf.Tensor]) -> tf.Tensor:
        """Computes the loss.
        
        :param y_true: A matrix of shape [batch size, factor count], that indicates for each pair in the batch and each factor, to 
            what extent the two instances from ``z_tilde_a`` and ``z_tilde_b`` share this factor. Similarity is assumed to be in the 
            range [0,1]. If the factors are all categorical, it makes sense to set these similarities either to 1 or 0, indicating 
            same class or not, respectviely. E.g. if there are two factors and 3 pairs of z_tilde, then ``y_true`` could be 
            [[0,1],[0,0],[0,1]], indicating that the first and last pairs of z_tilde share the concept of factor at index 1 and 
            the second pair does not have anything in common. The residual factor (located at index 0) is typically not the same for
            any two instances and thus usually stores a zero in this array. The hyperparameter sigma (typically close to 1) should be 
            set to reflect the category resemblance of instances. 
            do not resemble. In ca
        :type y_true: :class:`tensorflow.Tensor`
        :param y_pred: A tuple containing [z_tilde_a, z_tilde_b, j_a, j_b]. 
        :type y_pred: Tuple[:class:`tensorflow.Tensor`]

        - z_tilde_a (:class:`tensorflow.Tensor`) - The output of model T on the first input of the pair z^a, z^b. Shape == [batch size, channel count] where channel count is the number of channels in the flattened output of T.
        - z_tilde_b (:class:`tensorflow.Tensor`) - The output of model T on the second input of the pair z^a, z^b. Shape == [batch size, channel count] where channel count is the number of channels in the flattened output of T.
        - j_a (:class:`tensorflow.Tensor`) - The jacobian determinant on logarithmic scale of T at z^a. Shape == [batch size]
        - j_b (:class:`tensorflow.Tensor`) - The jacobian determinant on logarithmic scale of T at z^b. Shape == [batch size]

        :return: loss (tf.Tensor) - A single value indicating the amount of error the model makes in factoring its inputs.
        """

        # Input validity
        assert len(y_pred) == 4, f"The input y_pred is expected to be a tuple of the four tensorflow.Tensor objects z_tilde_a, z_tilde_b, j_a and j_b."
        z_tilde_a, z_tilde_b, j_a, j_b = y_pred
        assert len(z_tilde_a.shape) == 2, f"z_tilde_a has shape {z_tilde_a.shape} but was expected to have shape [batch size, channel count]."
        assert len(z_tilde_b.shape) == 2, f"z_tilde_b has shape {z_tilde_b.shape} but was expected to have shape [batch size, channel count]."
        assert z_tilde_a.shape == z_tilde_b.shape, f"The inputs z_tilde_a and z_tilde_b where expected to have the same shape [batch size, channel count] but found {z_tilde_a.shape} and {z_tilde_b.shape}, respectively."
        assert (z_tilde_a.shape[1] == self.__factor_masks__.shape[1]), f"z_tilde_a was expected to have as many channels along axis 1 as the sum of channels in channels_per_factor specified during initialization ({self.__factor_masks__.shape[1]}) but it has {z_tilde_a.shape[1]}."
        assert (z_tilde_b.shape[1] == self.__factor_masks__.shape[1]), f"z_tilde_b was expected to have as many channels along axis 1 as the sum of channels in channels_per_factor specified during initialization ({self.__factor_masks__.shape[1]}) but it has {z_tilde_b.shape[1]}."
    
        assert len(j_a.shape) == 1, f"The input j_a was expected to have shape [batch size] but found {j_a.shape}."
        assert len(j_b.shape) == 1, f"The input j_b was expected to have shape [batch size] but found {j_b.shape}."
        assert j_a.shape == j_b.shape, f"The inputs j_a and j_b where expected to have the same shape [batch size] but have {j_a.shape} and {j_b.shape}, respectively."
        assert j_a.shape[0] == z_tilde_a.shape[0], f"The inputs z_tilde and j are expected to have the same number of instances along the batch axis (axis 0)."
        
        assert len(y_true.shape) == 2, f"The input y_true is expected to have shape [batch size, factor count], but has shape {y_true.shape}."
        assert y_true.shape[0] == z_tilde_a.shape[0], f"The inputs y_true and z_tilde are assumed to have the same number of instances in the batch. Found {y_true.shape[0]} and {z_tilde_a.shape[0]}, respectively."
        
        # Convenience variables
        channel_count =  z_tilde_a.shape[1] 
        factor_mask = np.zeros([y_true.shape[0], channel_count], dtype=tf.keras.backend.floatx()) # Is 1 for all channels of factors shared by a pair z_a, z_b and 0 elsewhere
        for i, instance in enumerate(y_true):
            for f, similarity in enumerate(instance): # f = factor index
                factor_mask[i] = factor_mask[i] + similarity * self.__factor_masks__[f]
        
        # Implement formula (10) of referenced paper
        # L = sum_{F=1}^K expected_value_{x^a,x^b ~ p(x^a, x^b | F)} l(E(x^a), E(x^b)| F)       (term 10)
        # l(z^a, z^b | F) = sum_{k=0}^K ||T(z^a)_k||^2 - log|T'(z^a)|                           (term 7) 
        #                   + sum_{k != F} ||T(z^b)_k||^2 - log|T'(z^b)|                        (term 8) 
        #                   + ( || T(z^b)_F - sigma_{ab} T(z^a)_F || ^2) / (1-sigma_{ab}^2)     (term 9) 
        
        term_7 = tf.reduce_sum(tf.pow(z_tilde_a, 2), axis=1) - j_a # Shape == [batch size]
        term_8 = 0#tf.reduce_sum(tf.pow((1-factor_mask) * z_tilde_b, 2), axis=1) - j_b  # Shape == [batch size]
        term_9 = 0#tf.reduce_sum(factor_mask * tf.pow((z_tilde_b - z_tilde_a), 2), axis=1)   # Shape == [batch size]
        term_distance = 0#tf.reduce_sum((1-factor_mask) * 1.0 / (1+tf.pow((z_tilde_b - z_tilde_a), 2)), axis=1)
        cov = 0#tf_cov(tf.concat([z_tilde_a, z_tilde_b], axis=0))
        term_cov = 0#tf.reduce_sum(tf.pow(cov - tf.eye(channel_count), 2))
        loss = tf.reduce_mean(term_7 + term_8 + term_9 + term_distance + term_cov, axis=0)  # Shape == [1]
        
        # Outputs
        return loss

def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.keras.backend.floatx())
    cov_xx = vx - mx
    return cov_xx