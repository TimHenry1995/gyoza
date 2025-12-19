import gyoza.tutorials.data_iterators as mdis
import unittest
import tensorflow as tf
from typing import List, Tuple
import os
import numpy as np
import shutil

class TestFactorizedPairIterators(unittest.TestCase):
    
    @staticmethod
    def create_temporary_files(n: int, x_shape: List[int], factor_count: int, folder_suffix: str) -> Tuple[str, List[str]]:
        """Creates ``n`` temporary .npy files containing random numpy arrays of ``shape`` as well as ``n`` such files each containing
        ``factor_count`` many random integers in range [0,2] corresponding to labels along factors.
        Overrides the folder if it currently exists.

        :param n: The number of desired files.
        :type n: int
        :param x_shape: The desired shape of the random tensors.
        :type x_shape: List[int]
        :param factor_count: The number of factors in the y files.
        :type factor_count: int
        :param folder_suffix: A suffix added to the folder name to ensure uniqueness.
        :type folder_suffix: str
        :return: 
            - folder_path (str) - A path to the folder containing the temporary files.
            - x_file_names (List[str]) - The names of the ``n`` x files created inside the folder at ``folder_path``.
            - y_file_names (List[str]) - The names of the ``n`` y_files created inside the folder at ``folder_path``."""

        # Create a folder path
        folder_path = os.path.join(os.getcwd(), "temporary_data_directory_for_data_iterator_unit_tests_" + folder_suffix)
        if os.path.exists(folder_path): shutil.rmtree(folder_path)
        os.makedirs(name=folder_path)

        # Generate and save tensors
        x_file_names = [None] * n
        y_file_names = [None] * n
        for i in range(n):
            x_file_names[i] = f"X_{i}.npy"
            y_file_names[i] = f"Y_{i}.npy"
            np.save(os.path.join(folder_path, x_file_names[i]), np.random.rand(*x_shape))
            np.save(os.path.join(folder_path, y_file_names[i]), np.random.randint(low=0, high=3, size=factor_count))

        # Outputs
        return folder_path, x_file_names, y_file_names

    def test_persistent_factorized_pair_iterator(self):
        """Tests whether persistent_factorized_pair_iterator produces a sensible sequences of pair batches."""

        # Create a temporary folder with dummy files
        folder_path, x_file_names, y_file_names = TestFactorizedPairIterators.create_temporary_files(n=10, factor_count=4, x_shape=[3,4], folder_suffix='1')
        
        # Initialize
        batch_size = 3
        iterator = mdis.persistent_factorized_pair_iterator(data_path=folder_path, x_file_names=x_file_names, y_file_names= y_file_names, similarity_function=lambda y_a, y_b: y_a == y_b, batch_size=batch_size)
        np.random.seed(42) # Ensure x_b are selected predicatbly at random

        a_indices = [[6,3,7],[2,6,7],[7,2,5],[5,1,4]] # The second axis is for batch size
        b_indices = [[4,6,9],[4,3,7],[4,1,7],[0,9,5]]
        for i in range(len(a_indices)): # i is batch index
            X_ab, Y_ab = next(iterator)
            
            for j in range(batch_size): # j is instance index within batch
                # Assert X_ab_j
                x_target_a = np.load(os.path.join(folder_path, x_file_names[a_indices[i][j]]))
                x_target_b = np.load(os.path.join(folder_path, x_file_names[b_indices[i][j]]))
                x_target = np.concatenate([x_target_a[np.newaxis,:], x_target_b[np.newaxis,:]], axis=0)
                x_observed = X_ab[j].numpy()

                self.assertTupleEqual(x_target.shape, x_observed.shape)
                self.assertAlmostEqual(np.sum((x_target-x_observed)**2), 0)

                # Assert Y_ab_j
                y_target_a = np.load(os.path.join(folder_path, y_file_names[a_indices[i][j]]))
                y_target_b = np.load(os.path.join(folder_path, y_file_names[b_indices[i][j]]))
                y_target = np.array(y_target_a == y_target_b, dtype=np.float32)
                y_observed = Y_ab[j].numpy()
                self.assertTupleEqual(y_target.shape, y_observed.shape)
                self.assertAlmostEqual(np.sum((y_target - y_observed)**2), 0)

        # Restore precondition
        shutil.rmtree(folder_path)

    
if __name__ == "__main__":
    #unittest.main()
    TestFactorizedPairIterators().test_persistent_factorized_pair_iterator()