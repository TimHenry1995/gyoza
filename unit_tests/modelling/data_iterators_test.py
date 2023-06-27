import gyoza.modelling.data_iterators as mdis
import unittest
import tensorflow as tf
from typing import List, Tuple
import os
import numpy as np
import shutil

class TestPairIterator(unittest.TestCase):
    
    @staticmethod
    def create_temporary_files(n: int, shape: List[int]) -> Tuple[str, List[str]]:
        """Creates a list of ``n`` temporary .npy files containing random numpy arrays of ``shape``.
        Overrides the folder if it currently exists.

        :param n: The number of desired files.
        :type n: int
        :param shape: The desired shape of the random tensors.
        :type shape: List[int]
        :return: 
            - folder_path (str) - A path to the folder containing the temporary files.
            - file_names (List[str]) - The names of the ``n`` files created inside the folder at ``folder_path``."""

        # Create a folder path
        folder_path = os.path.join(os.getcwd(), "temporary_data_directory_for_data_iterator_unit_tests")
        if os.path.exists(folder_path): shutil.rmtree(folder_path)
        os.makedirs(name=folder_path)

        # Generate and save tensors
        file_names = [None] * n
        for i in range(n):
            file_names[i] = f"{i}.npy"
            np.save(os.path.join(folder_path, file_names[i]), np.random.rand(*shape))

        # Outputs
        return folder_path, file_names

    def test_init(self):
        """Tests whether PairIterator can be initialized."""

        # Create a temporary folder with dummy files
        folder_path, file_names = TestPairIterator.create_temporary_files(n=10, shape=[3,4])
        
        # Initialize
        iterator = mdis.PairIterator(data_path= folder_path, x_file_names=file_names, labels= [0,1,2,0,2,1,2,0,2,1], shape=[3,4], batch_size=4)
        
        # Target
        x_target = {0: ['0.npy','3.npy','7.npy'], 1: ['1.npy','5.npy','9.npy'],2:['2.npy','4.npy','6.npy','8.npy']}

        # Observe
        x_observed = iterator.__label_to_x_file_names__

        # Evaluate
        self.assertDictEqual(x_observed, x_target)

        # Restore precondition
        shutil.rmtree(folder_path)

    def test_next(self):
        """Tests whether PairIterator produces a sensible sequences of pair batches."""

        # Create a temporary folder with dummy files
        folder_path, file_names = TestPairIterator.create_temporary_files(n=10, shape=[3,4])
        
        # Initialize
        labels = [0,1,2,0,2,1,2,0,2,1]
        batch_size = 2
        iterator = mdis.PairIterator(data_path= folder_path, x_file_names=file_names, labels= labels, shape=[3,4], batch_size=batch_size)
        iterator.__indices__ = np.arange(len(labels)) # Ensure x_a and y are not shuffled
        i = 0
        for batch in iterator:
            X, y = batch
            
            # Assert x_a (x_b is difficult to test due to its being randomly selected)
            for j in range(batch_size):
                x_target = np.load(os.path.join(folder_path, file_names[i]))
                x_observed = X[j,0,:].numpy()

                self.assertTupleEqual(x_target.shape, x_observed.shape)
                self.assertAlmostEqual(np.sum((x_target-x_observed)**2), 0)
                i += 1

            # Assert y
            self.assertTupleEqual(y.numpy().shape, (batch_size,))
            self.assertAlmostEqual(np.sum((y.numpy() - np.array(labels[i-batch_size:i]))**2), 0)    

        # Restore precondition
        shutil.rmtree(folder_path)

    
if __name__ == "__main__":
    #unittest.main()
    TestPairIterator().test_next()