import unittest
import os
import tempfile
import shutil
import cv2
import numpy as np
import pandas as pd
from getData import load_and_process_training_images, load_ground_truth, load_testing_images


class TestGetData(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Create dummy train, test, and ground truth data
        self.train_directory = os.path.join(self.temp_dir, 'train')
        os.makedirs(self.train_directory)

        # Create dummy image files in the train directory
        for label in range(43):
            label_directory = os.path.join(self.train_directory, f'{label:05d}')
            os.makedirs(label_directory)
            for i in range(6):
                dummy_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                image_path = os.path.join(label_directory, f'{i:05d}_00000.ppm')
                cv2.imwrite(image_path, dummy_image)

        self.test_directory = os.path.join(self.temp_dir, 'test')
        os.makedirs(self.test_directory)

        # Create dummy image files in the test directory
        for i in range(5):
            dummy_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            image_path =  os.path.join(self.test_directory, f'{i:05d}_00000.ppm')
            cv2.imwrite(image_path, dummy_image)

        self.ground_truth_path = os.path.join(self.temp_dir, 'ground_truth.csv')
        with open(self.ground_truth_path, 'w') as f:
            f.write("label\n0\n1\n2\n")

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_load_and_process_training_images(self):
        images, labels = load_and_process_training_images(self.train_directory)
        self.assertIsInstance(images, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(images), len(labels))
        self.assertGreater(len(images), 0)

    def test_load_ground_truth(self):
        ground_truth_df = load_ground_truth(self.ground_truth_path)
        self.assertIsInstance(ground_truth_df, pd.DataFrame)
        self.assertGreater(len(ground_truth_df), 0)

    def test_load_testing_images(self):
        test_images = load_testing_images(self.test_directory)
        self.assertIsInstance(test_images, np.ndarray)
        self.assertGreater(len(test_images), 0)

if __name__ == '__main__':
    unittest.main()
