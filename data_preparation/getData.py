import glob
import os

import numpy as np
import time

from PIL import Image

import pandas as pd


def load_and_process_training_images(train_directory):
    # Use the 'glob' module to find all image files in the subdirectories
    image_paths = glob.glob(os.path.join(train_directory, '**', '*.ppm'), recursive=True)

    images = []
    labels = []

    start_time = time.time()  # Record the start time

    for image_path in image_paths:
        # Open and load the image using PIL
        with Image.open(image_path) as image:
            # Resize the image to 32x32 pixels
            image = image.resize((32, 32))
            # Convert the image to a NumPy array and normalize pixel values to [0, 1]
            image = np.array(image) / 255.0
            images.append(image)

        # Extract the label from the image path
        label = int(image_path.split(os.sep)[-2])
        labels.append(label)

    # Convert the lists to NumPy arrays for easier manipulation
    images = np.array(images)
    labels = np.array(labels)

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the overall time consumption

    print("Loaded", len(images), "images and", len(labels), "labels in", elapsed_time, "seconds.")

    return images, labels
    
def load_ground_truth(ground_truth_path, delimiter=";"):
    ground_truth_df = pd.read_csv(ground_truth_path, delimiter=delimiter)
    
    return ground_truth_df
    
def load_testing_images(test_directory):
    # Get the list of test image file names
    test_files = sorted(os.listdir(test_directory))

    test_images = []

    start_time = time.time()  # Record the start time

    for test_file in test_files:
        # Construct the full path to the test image
        image_path = os.path.join(test_directory, test_file)

        # Take into account of other files
        try:
            # Open the image using PIL
            image = Image.open(image_path)

            # Resize the image to 32x32 pixels
            image = image.resize((32, 32))

            # Convert the PIL image to a NumPy array
            image_array = np.array(image)

            # Normalize the pixel values to the range [0, 1]
            image_array = image_array.astype('float32') / 255.0

            # Append the image to the list of test images
            test_images.append(image_array)
        except Exception as e:
            print(f"Skipping file {image_path}: {str(e)}")
            continue

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the overall time consumption

    # Convert the list of test images to a NumPy array
    test_images = np.array(test_images)

    print("Loaded", len(test_images), "test images in", elapsed_time, "seconds.")

    return test_images





