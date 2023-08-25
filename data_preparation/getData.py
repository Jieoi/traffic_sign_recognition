import glob
import os
import cv2
import numpy as np
import time
import pandas as pd


def load_and_process_training_images(train_directory):
    # Use the 'glob' module to find all image files in the subdirectories
    image_paths = glob.glob(os.path.join(train_directory, '**', '*.ppm'), recursive=True)

    images = []
    labels = []

    start_time = time.time()  # Record the start time

    for image_path in image_paths:
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        # Resize the image to 32x32 pixels
        image = cv2.resize(image, (32, 32))
        # Normalize the pixel values to the range [0, 1]
        image = image.astype('float32') / 255.0
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

        # Read the test image using OpenCV
        image = cv2.imread(image_path)

        # Check if the image is empty
        if image is None:
            print(f"Skipping file: {image_path}")
            continue

        # Resize the image to 32x32 pixels
        image = cv2.resize(image, (32, 32))

        # Normalize the pixel values to the range [0, 1]
        image = image.astype('float32') / 255.0

        # Append the image to the list of test images
        test_images.append(image)

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the overall time consumption

    # Convert the list of test images to a NumPy array
    test_images = np.array(test_images)

    print("Loaded", len(test_images), "test images in", elapsed_time, "seconds.")

    return test_images