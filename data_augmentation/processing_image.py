import numpy as np
import cv2

import multiprocessing
from functools import partial

def convert_image_format(images):
    '''
    convert images array to uint8 format
    '''
    converted_images = []
    for image in images:
        # Convert the image to the correct format (3-channel uint8)
        if image.dtype != np.uint8 or len(image.shape) != 3 or image.shape[2] != 3:
            # Scale and convert to uint8
            image = (image * 255).astype(np.uint8)
        elif image.shape[2] == 4:
            image = image[:, :, :3]  # Drop the alpha channel if present
        converted_images.append(image)
    return np.array(converted_images)
    
    
    
 
 
def upscale_images(images, upscale_factor):
    '''
    upsacle using bicubic interpolation
    
    inputs
    images: array of images to be upsacled
    upscale_factor: number used as scale for upscaling
    
    output
    array of upscaled images
    '''
    # Initialize an empty array to store the upscaled images
    upscaled_images = np.empty((len(images), images.shape[1] * upscale_factor, images.shape[2] * upscale_factor, images.shape[3]), dtype=np.uint8)

    # Apply bicubic interpolation to each image using CPU
    for i in range(images.shape[0]):
        for c in range(images.shape[3]):  # Channels (R, G, B)
            upscaled_images[i, :, :, c] = cv2.resize(images[i, :, :, c],
                                                     (images.shape[1] * upscale_factor, images.shape[2] * upscale_factor),
                                                     interpolation=cv2.INTER_CUBIC)
    
    return upscaled_images
    
    
    
    
def denoise_single_image(image):
    '''
    denoising of one image using fast non local denoising 
    '''
    print("excuted once")
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def denoise_images_parallel(images):
    '''
    denoising an array of images using non local means denoising on coloured images
    
    input
    images: array of images
    
    output
    denoised_images: denoised version of image
    '''
    pool = multiprocessing.Pool()
    denoised_images = pool.map(denoise_single_image, images)
    pool.close()
    pool.join()
    return np.array(denoised_images)

def denoise_images_bilateral(images):
    '''
    Denoise an array of images using the bilateral filter.
    
    input
    images: array of images
    
    output
    denoised_images: denoised version of images
    '''
    denoised_images = []

    for image in images:
        # Apply bilateral filter
        denoised_image = cv2.bilateralFilter(image, d=3, sigmaColor=75, sigmaSpace=75)
        denoised_images.append(denoised_image)

    return np.array(denoised_images)

   
    
def equalize_images(images):
    '''
    Equalize images by converting to LAB color space and applying histogram equalization to the luminance channel (L).
    
    input
    images: array of images
    
    output
    equalize_images: array of images, equalised
    '''
    equalized_images = np.zeros_like(images)

    # Convert to LAB color space
    for i in range(images.shape[0]):
        lab_image = cv2.cvtColor(images[i], cv2.COLOR_BGR2LAB)

        # Apply histogram equalization only to the L channel (luminance)
        lab_image[:, :, 0] = cv2.equalizeHist(lab_image[:, :, 0])

        # Convert back to BGR color space
        equalized_images[i] = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    return equalized_images
    
def equalize_images_with_AHE(images):
    '''
    Equalize images using Adaptive Histogram Equalization (AHE) on the luminance channel (L) in the LAB color space.
    
    input
    images: array of images
    
    output
    equalized_images: array of images, equalized
    '''
    equalized_images = np.zeros_like(images)

    # Create an AHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Convert to LAB color space
    for i in range(images.shape[0]):
        lab_image = cv2.cvtColor(images[i], cv2.COLOR_BGR2LAB)

        # Apply AHE to the L channel (luminance)
        lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])

        # Convert back to BGR color space
        equalized_images[i] = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    return equalized_images
    
    
    
def gaussian_smoothing(images, kernel_size=(5, 5)):
    '''
    smoothing an array of images using gausuan smoothing, default kernel size is 5 by 5
    '''
    smoothed_images = [cv2.GaussianBlur(image, kernel_size, 0) for image in images]
    return np.array(smoothed_images)
    
def laplacian_edge_detection(images):
    '''
    edge detection using laplace smoothing
    '''
    edge_detected_images = []
    for image in images:
        edge_channels = [cv2.Laplacian(channel, cv2.CV_64F) for channel in cv2.split(image)]
        edge_image = cv2.merge(edge_channels)
        edge_detected_images.append(edge_image)
    return np.array(edge_detected_images)
    
    
def apply_augmentations(image):
    '''
    Apply rotation, zooming and contrast adjustment randomly to one image
    
    '''
    augmented_image = image.copy()

    # Rotation
    angle = np.random.uniform(-45, 45)  # Rotate by up to 45 degrees
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    augmented_image = cv2.warpAffine(augmented_image, rotation_matrix, (width, height))

    # Zooming
    scale = np.random.uniform(0.8, 1.2)  # Zoom in/out by up to 20%
    zoom_matrix = np.array([[scale, 0, 0], [0, scale, 0]])
    augmented_image = cv2.warpAffine(augmented_image, zoom_matrix, (width, height))

    # Contrast adjustment
    alpha = np.random.uniform(0.95, 1.05)  # Adjust contrast by up to 5%
    beta = np.random.uniform(-1, 1)   # Adjust brightness by up to 1 units
    augmented_image = cv2.convertScaleAbs(augmented_image, alpha=alpha, beta=beta)

    return augmented_image