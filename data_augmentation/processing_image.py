import numpy as np
import cv2

import torch
import os
import glob
import os.path as osp

from models import RRDBNet_arch as arch

def upscale_images(images, upscale_factor):
    # Initialize an empty array to store the upscaled images
    upscaled_images = np.empty((len(images), images.shape[1] * upscale_factor, images.shape[2] * upscale_factor, images.shape[3]), dtype=np.float32)

    # Apply bicubic interpolation to each image using CPU
    for i in range(images.shape[0]):
        for c in range(images.shape[3]):  # Channels (R, G, B)
            upscaled_images[i, :, :, c] = cv2.resize(images[i, :, :, c],
                                                     (images.shape[1] * upscale_factor, images.shape[2] * upscale_factor),
                                                     interpolation=cv2.INTER_CUBIC)

    # Convert upscaled_images to the correct format (CV_8UC3)
    upscaled_images_uint8 = (upscaled_images * 255.0).clip(0, 255).astype(np.uint8)
    
    return upscaled_images_uint8


def denoise_images(images):
    # Initialize an empty array to store the denoised images
    denoised_images = np.zeros_like(images)

    # Denoise each image using fastNlMeansDenoisingColored on CPU
    for i in range(images.shape[0]):
        denoised_images[i] = cv2.fastNlMeansDenoisingColored(images[i], None, 10, 10, 7, 21)

    return denoised_images

def equalize_images(images):
    # Initialize arrays to store grayscale and equalized images
    gray_images = np.zeros((images.shape[0], images.shape[1], images.shape[2]), dtype=np.uint8)
    equalized_images = np.zeros_like(images)

    # Convert to grayscale and apply histogram equalization for each image
    for i in range(images.shape[0]):
        gray_images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        for c in range(3):  # 3 channels (B, G, R)
            equalized_images[i, :, :, c] = cv2.equalizeHist(images[i, :, :, c])

    return equalized_images
    
    
# adapt from https://github.com/xinntao/ESRGAN/blob/master/test.py
def enhance_images_rrdb(img_folder, output_dir, model_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    idx = 0
    for path in glob.glob(img_folder):
        if path.lower().endswith('.csv'):
            continue  # Skip CSV files

        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)

        # Read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"Skipping {path} as it could not be loaded.")
            continue

        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()

        # Save the enhanced image to the specified directory
        output_path = osp.join(output_dir, '{:s}_enhanced.png'.format(base))
        cv2.imwrite(output_path, output)