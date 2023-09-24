import torch
import os
import glob
import os.path as osp

import numpy as np
import cv2

from models import RRDBNet_arch as arch

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