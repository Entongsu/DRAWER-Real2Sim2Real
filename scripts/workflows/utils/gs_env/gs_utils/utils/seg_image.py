import os

import numpy as np
import cv2


def extract_image_basedon_mask(rgb_dir, mask_dir, seg_dir, folder_paths=None):

    os.makedirs(seg_dir, exist_ok=True)
    rgb_images = [
        img for img in os.listdir(rgb_dir)
        if img.endswith((".png", ".jpg", ".jpeg"))
    ]

    # Sort images by filename
    rgb_images.sort()

    mask_images = [
        img for img in os.listdir(mask_dir)
        if img.endswith((".png", ".JPG", ".jpeg"))
    ]

    # Sort images by filename
    mask_images.sort()
    image_count = 0

    for index, im in enumerate(rgb_images):

        rgb_image = cv2.imread(rgb_dir + "/" + im)

        mask_image = cv2.imread(mask_dir + f"/{index:05d}.png")

        extract_image = np.zeros_like(rgb_image)

        extract_image[np.where(mask_image > 0)] = rgb_image[np.where(
            mask_image > 0)]
        cv2.imwrite(f"{seg_dir}/frame_{image_count+1:05d}.jpg", extract_image)
        image_count += 1


extract_image_basedon_mask(
    rgb_dir="/home/ensu/Downloads/bridge_kitchen01/images",
    mask_dir=
    "/home/ensu/Documents/weird/Segment-and-Track-Anything/tracking_results/polycam_video/polycam_video_masks",
    seg_dir="/home/ensu/Downloads/bridge_kitchen01/seg_images",
    folder_paths=None)
