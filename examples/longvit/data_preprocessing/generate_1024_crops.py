import os
import sys
import cv2
import json
import numpy as np
import openslide
import time
import torch
import openslide
import argparse
import random
import shutil

import glob  
from concurrent.futures import ProcessPoolExecutor

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


def is_similar_pixel(pixel1, pixel2, threshold=30):
    return np.linalg.norm(pixel1 - pixel2) < threshold


def should_discard_image(image_path, target_pixel=np.array([243, 243, 243]), threshold=30, similarity_ratio=0.99):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    similar_pixels = 0
    total_pixels = height * width

    for y in range(height):
        for x in range(width):
            pixel = image[y, x]

            if is_similar_pixel(pixel, target_pixel, threshold):
                similar_pixels += 1

    ratio = similar_pixels / total_pixels
    return ratio > similarity_ratio


def random_crop(slide_path, output_path, min_crop_size, max_crop_size, level=0):
    slide = openslide.open_slide(slide_path)
    level_dim = slide.level_dimensions
    slide_width, slide_height = slide.dimensions

    crop_width = random.randint(min_crop_size, max_crop_size)
    crop_height = random.randint(min_crop_size, max_crop_size)
    
    x = random.randint(0, slide_width - crop_width)
    y = random.randint(0, slide_height - crop_height)

    region = slide.read_region((x, y), level, (crop_width, crop_height))
    region = region.convert("RGB")
    region.save(output_path)


def get_crops(slide_path, output_folder, crop_number, min_crop_size, max_crop_size):
    print(slide_path)

    index = 0
    while index < crop_number:
        output_path = os.path.join(output_folder, os.path.basename(slide_path).split(".svs")[0], f"{str(index).zfill(8)}.JPEG")
        
        dir_path = os.path.dirname(output_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        random_crop(slide_path, output_path, min_crop_size, max_crop_size)
        if not should_discard_image(output_path):
            index += 1


def process_slides(input_folder, output_folder, crop_number=100, min_crop_size=1024, max_crop_size=1536):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    slide_paths = glob.glob(f"{input_folder}/**/*.svs", recursive=True)

    with ProcessPoolExecutor(max_workers=4) as executor:
        for slide_path in slide_paths:
            executor.submit(get_crops, slide_path, output_folder, crop_number, min_crop_size, max_crop_size)


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Generate crops from slides")  
    parser.add_argument("input_folder", type=str, help="")
    parser.add_argument("output_folder", type=str, help="")
    parser.add_argument("crop_number", type=int, help="")    

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    crop_number = args.crop_number
  
    process_slides(input_folder, output_folder, crop_number=crop_number)
