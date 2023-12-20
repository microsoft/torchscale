import os
import sys
import torch
import random
import argparse
from PIL import Image, ImageFilter, ImageOps
from multiprocessing import Pool, cpu_count
from timm.data.transforms import RandomResizedCropAndInterpolation
import torchvision.transforms as transforms  

Image.MAX_IMAGE_PIXELS = 6400000000


def build_transform(input_size):
    train_interpolation = "bicubic"
    t = [
        RandomResizedCropAndInterpolation(input_size, scale=(0.5, 1.0), interpolation=train_interpolation), 
        transforms.RandomHorizontalFlip(),
    ]
    t = transforms.Compose(t)

    return t


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def save_image(transformed_img, output_image_path):
    if isinstance(transformed_img, torch.Tensor):
        transformed_img = transforms.ToPILImage()(transformed_img)
    transformed_img.save(output_image_path)


def get_image_files(input_dir):  
    for root, _, files in os.walk(input_dir):  
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  
                yield os.path.join(root, file)


def transform_and_save_crops(args):
    input_path, input_dir, output_dir, transform = args
    print(input_path)
    file_basename = os.path.basename(input_path)

    img = pil_loader(input_path)
    transformed_img = transform(img)
    output_image_path = os.path.join(output_dir, file_basename)
    save_image(transformed_img, output_image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save transformed images in a directory.')
    parser.add_argument('input_dir', help='Path to the input directory.')
    parser.add_argument('output_dir', help='Path to the output directory.')
    parser.add_argument('-p', '--processes', type=int, default=cpu_count(), help='Number of processes to use. Default: number of CPU cores')
    parser.add_argument('--input_size', type=int, default=16384, help='input image size')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    num_processes = args.processes
    input_size = args.input_size
    print("num_processes: {}".format(num_processes))
    print("input_size: {}".format(input_size))

    transform = build_transform(input_size=input_size)

    image_files = list(get_image_files(input_dir))
    task_args = [(file, input_dir, output_dir, transform) for file in image_files]

    os.makedirs(output_dir, exist_ok=True)
    
    with Pool(processes=num_processes) as pool:
        pool.map(transform_and_save_crops, task_args)
