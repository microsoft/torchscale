import os  
import json
import shutil
import argparse  
from PIL import Image  
from concurrent.futures import ProcessPoolExecutor  
  
Image.MAX_IMAGE_PIXELS = 6400000000


def split_image(image_path, input_folder, output_folder, num_splits):  
    print(image_path)
    file_name, file_ext = os.path.splitext(os.path.basename(image_path))

    img = Image.open(image_path)
    width, height = img.size

    block_width = width
    block_height = height // num_splits
    
    for i in range(num_splits):
        left = 0
        upper = i * block_height
        right = block_width
        lower = (i + 1) * block_height
        cropped_img = img.crop((left, upper, right, lower))
        cropped_img.save(f"{output_folder}/{file_name}_{i}{file_ext}")

 
def find_images(input_folder):  
    image_files = []  
    for root, _, files in os.walk(input_folder):  
        for f in files:  
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):  
                image_files.append(os.path.join(root, f))  
    return image_files  


def process_images(image_files, input_folder, output_folder, num_splits, num_processes):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:  
        for image_file in image_files:  
            executor.submit(split_image, image_file, input_folder, output_folder, num_splits)  


def main():  
    parser = argparse.ArgumentParser(description='Split images into smaller tiles')  
    parser.add_argument('--input', type=str, required=True, help='Path to the input folder containing images')  
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder for saving the tiles')  
    parser.add_argument('--num_splits', type=int, default=16, help='Size of the tiles (default: 4096)')  
    parser.add_argument('--processes', type=int, default=1, help='Number of processes (default: number of CPU cores)')  
    args = parser.parse_args()
    
    input_folder = args.input  
    output_folder = args.output  
    num_splits = args.num_splits  
    num_processes = args.processes  
  
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = find_images(input_folder)
    process_images(image_files, input_folder, output_folder, num_splits, num_processes)  


if __name__ == "__main__":
    main()

