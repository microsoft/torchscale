import os
import glob
import argparse
import openslide

from PIL import Image
from concurrent.futures import ProcessPoolExecutor


def convert_wsi_to_images(slide_path, image_path, target_size, level=0):
    slide = openslide.open_slide(slide_path)
    level_dims = slide.level_dimensions
    region = slide.read_region((0,0), level, level_dims[level])
    region = region.convert("RGB")
    print("convert: {}({}) -> {}".format(slide_path, region.size, image_path))  
    resized_img = region.resize((target_size, target_size), Image.BICUBIC)
    resized_img.save(image_path)


def process_slides(input_folder, output_folder, target_size, level=0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    slide_paths = glob.glob(os.path.join(input_folder, "*.svs"))

    with ProcessPoolExecutor(max_workers=1) as executor:
        for slide_path in slide_paths:
            image_path = os.path.join(output_folder, os.path.basename(slide_path).split(".svs")[0] + ".jpg")  
            executor.submit(convert_wsi_to_images, slide_path, image_path, target_size, level=level)


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Convert slides into images")  
    parser.add_argument("input_folder", type=str, help="")
    parser.add_argument("output_folder", type=str, help="")
    parser.add_argument("target_size", type=int, help="")
    parser.add_argument("level", type=int, help="")
 
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    target_size = args.target_size
    level = args.level

    process_slides(input_folder, output_folder, target_size, level=level)
