import os
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from skimage.transform import resize
import random


def process_image(image_file, input_dir, output_dir, preprocessing_split):
    results = []
    img = Image.open(input_dir + image_file)
    img = img.resize((256, 256))
    margin_left = np.random.randint(256 - 224)
    margin_top = np.random.randint(256 - 224)

    for i in range(preprocessing_split):
        rotation = random.randint(-10, 10)
        new_img = img.rotate(rotation)
        if bool(random.getrandbits(1)):
            new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
        new_img = new_img.crop(
            (margin_left, margin_top, margin_left + 224, margin_top + 224)
        )
        results.append(new_img)

    for i in range(preprocessing_split):
        results[i].save(os.path.join(output_dir, f"{image_file.split('.')[0]}_{i}.png"))


if __name__ == "__main__":
    print("Start preprocessing? [Y|N]")
    if input().lower() == "y":
        parser = ArgumentParser()
        parser.add_argument("--input_dir", default="data/sample/images/", type=str)
        parser.add_argument(
            "--output_dir", default="data/preprocessing/images/", type=str
        )
        parser.add_argument("--preprocessing_split", default=4, type=int)
        parser.add_argument("--num", default=-1, type=int)
        args = parser.parse_args()
        os.makedirs(args.output_dir, exist_ok=True)
        img_list = os.listdir(args.input_dir)
        for idx, image_file in enumerate(img_list):
            process_image(
                image_file, args.input_dir, args.output_dir, args.preprocessing_split
            )
            if idx == args.num:
                break
    else:
        print("Preprocessing cancelled")
