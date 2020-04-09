import os
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from skimage.transform import resize


def process_image(image_file, input_dir, output_dir):
    img = Image.open(input_dir + image_file)
    img = img.resize((256, 256))
    results = []
    results.append(img.rotate(-5))
    results.append(img.rotate(-10))
    results.append(img.rotate(+5))
    results.append(img.rotate(+10))
    for i in range(4):
        results.append(results[i].transpose(Image.FLIP_LEFT_RIGHT))
    for i in range(8):
        margin_left = np.random.randint(256 - 224)
        margin_top = np.random.randint(256 - 224)
        results.append(
            results[i].crop(
                (margin_left, margin_top, margin_left + 224, margin_top + 224)
            )
        )
    margin_left = np.random.randint(256 - 224)
    margin_top = np.random.randint(256 - 224)
    results[i] = results[i].crop(
        (margin_left, margin_top, margin_left + 224, margin_top + 224)
    )

    for i in range(16):
        results[i].save(os.path.join(output_dir, f"{image_file.split('.')[0]}_{i}.png"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-dir", default="data/sample/images/", type=str)
    parser.add_argument("--output-dir", default="data/preprocessing/images/", type=str)
    parser.add_argument("--num", default=-1, type=int)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    img_list = os.listdir(args.input_dir)
    for idx, image_file in enumerate(img_list):
        process_image(image_file, args.input_dir, args.output_dir)
        if idx == 2:
            break
