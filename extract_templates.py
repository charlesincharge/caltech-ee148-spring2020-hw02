import argparse
import json
import numpy as np
import random
import pathlib
from PIL import Image, ImageDraw
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate templates randomly from training data.'
    )
    parser.add_argument(
        '-d',
        '--data-folder',
        help='folder of images with red lights',
        default='data/RedLights2011_Medium',
        type=pathlib.Path,
    )
    parser.add_argument(
        '-o',
        '--output-folder',
        help='folder to output filter templates to',
        default='templates/',
        type=pathlib.Path,
    )
    parser.add_argument(
        '-r',
        '--random-seed',
        help='random number seed, to ensure we always get the same train/test split',
        default=2021,
        type=int,
    )
    parser.add_argument(
        '-a',
        '--annotations-path',
        help='JSON file with annotataions',
        default='results/hw02_splits/annotations_train.json',
        type=pathlib.Path,
    )
    parser.add_argument(
        '-n',
        '--num-templates',
        help='Number of filter templates to extract',
        default=10,
        type=int,
    )

    return parser.parse_args()


def extract_template(image_path: pathlib.Path, bbox: List) -> Image:
    """Extract bounding box selection from image path.

    bbox: (top_row, left_col, bottom_row, right_col)
    """
    assert len(bbox) == 4

    # Open image with PIL
    image = Image.open(image_path)

    # image.crop takes format (left, upper, right, lower)
    template = image.crop(swap_bbox_format(bbox))

    return template


def swap_bbox_format(bbox_tuple):
    """Swap between (row0, col0, row1, col1) and (x0, y0, x1, y1) formats."""
    assert len(bbox_tuple) == 4
    return (bbox_tuple[1], bbox_tuple[0], bbox_tuple[3], bbox_tuple[2])


def main():
    args = parse_args()
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Load bounding boxes
    with args.annotations_path.open() as f:
        bounding_boxes_preds = json.load(f)

    # Create output folder, based on input JSON
    output_folder = args.output_folder.joinpath(args.annotations_path.stem)
    output_folder.mkdir(exist_ok=True)

    # Generate list of image, bounding-box pairs
    image_bbox_pair_list = []
    for image_name, bbox_list in bounding_boxes_preds.items():
        for bbox in bbox_list:
            image_bbox_pair_list.append((image_name, bbox))

    # Select n from the list
    template_choices = random.sample(image_bbox_pair_list, k=args.num_templates)

    # Extract templates
    for image_name, bbox in template_choices:
        image_path = args.data_folder.joinpath(image_name)
        template = extract_template(image_path, bbox)
        save_path = output_folder.joinpath(image_name)

        print('Saving', (image_name, bbox), 'to:', save_path)
        template.save(save_path)


if __name__ == '__main__':
    main()
