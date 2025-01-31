import argparse
import json
import numpy as np
import os
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split dataset into train and test sets.'
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
        help='folder to output train/tes splits to',
        default='results/hw02_splits',
        type=pathlib.Path,
    )
    parser.add_argument(
        '-r',
        '--random-seed',
        help='random number seed, to ensure we always get the same train/test split',
        default=2020,
        type=int,
    )
    parser.add_argument(
        '-f',
        '--train-fraction',
        help='fraction of the dataset to include in the training set.. should be in (0, 1)',
        default=0.85,
        type=float,
    )
    parser.add_argument(
        '-a',
        '--annotations-path',
        help='JSON file with annotataions',
        default='data/hw02_annotations/annotations.json',
        type=pathlib.Path,
    )
    parser.add_argument(
        '-s',
        '--split-annotations',
        help='whether to split the annotations.json file as well',
        action='store_true',
    )

    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.random_seed)

    # Create output directory if needed
    args.output_folder.mkdir(exist_ok=True)

    # get sorted list of files:
    file_paths = sorted(args.data_folder.iterdir())
    # remove any non-JPEG files:
    file_paths = [f for f in file_paths if (f.suffix == '.jpg')]
    # Convert from pathlib.Path to string file names (no folder)
    file_names = [f.name for f in file_paths]

    # split file names into train and test
    num_train = round(len(file_names) * args.train_fraction)
    file_names_train = np.random.choice(file_names, size=num_train, replace=False)
    file_names_test = np.setdiff1d(file_names, file_names_train)

    assert (len(file_names_train) + len(file_names_test)) == len(file_paths)
    assert len(np.intersect1d(file_names_train, file_names_test)) == 0

    np.save(args.output_folder.joinpath('file_names_train.npy'), file_names_train)
    np.save(args.output_folder.joinpath('file_names_test.npy'), file_names_test)

    if args.split_annotations:
        with args.annotations_path.open('r') as f:
            annotations = json.load(f)

        # Use file_names_train and file_names_test to apply the split to the
        # annotations
        annotations_train = {
            k: v for k, v in annotations.items() if k in file_names_train
        }
        annotations_test = {
            k: v for k, v in annotations.items() if k in file_names_test
        }

        with args.output_folder.joinpath('annotations_train.json').open('w') as f:
            json.dump(annotations_train, f)

        with args.output_folder.joinpath('annotations_test.json').open('w') as f:
            json.dump(annotations_test, f)


if __name__ == '__main__':
    main()
