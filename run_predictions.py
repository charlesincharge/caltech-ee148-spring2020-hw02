import argparse
import pathlib
import os
import numpy as np
from scipy.stats import pearsonr
import json
from typing import List
from PIL import Image
from joblib import Parallel, delayed


def compute_convolution(I, T, stride: int = 1, padding: int = 0):
    """
    This function takes an image <I> and a template <T> (both numpy arrays)
    and returns a heatmap where each grid represents the output produced by
    convolution at each location.
    """
    # Validate input
    assert np.ndim(I) == np.ndim(T) == 3
    (n_rows_i, n_cols_i, n_channels_i) = np.shape(I)
    (n_rows_t, n_cols_t, n_channels_t) = np.shape(T)
    assert n_rows_t <= n_rows_i
    assert n_cols_t <= n_cols_i
    assert n_channels_t == n_channels_i

    # We downsize the heatmap slightly so that the template can match
    # only valid pixels
    # Calculate shapes along the convolution dimensions (non-channel)
    shape_i = np.array(I.shape[:-1], dtype=int)
    shape_t = np.array(T.shape[:-1], dtype=int)
    shape_h = ((shape_i + (2 * padding) - shape_t) // stride) + 1
    heatmap = np.zeros(shape_h)

    # Iterate over rows and columns of heatmap
    for row_h in range(shape_h[0]):
        for col_h in range(shape_h[1]):
            # Translate strides/padding to image-indexing
            row_i, col_i = heatmap_idx_to_image_idx(
                np.array([row_h, col_h]), stride=stride, padding=padding
            )

            # Slice input image to template size
            sub_image = I[row_i : (row_i + n_rows_t), col_i : (col_i + n_cols_t)]

            # Store the correlation between this image slice and the template
            corr = pearsonr(sub_image.flatten(), T.flatten())[0]
            heatmap[row_h, col_h] = corr

    return heatmap


def predict_boxes(
    heatmap: np.ndarray,
    bbox_shape: tuple,
    stride: int = 1,
    padding: int = 0,
    threshold: float = 0.7,
):
    """
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.

    Arguments
    ---------
    heatmap: np.array of confidence scores (correlations with template)
    bbox_shape: 2-tuple (n_rows_b, n_cols_b) of the bounding box, usually the
        same as the template shape.
    stride, padding: same values used to construct the heatmap
    """

    # Threshold heatmap to find objects
    object_detected_mask = heatmap > threshold
    object_locs = np.argwhere(object_detected_mask)

    bbox_list = []
    for row_h, col_h in object_locs:
        # Convert heatmap coordinates back to original coordinates
        tl_row_i, tl_col_i = heatmap_idx_to_image_idx(
            np.array([row_h, col_h]), stride=stride, padding=padding
        )

        # Bounding box size is pre-defined
        br_row_i, br_col_i = np.array([tl_row_i, tl_col_i]) + np.asarray(bbox_shape)

        score = heatmap[row_h, col_h]
        # Convert to native Python integers to fix JSON parsing
        bbox_list.append(
            [tl_row_i.item(), tl_col_i.item(), br_row_i.item(), br_col_i.item(), score]
        )

    return bbox_list


def heatmap_idx_to_image_idx(idx_h: int, stride: int, padding: int):
    """
    Helper function to convert between heatmap coordinates to image coordinates

    Arguments
    ---------
    idx_h : int or 1-D np.array of heatmap indices
    """
    idx_i = (stride * idx_h) - padding
    return idx_i


def detect_red_light_mf(I, template_list: List):
    """
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>.
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>.
    The first four entries are four integers specifying a bounding box
    (the row and column index of the top left corner and the row and column
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel

    template_list: list of numpy arrays corresponding to multiple filters
    """

    # Use a shallow, "wide"-CNN with multiple matched filters
    # If an image matches any one of the templates, it gets added.
    stride = 2  # Speed up the computations

    def helper_func(template):
        heatmap = compute_convolution(I, template, stride=stride)
        bbox_list = predict_boxes(heatmap, template.shape[:-1], stride=stride)
        return bbox_list

    # Compute template matches in parallel to speed up
    bbox_list_list = Parallel(n_jobs=-3)(
        delayed(helper_func)(template) for template in template_list
    )

    # Flatten list of lists (of lists)
    bbox_list = [bbox for bbox_list in bbox_list_list for bbox in bbox_list]

    # Check on output
    for idx, bbox in enumerate(bbox_list):
        assert len(bbox) == 5
        assert (bbox[4] >= 0.0) and (bbox[4] <= 1.0)

    return bbox_list


def parse_args():
    parser = argparse.ArgumentParser(
        description='Detect red lights in images using matched filters.'
    )
    parser.add_argument(
        '-d',
        '--data-folder',
        help='folder of images with red lights',
        default='data/RedLights2011_Medium',
        type=pathlib.Path,
    )
    parser.add_argument(
        '-s',
        '--splits-folder',
        help='folder with data splits',
        default='results/hw02_splits',
        type=pathlib.Path,
    )
    parser.add_argument(
        '-o',
        '--output-folder',
        help='folder to output predictions',
        default='results/hw02_preds',
        type=pathlib.Path,
    )
    parser.add_argument(
        '-t',
        '--template-folder',
        help='path to matched filter templates',
        default='templates/annotations_train',
        type=pathlib.Path,
    )
    parser.add_argument(
        '-n',
        '--num-images',
        help='number of images to process. defaults to all, set to int to process fewer (eg, for debugging)',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--done-tweaking',
        help='whether to use test data. Set to True when done with algorithm development',
        action='store_true',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load splits
    file_names_train = np.load(args.splits_folder.joinpath('file_names_train.npy'))
    file_names_test = np.load(args.splits_folder.joinpath('file_names_test.npy'))
    # Potentially sub-sample the file names
    if args.num_images is not None:
        file_names_train = np.random.choice(file_names_train, args.num_images)

    # Create folder for saving predictions, if it doesn't exist
    args.output_folder.mkdir(exist_ok=True)

    # Load in templates
    template_list = []
    for template_path in args.template_folder.iterdir():
        # Ignore non-jpg though
        if template_path.suffix != '.jpg':
            continue
        template = Image.open(template_path)
        template_list.append(np.asarray(template))

    '''
    Make predictions on the training set.
    '''
    if not args.done_tweaking:
        preds_train = {}
        for fname in file_names_train:
            print('Processing train set:', fname)

            # read image using PIL:
            I = Image.open(args.data_folder.joinpath(fname))

            # convert to numpy array:
            I = np.asarray(I)

            preds_train[fname] = detect_red_light_mf(I, template_list)

        # save preds (overwrites any previous predictions!)
        output_path = args.output_folder.joinpath('preds_train.json')
        with output_path.open('w') as f:
            print('Saving predictions to:', f.name)
            json.dump(preds_train, f)

    if args.done_tweaking:
        """
        Make predictions on the test set.
        """
        preds_test = {}
        for fname_test in file_names_test:
            print('Processing test set:', fname_test)

            # read image using PIL:
            I = Image.open(args.data_folder.joinpath(fname_test))

            # convert to numpy array:
            I = np.asarray(I)

            preds_test[fname_test] = detect_red_light_mf(I, template_list)

        # save preds (overwrites any previous predictions!)
        output_path_test = args.output_folder.joinpath('preds_test.json')
        with output_path_test.open('w') as f:
            print('Saving predictions to:', f.name)
            json.dump(preds_test, f)


if __name__ == '__main__':
    main()
