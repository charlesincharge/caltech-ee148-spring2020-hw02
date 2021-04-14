import os
import json
import numpy as np


def compute_iou(box_1, box_2):
    """
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    """
    tl_row_1, tl_col_1, br_row_1, br_col_1 = box_1
    tl_row_2, tl_col_2, br_row_2, br_col_2 = box_2

    assert tl_row_1 < br_row_1
    assert tl_col_1 < br_col_1
    assert tl_row_2 < br_row_2
    assert tl_col_2 < br_col_2

    # Compute area of each respective box
    area_1 = (br_row_1 - tl_row_1) * (br_col_1 - tl_col_1)
    area_2 = (br_row_2 - tl_row_2) * (br_col_2 - tl_col_2)

    # Compute area of intersection
    tl_row_i = max(tl_row_1, tl_row_2)
    tl_col_i = max(tl_col_1, tl_col_2)
    br_row_i = min(br_row_1, br_row_2)
    br_col_i = min(br_col_1, br_col_2)
    if (br_row_i < tl_row_i) or (br_col_i < tl_col_i):
        intersection_area = 0
    else:
        intersection_area = (br_row_i - tl_row_i) * (br_col_i - tl_col_i)

    # Compute area of union
    union_area = area_1 + area_2 - intersection_area

    iou = intersection_area / union_area
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    """
    This function takes a pair of dictionaries (with our JSON format; see ex.)
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives.
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    """
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.iteritems():
        gt = gts[pred_file]
        for i in range(len(gt)):
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])

    '''
    END YOUR CODE
    '''

    return TP, FP, FN


def parse_args():
    parser = argparse.ArgumentParser(description='compute error metrics from annotations.')
    parser.add_argument(
        '-p',
        '--preds-folder',
        help='path to bounding box predictions',
        default='results/hw02_preds/',
        type=pathlib.Path,
    )
    parser.add_argument(
        '-a',
        '--annotations-path',
        help='path to bounding box ground-truth folder',
        default='data/hw02_annotations',
        type=pathlib.Path,
    )
    parser.add_argument(
        '-s',
        '--splits-path',
        help='path to folder with splits',
        default='results/hw02_splits',
        type=pathlib.Path,
    )
    parser.add_argument(
        '-d',
        '--done-tweaking',
        help='Set to True when done with algorithm development'
        action='store_true',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # load splits:
    file_names_train = np.load(args.splits_path.joinpath('file_names_train.npy'))
    file_names_test = np.load(args.splits_path.joinpath('file_names_test.npy'))

    '''
    Load training data.
    '''
    with args.preds_path.joinpath('preds_train.json').open('r') as f:
        preds_train = json.load(f)

    with args.annotations_path.joinpath('annotations_train.json').open('r') as f:
        gts_train = json.load(f)

    if args.done_tweaking:

        """
        Load test data.
        """

        with args.preds_path.joinpath('preds_test.json').open('r') as f:
            preds_test = json.load(f)

        with args.annotations_path.joinpath('annotations_test.json').open('r') as f:
            gts_test = json.load(f)


    # For a fixed IoU threshold, vary the confidence thresholds.
    # The code below gives an example on the training set for one IoU threshold.


    confidence_thrs = np.sort(
        np.array([preds_train[fname][4] for fname in preds_train], dtype=float)
    )  # using (ascending) list of confidence scores as thresholds
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(
            preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr
        )

    # Plot training set PR curves

    if args.done_tweaking:
        print('Code for plotting test set PR curves.')


if __name__ == '__main__':
    main()
