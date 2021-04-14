import argparse
import os
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt


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


def compute_counts(preds, ground_truths, iou_thr=0.5, conf_thr=0.5):
    """
    This function takes a pair of dictionaries (with our JSON format; see ex.)
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives.
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <ground_truths> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    """
    TP = 0
    FP = 0
    FN = 0

    for pred_file, pred in preds.items():

        ground_truth = ground_truths[pred_file]
        for bbox_pred in pred:
            # Discard predicted bounding boxes with low confidence scores
            if bbox_pred[4] < conf_thr:
                continue

            for bbox_gt in ground_truth:
                # See if it matched any ground-truth boxes
                iou = compute_iou(bbox_pred[:4], bbox_gt)

                if iou > iou_thr:
                    # Count it as a true-positive-match
                    TP += 1
                    break
            else:
                # There were no true-matches for this prediction,
                # so count it as a false-positive
                FP += 1

        # False negatives: any bboxes we missed
        FN += (len(ground_truth) - TP)

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
        '-s',
        '--splits-folder',
        help='path to folder with splits',
        default='results/hw02_splits',
        type=pathlib.Path,
    )
    parser.add_argument(
        '-d',
        '--done-tweaking',
        help='Set to True when done with algorithm development',
        action='store_true',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    '''
    Load training data.
    '''
    with args.preds_folder.joinpath('preds_train.json').open('r') as f:
        preds_train = json.load(f)

    with args.splits_folder.joinpath('annotations_train.json').open('r') as f:
        ground_truths_train = json.load(f)

    if args.done_tweaking:

        """
        Load test data.
        """

        with args.preds_folder.joinpath('preds_test.json').open('r') as f:
            preds_test = json.load(f)

        with args.splits_folder.joinpath('annotations_test.json').open('r') as f:
            ground_truths_test = json.load(f)


    # For a fixed IoU threshold, vary the confidence thresholds.
    # The code below gives an example on the training set for one IoU threshold.


    # Plot all curves on the same figure
    fig, ax = plt.subplots()
    ax.set_title('Precision-recall for RedLights2011_Medium (train set)')
    # Different iou_thresholds to try:
    for iou_thr in [0.25, 0.5, 0.75]:
        # Different confidence thresholds to try
        confidence_thrs = np.linspace(start=0.5, stop=1)
        tp_train = np.zeros(len(confidence_thrs))
        fp_train = np.zeros(len(confidence_thrs))
        fn_train = np.zeros(len(confidence_thrs))
        for i, conf_thr in enumerate(confidence_thrs):
            tp_train[i], fp_train[i], fn_train[i] = compute_counts(
                preds_train, ground_truths_train, iou_thr=iou_thr, conf_thr=conf_thr
            )

        # Plot training set PR curves
        precision_train = tp_train / (tp_train + fp_train)
        recall_train = tp_train / (tp_train + fn_train)

        ax.plot(recall_train, precision_train, marker='o',
                 label=f'iou_thr={iou_thr}')

    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    fig.legend()
    plt.show()


    if args.done_tweaking:
        # Plot test set precision-recall curves
        print('Plotting test set PR curves.')

        # Plot all curves on the same figure
        fig, ax = plt.subplots()
        ax.set_title('Precision-recall for RedLights2011_Medium (test set)')
        # Different iou_thresholds to try:
        for iou_thr in [0.25, 0.5, 0.75]:
            # Different confidence thresholds to try
            confidence_thrs = np.linspace(start=0.5, stop=1)
            tp_test = np.zeros(len(confidence_thrs))
            fp_test = np.zeros(len(confidence_thrs))
            fn_test = np.zeros(len(confidence_thrs))
            for i, conf_thr in enumerate(confidence_thrs):
                tp_test[i], fp_test[i], fn_test[i] = compute_counts(
                    preds_test, ground_truths_test, iou_thr=iou_thr, conf_thr=conf_thr
                )

            # Plot testing set PR curves
            precision_test = tp_test / (tp_test + fp_test)
            recall_test = tp_test / (tp_test + fn_test)

            ax.plot(recall_test, precision_test, marker='o',
                     label=f'iou_thr={iou_thr}')

        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        fig.legend()
        plt.show()


if __name__ == '__main__':
    main()
