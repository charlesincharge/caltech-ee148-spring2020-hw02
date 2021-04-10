import os
import numpy as np
import json
from PIL import Image


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
    # Have not yet implemented this for non-default stride/padding
    if stride != 1:
        raise NotImplementedError
    if padding != 0:
        raise NotImplementedError

    # We downsize the heatmap slightly so that the template can match
    # only valid pixels
    # Calculate shapes along the convolution dimensions (non-channel)
    shape_i = np.array(I.shape[:-1], dtype=int)
    shape_t = np.array(T.shape[:-1], dtype=int)
    shape_h = ((shape_i + (2 * padding) - shape_h) // stride) + 1
    heatmap = np.random.zeros(shape_h)

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
            corr_matrix = np.corrcoef(sub_image.flatten(), T.flatten())
            corr = corr_matrix[1, 0]
            heatmap[row_h, col_h] = corr

    return heatmap


def predict_boxes(
    heatmap: np.ndarray,
    bbox_shape: tuple,
    stride: int = 1,
    padding: int = 0,
    threshold: float = 0.9,
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
        br_row, br_col = np.array([tl_row_i, tl_col_i]) + np.asarray(bbox_shape)

        score = heatmap[row_h, col_h]
        bbox_list.append([tl_row, tl_col, br_row, br_col, score])

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


def detect_red_light_mf(I):
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
    """

    '''
    BEGIN YOUR CODE
    '''
    template_height = 8
    template_width = 6

    # You may use multiple stages and combine the results
    T = np.random.random((template_height, template_width))

    heatmap = compute_convolution(I, T)
    output = predict_boxes(heatmap, T.shape)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output


# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path, 'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_Path, 'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True)  # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path, 'preds_train.json'), 'w') as f:
    json.dump(preds_train, f)

if done_tweaking:
    """
    Make predictions on the test set.
    """
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path, file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path, 'preds_test.json'), 'w') as f:
        json.dump(preds_test, f)
