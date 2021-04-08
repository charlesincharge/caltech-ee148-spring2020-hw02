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
            row_i = (stride * row_h) - padding
            col_i = (stride * col_h) - padding

            # Slice input image to template size
            sub_image = I[row_i : (row_i + n_rows_t), col_i : (col_i + n_cols_t)]

            # Store the correlation between this image slice and the template
            corr_matrix = np.corrcoef(sub_image.flatten(), T.flatten())
            corr = corr_matrix[1, 0]
            heatmap[row_h, col_h] = corr

    return heatmap


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    box_height = 8
    box_width = 6

    num_boxes = np.random.randint(1,5)

    for i in range(num_boxes):
        (n_rows,n_cols,n_channels) = np.shape(I)

        tl_row = np.random.randint(n_rows - box_height)
        tl_col = np.random.randint(n_cols - box_width)
        br_row = tl_row + box_height
        br_col = tl_col + box_width

        score = np.random.random()

        output.append([tl_row,tl_col,br_row,br_col, score])

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    '''
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
    '''

    '''
    BEGIN YOUR CODE
    '''
    template_height = 8
    template_width = 6

    # You may use multiple stages and combine the results
    T = np.random.random((template_height, template_width))

    heatmap = compute_convolution(I, T)
    output = predict_boxes(heatmap)

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
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_Path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
