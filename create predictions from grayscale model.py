import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = './input/stage1_train/'
TEST_PATH = './input/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
limit = 10
train_ids = next(os.walk(TRAIN_PATH))[1][:limit]
test_ids = next(os.walk(TEST_PATH))[1][:limit]

# Get and resize train images and masks
X_train_rgb = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
X_train_gray = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    # Get RGB Image
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train_rgb[n] = img
    # Get Grayscale Image
    img = imread(path + '/grayscale/' + id_ + '.png')[:, :]
    img = np.expand_dims(resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
    # Normalize range of grayscale images
    img_max = np.amax(img)
    img_min = np.amin(img)
    img = 255 * (img - img_min) / (img_max - img_min)
    X_train_gray[n] = img
    # Get masks with boundaries
    mask = imread(path + '/masks with boundaries-2/' + id_ + '.png')
    test=list(set(tuple(v) for m2d in mask for v in m2d))
    if (0, 0, 0) in test or len(test) > 3:
        print(id_)
        print(test)
        imshow(mask)
    #mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    Y_train[n] = mask / 255

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/grayscale/' + id_ + '.png')[:,:]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = np.expand_dims(resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
    X_test[n] = img

print('Done!')

# Check if training data looks all right
ix = random.randint(0, len(train_ids)-1)
imshow(X_train_rgb[ix])
plt.show()
imshow(X_train_gray[ix].reshape((IMG_HEIGHT, IMG_WIDTH)))
plt.show()
#imshow(np.squeeze(Y_train[ix]))
imshow((Y_train[ix]*255).astype(np.uint8))
plt.show()
print(list(set(tuple(v) for m2d in Y_train[ix] for v in m2d)))

# Pixelwise Crossentropy
def pixelwise_crossentropy(target, output):
    output = tf.clip_by_value(output, 10e-8, 1. - 10e-8)
    return - tf.reduce_sum(target * tf.log(output))

# Define IoU metric
def mean_iou(y_true, y_pred): # this needs to be MAJORLY reworked
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Predict on train, val and test
model = load_model('model-dsbowl2018-gray-v1.2-0.h5', custom_objects={'pixelwise_crossentropy': pixelwise_crossentropy})
preds_train = model.predict(X_train_gray[:int(X_train_gray.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train_gray[int(X_train_gray.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train == preds_train.max(axis=3)[:,:,:,None]).astype(np.uint8)
print(preds_test)
print('check 1')
print(np.argwhere(np.all((preds_train_t-np.array([0,1,0]))==0, axis=3)))
preds_val_t = (preds_val == preds_val.max(axis=3)[:,:,:,None]).astype(np.uint8)
preds_test_t = (preds_test == preds_test.max(axis=3)[:,:,:,None]).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))


# Perform a sanity check on some random training samples
for i in range(10):
    ix = random.randint(0, len(preds_train_t) - 1)
    imshow(X_train_rgb[ix])
    plt.show()
    imshow(X_train_gray[ix].reshape((IMG_HEIGHT, IMG_WIDTH)))
    plt.show()
    #imshow(np.squeeze(Y_train[ix]))
    imshow(Y_train[ix]*255)
    plt.show()
    #imshow(np.squeeze(preds_train_t[ix]))
    imshow(preds_train_t[ix]*255)
    plt.show()


# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t) - 1)
print(len(preds_val_t))
print(ix)
imshow(X_train_rgb[int(X_train_rgb.shape[0]*0.9):][ix])
plt.show()
imshow(X_train_gray[int(X_train_gray.shape[0]*0.9):][ix].reshape((IMG_HEIGHT, IMG_WIDTH)))
plt.show()
imshow(Y_train[int(Y_train.shape[0]*0.9):][ix]*255)
plt.show()
imshow(preds_val_t[ix]*255)
plt.show()

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x[:,:,2] > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.2.csv', index=False)