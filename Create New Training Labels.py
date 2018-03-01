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
from keras.layers.core import Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

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
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True)
        #newmask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        for i in range(IMG_HEIGHT):
            for j in range(IMG_WIDTH):
                if mask_[i,j] == 0:
                    continue
                elif (i == 0) or (j == 0) or (i == IMG_HEIGHT - 1) or (j == IMG_WIDTH - 1): # Boundary Point
                    mask[i,j] = [0, 1, 0]
                elif np.amin(mask_[i-1:i+2,j-1:j+2]) == 0: # Boundary Point
                    mask[i,j] = [0, 1, 0]
                else:
                    mask[i,j] = [0, 0, 1] # Interior Point

        #mask = np.maximum(mask, newmask)

    for i in range(IMG_HEIGHT):
        for j in range(IMG_WIDTH):
            if (mask[i,j] == [0, 0, 0]).all():  # External Point
                mask[i,j] = [1, 0, 0]

    for i in range(IMG_HEIGHT):
        for j in range(IMG_WIDTH):
            if i == 0:
                mk = 0
            else:
                mk = 1

            if i == IMG_HEIGHT - 1:
                nk = 0
            else:
                nk = 1

            if j == 0:
                ml = 0
            else:
                ml = 1

            if j == IMG_WIDTH - 1:
                nl = 0
            else:
                nl = 1

            for k in range(i-mk,i+nk+1):
                for l in range (j-ml, j+nl+1):
                    if k == i and l == j:
                        continue
                    elif (mask[k,l] == [0, 1, 0]).all():
                        if (mask[i, j] == [1, 0, 0]).all():  # External Point
                            mask[i, j] = [.5, .5, 0]
                            break

                        elif (mask[i, j] == [0, 0, 1]).all():  # External Point
                            mask[i, j] = [0, .5, .5]
                            break

    Y_train[n] = mask
    im = Image.fromarray((mask * 255).astype(dtype=np.uint8), 'RGB')
    directory = path + '/masks with boundaries-2/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    im.save(directory + id_ + '.png')

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')

# Check if training data looks all right
ix = random.randint(0, len(train_ids))
imshow(X_train[ix])
plt.show()
#imshow(np.squeeze(Y_train[ix]))
imshow(Y_train[ix]*255)
plt.show()

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Define

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (6,6), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.2) (c1)
c1 = Conv2D(16, (6,6), activation='elu', kernel_initializer='he_normal', padding='same') (s)

outputs = Conv2D(3, (8,8), activation = 'softmax', padding='same') (c1)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,
                    callbacks=[earlystopper, checkpointer])

# Predict on train, val and test
model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
#imshow(np.squeeze(Y_train[ix]))
imshow(Y_train[ix]*255)
plt.show()
#imshow(np.squeeze(preds_train_t[ix]))
imshow(preds_train_t[ix]*255)
plt.show()


# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
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
    lab_img = label(x > cutoff)
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
sub.to_csv('sub-dsbowl2018-1.csv', index=False)