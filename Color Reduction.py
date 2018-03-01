import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
from PIL import Image,ImageOps

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tqdm import tqdm
from itertools import chain
from skimage import img_as_ubyte
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from sklearn.decomposition import PCA

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = './input/stage1_train/'
TEST_PATH = './input/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# Get train and test IDs
train_test_limit = 50
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# Get and resize train images and masks
print('Getting and performing color reduction on train images ... ')
sys.stdout.flush()
variance_ratios = np.zeros((len(train_ids),3))
top_components = np.zeros((len(train_ids),3))
color_means = np.zeros((len(train_ids),3))
pca = PCA(n_components=3)
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    colors = np.reshape(img, (-1, 3))
    color_mean = colors.mean(axis=0)
    normalized = colors - color_mean[None,:]
    pca.fit(normalized)
    variance_ratios[n] = pca.explained_variance_ratio_
    top_components[n] = pca.components_[0]
    color_means[n] = color_mean
    norm_grayscale = [np.dot(vec,pca.components_[0]) for vec in normalized]
    img_grayscale = 255*(norm_grayscale + np.min(norm_grayscale)) / (np.max(norm_grayscale - np.min(norm_grayscale)))
    if (np.median(img_grayscale) > 127.5):
        img_grayscale = - (img_grayscale - 255)
    img_grayscale = np.reshape(img_grayscale, (IMG_HEIGHT, IMG_HEIGHT)).astype(np.uint8)

    # save pca im
    im = Image.fromarray(img_grayscale)
    directory = path + '/color_pca/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    im.save(directory + id_ + '.png')

    # create and save grayscale of original img
    directory = path + '/grayscale/'
    gray = Image.fromarray(img.astype(np.uint8))
    gray = ImageOps.grayscale(gray)
    img_g = np.fromstring(gray.tobytes(), dtype=np.uint8)
    if (np.median(img_g) > np.mean(img_g)):
        img_g = - (img_g - 255)
    img_g = img_g.reshape((IMG_HEIGHT,IMG_WIDTH))
    gray = Image.fromarray(img_g)
    if not os.path.exists(directory):
        os.makedirs(directory)
    gray.save(directory + id_ + '.png')

    # some checks
    if (random.random() > 5):
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        ax1.imshow(img.astype(np.uint8))
        ax2.imshow(img_grayscale, vmin=0, vmax=255, cmap='Greys')
        ax3.imshow(img_g,vmin=0, vmax=255, cmap='Greys')
        plt.show()
        plt.imshow(img_grayscale)
        plt.colorbar()
        plt.show()
        plt.imshow(img_g)
        plt.colorbar()
        plt.show()



# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    colors = np.reshape(img, (-1, 3))
    color_mean = colors.mean(axis=0)
    normalized = colors - color_mean[None, :]
    pca.fit(normalized)
    variance_ratios[n] = pca.explained_variance_ratio_
    top_components[n] = pca.components_[0]
    color_means[n] = color_mean
    norm_grayscale = [np.dot(vec, pca.components_[0]) for vec in normalized]
    img_grayscale = 255 * (norm_grayscale + np.min(norm_grayscale)) / (np.max(norm_grayscale - np.min(norm_grayscale)))
    if (np.median(img_grayscale) > 127.5):
        img_grayscale = - (img_grayscale - 255)
    img_grayscale = np.reshape(img_grayscale, (IMG_HEIGHT, IMG_HEIGHT)).astype(np.uint8)

    # save pca im
    im = Image.fromarray(img_grayscale)
    directory = path + '/color_pca/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    im.save(directory + id_ + '.png')

    # create and save grayscale of original img
    directory = path + '/grayscale/'
    gray = Image.fromarray(img.astype(np.uint8))
    gray = ImageOps.grayscale(gray)
    img_g = np.fromstring(gray.tobytes(), dtype=np.uint8)
    if (np.median(img_g) > np.mean(img_g)):
        img_g = - (img_g - 255)
    img_g = img_g.reshape((IMG_HEIGHT, IMG_WIDTH))
    gray = Image.fromarray(img_g)
    if not os.path.exists(directory):
        os.makedirs(directory)
    gray.save(directory + id_ + '.png')

print('Done!')