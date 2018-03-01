import os
import sys
import random
import warnings
import csv
import numpy as np
import matplotlib.pyplot as plt

# TRAIN_PATH = './input/stage1_train/'
# TEST_PATH = './input/stage1_test/'
#
# train_ids = next(os.walk(TRAIN_PATH))[1]
# test_ids = next(os.walk(TEST_PATH))[1]

with open('./input/stage1_train_labels.csv/stage1_train_labels.csv') as csvfile:
    reader = csv.reader(csvfile)
    areas = []
    i = 0
    ix = -2
    for row in reader:
        if ix == -2: # header row
            ix += 1
            i += 1
            continue

        if ix == -1: # first label row
            ix += 1
            areas.append([row[0], []])

        if areas[ix][0] != row[0]: # if we are onto another image, start a new row in output
            ix += 1
            areas.append([row[0], []])

        numbers = list(map(int,row[1].split(' ')))
        runs = numbers[1::2]
        areas[ix][1].append(sum(runs))
        i += 1

print(areas)

areaStats = [[row[0], len(row[1]), np.mean(row[1]), np.std(row[1]), np.mean(row[1])/np.std(row[1])] for row in areas]
counts = [row[1] for row in areaStats]
means = [row[2] for row in areaStats]
stds = [row[3] for row in areaStats]

binwidth = 75
meanPlt = plt.hist(means, bins=range(0, int(np.ceil(max(means)/binwidth)*binwidth), binwidth))
plt.show()

binwidth = 50
stdPlt = plt.hist(stds, bins=range(0, int(np.ceil(max(stds)/binwidth)*binwidth), binwidth))
plt.show()

print(max(means))
print(max(stds))
print(range(0, int(np.ceil(max(means)/binwidth)*binwidth), binwidth))
print(max(means)/binwidth)

with open('./output/areaStats.csv', 'w') as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows([['image_id', 'number of masks', 'average area', 'average std', 'ratio of mean/std']])
    writer.writerows(areaStats)