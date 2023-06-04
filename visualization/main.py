#!/usr/bin/python3
import os
import json
import yaml
from matplotlib import pyplot as plt
import numpy as np

# Get the input file path from the environment variable
# print(os.environ, file=os.sys.stderr)

importances_file_path = f"{json.loads(os.environ['INPUT'])}/features.txt"

input_file_path = f"{json.loads(os.environ['INPUT'])}/output.txt"

importances_data = []

with open(importances_file_path, 'r') as input_file:
    importances_data = [line.split() for line in input_file]

all_data = []
train_data = []
test_data = []

# Read the input file
with open(input_file_path, 'r') as input_file:
    all_data = [line.split() for line in input_file]


idx = all_data.index([])

train_data = all_data[:idx]
test_data = all_data[idx + 1:]

del all_data[idx]

feature_names = [vi[0] for vi in importances_data]
feature_importances = [float(vi[1]) for vi in importances_data]
feature_ranks = range(len(feature_names))

complete_differences = np.clip([float(x[1]) - float(x[0]) for x in all_data], -100000, 100000)
train_differences = np.clip([float(x[1]) - float(x[0]) for x in train_data], -100000, 100000)
test_differences = np.clip([float(x[1]) - float(x[0]) for x in test_data], -100000, 100000)

plt.figure(figsize=(18,12))
plt.tick_params(axis='x', which='major', labelsize=11)
plt.bar(feature_ranks, feature_importances, tick_label=feature_names)
plt.title("Importance of all features")
plt.xlabel('Feature name')
plt.ylabel('Importance')
plt.xticks(rotation='vertical')
plt.savefig('/result/importances.png')



fig = plt.figure(figsize=(18, 12))
ax1 = plt.subplot(131)
ax1.tick_params(axis='x', which='major', rotation=30)
ax2 = plt.subplot(132)
ax2.tick_params(axis='x', which='major', rotation=30)
ax3 = plt.subplot(133)
ax3.tick_params(axis='x', which='major', rotation=30)
ax1.hist(train_differences, bins=50, label='Train Data', density=True, lw=3, fc=(1, 0, 0, 1))
ax1.title.set_text('Training data differences')
ax2.hist(test_differences, bins=50, label='Test Data', density=True, lw=3, fc=(0, 1, 0, 1))
ax2.title.set_text('Test data differences')
ax3.hist(complete_differences, bins=50, label='All Data', density=True, lw=3, fc=(0, 0, 1, 1))
ax3.title.set_text('All data differences')
fig.legend()
plt.savefig('/result/differences.png')


