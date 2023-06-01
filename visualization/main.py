#!/usr/bin/python3
import os
import json
import yaml
from matplotlib import pyplot as plt
import numpy as np

# Get the input file path from the environment variable
# print(os.environ, file=os.sys.stderr)


importances_file_path = f"{json.loads(os.environ['INPUT'])}/features.txt"
# importances_file_path = 'features.txt'

input_file_path = f"{json.loads(os.environ['INPUT'])}/output.txt"
# input_file_path = 'output.txt'

# train_input_file_path = f"{json.loads(os.environ['TRAIN'])}/output.txt"
# test_input_file_path = f"{json.loads(os.environ['TEST'])}/output.txt"

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
# print(idx)

train_data = all_data[:idx]
test_data = all_data[idx + 1:]

del all_data[idx]

feature_names = [vi[0] for vi in importances_data]
feature_importances = [float(vi[1]) for vi in importances_data]
feature_ranks = range(len(feature_names))

# print(feature_names, feature_importances)

complete_differences = np.clip([float(x[1]) - float(x[0]) for x in all_data], -100000, 100000)
train_differences = np.clip([float(x[1]) - float(x[0]) for x in train_data], -100000, 100000)
test_differences = np.clip([float(x[1]) - float(x[0]) for x in test_data], -100000, 100000)

# From https://www.tensorflow.org/decision_forests/tutorials/advanced_colab
# bar = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
# plt.yticks(feature_ranks, feature_names)
# plt.gca().invert_yaxis()

# for importance, patch in zip(feature_importances, bar.patches):
#   importance = float(importance)
#   plt.text(patch.get_x() + patch.get_width(), patch.get_y(), "{:.4f}".format(importance), va="top")

# plt.xlabel('INV_MEAN_MIN_DEPTH')
# plt.title("Don't really know")
# plt.tight_layout()
# plt.show()

# print('We come here!')

plt.bar(feature_ranks, feature_importances, tick_label=feature_names)
plt.title("Importance of all features")
plt.xlabel('Feature name')
plt.ylabel('Importance')
plt.xticks(rotation='vertical')
# plt.show()
plt.savefig('/result/importances.png')



fig = plt.figure()
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)
ax1.hist(train_differences, bins=50, label='Train Data', density=True, lw=3, fc=(1, 0, 0, 1))
ax1.title.set_text('Training data differences')
ax2.hist(test_differences, bins=50, label='Test Data', density=True, lw=3, fc=(0, 1, 0, 1))
ax2.title.set_text('Test data differences')
ax3.hist(complete_differences, bins=50, label='All Data', density=True, lw=3, fc=(0, 0, 1, 1))
ax3.title.set_text('All data differences')
fig.legend()
# plt.show()
# plt.hist(train_differences, bins=50, label='Train Data', density=True, lw=3, fc=(1, 0, 0, 0.5))
# plt.hist(test_differences, bins=50, label='Test Data', density=True, lw=3, fc=(0, 1, 0, 0.5))
# plt.hist(complete_differences, bins=50, label='All Data', density=True, lw=3, fc=(0, 0, 1, 0.5))
# plt.xlabel('Difference between prediction and actual price')
# plt.ylabel('Percentage of predictions')
# plt.legend()
# plt.show()
plt.savefig('/result/differences.png')


# train_pred = [x[0] for x in train_data]
# train_real = [x[1] for x in train_data]

# test_pred = [x[1] for x in test_data]
# test_real = [x[1] for x in test_data]



    # input_file_content = input_file.read()


# with open(train_input_file_path, 'r') as input_file:
#     train_lines = [line.rstrip().split() for line in input_file]

# with open(test_input_file_path, 'r') as input_file:
#     test_lines = [line.rstrip().split() for line in input_file]

# print(train_lines[0])
# print(test_lines[0])

# Print the input file content
# print(yaml.dump({"output": input_file_content}))
