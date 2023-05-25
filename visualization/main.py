#!/usr/bin/python3
import os
import json
import yaml
from matplotlib import pyplot as plt

# Get the input file path from the environment variable
print(os.environ, file=os.sys.stderr)

input_file_path = f"{json.loads(os.environ['INPUT'])}/output.txt"
# input_file_path = 'output.txt'

# train_input_file_path = f"{json.loads(os.environ['TRAIN'])}/output.txt"
# test_input_file_path = f"{json.loads(os.environ['TEST'])}/output.txt"

all_data = []
train_data = []
test_data = []

# Read the input file
with open(input_file_path, 'r') as input_file:
    all_data = [line.split() for line in input_file]


idx = all_data.index([])
print(idx)

train_data = all_data[:idx]
test_data = all_data[idx + 1:]

del all_data[idx]

complete_differences = [float(x[1]) - float(x[0]) for x in all_data]
train_differences = [float(x[1]) - float(x[0]) for x in train_data]
test_differences = [float(x[1]) - float(x[0]) for x in test_data]


plt.hist(train_differences, bins=25, label='Train Data', density=True, lw=3, fc=(1, 0, 0, 1))
plt.hist(test_differences, bins=25, label='Test Data', density=True, lw=3, fc=(0, 1, 0, 0.75))
plt.hist(complete_differences, bins=25, label='All Data', density=True, lw=3, fc=(0, 0, 1, 0.5))
plt.xlabel('Difference between prediction and actual price')
plt.ylabel('Amount of predictions')
plt.legend()
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
