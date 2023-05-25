#!/usr/bin/python3
import os
import json
import yaml

# Get the input file path from the environment variable
print(os.environ, file=os.sys.stderr)

input_file_path = f"{json.loads(os.environ['INPUT'])}/output.txt"

# train_input_file_path = f"{json.loads(os.environ['TRAIN'])}/output.txt"
# test_input_file_path = f"{json.loads(os.environ['TEST'])}/output.txt"

# Read the input file
with open(input_file_path, 'r') as input_file:
    input_file_content = input_file.read()


# with open(train_input_file_path, 'r') as input_file:
#     train_lines = [line.rstrip().split() for line in input_file]

# with open(test_input_file_path, 'r') as input_file:
#     test_lines = [line.rstrip().split() for line in input_file]

# print(train_lines[0])
# print(test_lines[0])

# Print the input file content
print(yaml.dump({"output": input_file_content}))
