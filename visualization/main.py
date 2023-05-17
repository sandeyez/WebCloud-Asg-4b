#!/usr/bin/python3
import os
import json
import yaml

# Get the input file path from the environment variable
print(os.environ, file=os.sys.stderr)
input_file_path = f"{json.loads(os.environ['INPUT'])}/output.txt"

# Read the input file
with open(input_file_path, 'r') as input_file:
    input_file_content = input_file.read()

# Print the input file content
print(yaml.dump({"output": input_file_content}))
