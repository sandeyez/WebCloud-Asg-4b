#!/usr/bin/python3

# CODE USED AS STARTING POINT: https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf/notebook

import os
import json
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import sys

# Redirect all stdout to stderr
sys.stdout = sys.stderr

VARIABLE_IMPORTANCE_METRIC = 'NUM_AS_ROOT'

# Get the input file path from the environment variable
# print(os.environ, file=os.sys.stderr)

input_file_path = f"{json.loads(os.environ['HOUSING'])}"

# We import the training- and testing-dataset from the specified paths.
data = pd.read_csv(input_file_path)
# We drop the 'Id' column because it is not a feature.
df = data.drop(['Id'], axis=1)


# We split the dataset into a training- and testing-dataset.
# 80% of the data is used for training and the remaining 20% for testing.
train_df = df.sample(frac=0.8, random_state=0)
test_df = df.drop(train_df.index)


# Convert the pandas dataframe to a tensorflow dataset.
# Because this is a regression problem, we specify the task as a regression task.
# The label is the column 'SalePrice'. This is the value we want to predict.
label = 'SalePrice'
# complete_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df, label=label, task=tfdf.keras.Task.REGRESSION)
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=label, task=tfdf.keras.Task.REGRESSION)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label=label, task=tfdf.keras.Task.REGRESSION)


# Create a random forest model.
random_forests = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION, compute_oob_variable_importances=True)


# Train the model.
random_forests.fit(x=train_ds)


random_forests.compile(metrics=["mse"])

inspector = random_forests.make_inspector()

importances = inspector.variable_importances()['INV_MEAN_MIN_DEPTH']

feature_names = [vi[0].name for vi in importances]
feature_importances = [vi[1] for vi in importances]

train_predictions = random_forests.predict(x=train_ds).flatten()
test_predictions = random_forests.predict(x=test_ds).flatten()

train_actual_prices = np.array(train_df['SalePrice'])
test_actual_prices = np.array(test_df['SalePrice'])

train_merged = list(zip(train_predictions, train_actual_prices))
test_merged = list(zip(test_predictions, test_actual_prices))

features_merged = list(zip(feature_names, feature_importances))

# with open(f"output.txt", 'w') as output_file:
with open(f"/result/output.txt", 'w') as output_file:
    for pred, actual in train_merged:
        output_file.write("{} {}\n".format(pred, actual))
    output_file.write("\n")
    for pred, actual in test_merged:
        output_file.write("{} {}\n".format(pred, actual))


# with open(f"features.txt", 'w') as output_file:
with open(f"/result/features.txt", 'w') as output_file:
    for name, importance in features_merged:
        output_file.write("{} {}\n".format(name, importance))

