#!/usr/bin/python3

# CODE USED AS STARTING POINT: https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf/notebook

import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


DATASET_PATH = 'data.csv'
VARIABLE_IMPORTANCE_METRIC = 'NUM_AS_ROOT'


# We import the training- and testing-dataset from the specified paths.
data = pd.read_csv(DATASET_PATH)
# We drop the 'Id' column because it is not a feature.
df = data.drop(['Id'], axis=1)


# We split the dataset into a training- and testing-dataset.
# 80% of the data is used for training and the remaining 20% for testing.
train_df = df.sample(frac=0.8, random_state=0)
test_df = df.drop(train_df.index)
print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")


# Convert the pandas dataframe to a tensorflow dataset.
# Because this is a regression problem, we specify the task as a regression task.
# The label is the column 'SalePrice'. This is the value we want to predict.
label = 'SalePrice'
# complete_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df, label=label, task=tfdf.keras.Task.REGRESSION)
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=label, task=tfdf.keras.Task.REGRESSION)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label=label, task=tfdf.keras.Task.REGRESSION)


# Create a random forest model.
random_forests = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)


# Train the model.
random_forests.fit(x=train_ds)


random_forests.compile(metrics=["mse"])
# evaluation = random_forests.evaluate(test_ds, return_dict=True)

train_predictions = random_forests.predict(x=train_ds).flatten()
print(train_predictions)
print(train_predictions[0])
test_predictions = random_forests.predict(x=test_ds).flatten()
# complete_predictions = random_forests.predict(x=complete_ds)[0]

train_actual_prices = np.array(train_df['SalePrice'])
test_actual_prices = np.array(test_df['SalePrice'])

train_merged = list(zip(train_predictions, train_actual_prices))
test_merged = list(zip(test_predictions, test_actual_prices))

# print(train_merged)
# print(test_merged)


# for name, value in evaluation.items():
#   print(f"{name}: {value:.4f}")


with open(f"/result/output.txt", 'w') as output_file:
    for pred, actual in train_merged:
        output_file.write("{} {}\n".format(pred, actual))
    output_file.write("\n")
    for pred, actual in test_merged:
        output_file.write("{} {}\n".format(pred, actual))

# with open(f"/result/test.txt", 'w') as output_file:
#     for pred, actual in test_merged:
#         output_file.write("{} {}\n".format(pred, actual))

# with open(f"/result/train.txt", 'w') as output_file:
#     for pred, actual in train_merged:
#         output_file.write("{} {}".format(pred, actual))

# with open(f"/result/output.txt", 'w') as output_file:

