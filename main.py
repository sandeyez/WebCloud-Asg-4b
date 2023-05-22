# CODE USED AS STARTING POINT: https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf/notebook

import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=label, task=tfdf.keras.Task.REGRESSION)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label=label, task=tfdf.keras.Task.REGRESSION)


# Create a random forest model.
random_forests = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)


# Train the model.
random_forests.fit(x=train_ds)


random_forests.compile(metrics=["mse"])
evaluation = random_forests.evaluate(test_ds, return_dict=True)


for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")
