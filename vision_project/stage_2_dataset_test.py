import os
import pandas as pd
import tensorflow as tf

from vision_project.vision_models.stage2.helper import target_labels_to_tensor, tensorize
from vision_project.vision_models.stage2.severity_prediction_dataset import SeverityPredictionDataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

ground_truth_path = 'vision_project/vision_models/stage2/data/train.csv'
predictions_df = pd.read_csv(ground_truth_path)

balance_variables_df = pd.read_csv('vision_models/stage2/data/train_label_coordinates.csv')
balance_variables = pd.get_dummies(df=balance_variables_df, columns=balance_variables_df.columns.difference(['study_id', 'series_id']))

dataset_instance = SeverityPredictionDataset( 
    batch_size = 1, 
    balance_variables = balance_variables,
)

stage_1_target_labels = pd.read_csv('vision_project/vision_models/stage2/data/train_label_coordinates.csv')
dataset_instance = dataset_instance.augment(predictions=stage_1_target_labels)

dataset_instance = tensorize(dataset_instance)
dataset, steps_per_epoch  = dataset_instance.load_data("test")

# print
print("Steps per epoch: ", steps_per_epoch)

# Create an iterator
iterator = iter(dataset)

try:
    element = dataset.take(1)
    for img, label in element:
        print("Image tensor shape: ", img.shape)
        print("Label tensor shape: ", label.shape)
        break
except tf.errors.OutOfRangeError:
    print("End of dataset")