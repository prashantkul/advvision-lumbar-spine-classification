import os
import pandas as pd
import tensorflow as tf

from vision_models.stage2.helper import tensorize
from vision_models.stage2.severity_prediction_dataset import SeverityPredictionDataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

ground_truth_path = 'vision_models/stage2/data/stage1_ground_truth.csv'
stratification_variables_path = 'vision_models/stage2/data/train.csv'

predictions_df = pd.read_csv(ground_truth_path)
stratification_variables = pd.read_csv(stratification_variables_path)

balance_variables = stratification_variables.columns.difference(['study_id', 'series_id'])

dataset_instance = SeverityPredictionDataset(
    disease_predictions=predictions_df,
    balance_variables=balance_variables
)

stage_1_target_labels = pd.read_csv('vision_models/stage2/data/train_label_coordinates.csv')
dataset_instance = dataset_instance.augment(predictions=stage_1_target_labels)

dataset_instance = tensorize(dataset_instance)
dataset, steps_per_epoch  = dataset_instance.load_data("test")

print("Steps per epoch: ", steps_per_epoch)
iterator = iter(dataset)

try:
    element = dataset.take(1)
    for img, label in element:
        print("Image tensor shape: ", img.shape)
        print("Label tensor shape: ", label.shape)
        break
except tf.errors.OutOfRangeError:
    print("End of dataset")