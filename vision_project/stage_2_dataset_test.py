import os
import pandas as pd
import tensorflow as tf

from vision_project.vision_models.stage2.helper import target_labels_to_tensor, tensorize
from vision_project.vision_models.stage2.severity_prediction_dataset import SeverityPredictionDataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

# TODO: steve to give me the real file
target_label_path = 'vision_project/vision_models/stage2/data/train.csv'
predictions_df = pd.read_csv(target_label_path)

dataset_instance = SeverityPredictionDataset( 
    batch_size = 1, 
    balance_variables = predictions_df.columns,
)

# TODO: we will need to pull the train label coordinates we get from dharti's predictions
stage_1_target_labels = pd.read_csv('vision_project/vision_models/stage2/data/train_label_coordinates.csv')
dataset_instance = dataset_instance.augment(predictions=predictions_df)

dataset_instance = tensorize(dataset_instance)

# load the dataset
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