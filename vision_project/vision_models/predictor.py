import tensorflow as tf
import numpy as np
import pandas as pd
import constants
from vision_models.densenetpredict import DenseNetModelPredictor
from vision_models.densenetmodel import DenseNetVisionModel
from vision_models.dataset import Dataset
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from tfidgenerator import TFSampleIDGenerator

# Usage example:
weights_path = "best_model.weights.h5"
num_classes = 25
input_shape = (200, 224, 224, 3)
batch_size = 4

def _get_strategy():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) > 1:
            print(f"Using MirroredStrategy with {len(gpus)} GPUs")
            return tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        elif len(gpus) == 1:
            print("Using single GPU")
            return tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            print("Using CPU")
            return tf.distribute.OneDeviceStrategy(device="/cpu:0")

def load_data(mode):
        print("Creating datasets...")
        
        image_loader = Dataset(batch_size=batch_size)
        dataset = image_loader.load_data(mode)

        return dataset
        
# Initialize your predictor
predictor = DenseNetModelPredictor(weights_path, num_classes, input_shape)

# Initialize your dataset
dataset = load_data(constants.TEST)

# Take 100 samples
dataset_samples = dataset.take(10)

# Run evaluation
evaluation_results = predictor.evaluate(dataset_samples)

# Print results
print("\nEvaluation Results:")
if isinstance(evaluation_results, pd.DataFrame):
    print("Results are in DataFrame format:")
    print(f"Shape: {evaluation_results.shape}")
    print("Columns:")
    for column in evaluation_results.columns:
        print(f"- {column}")
    print("\nFirst few rows:")
    print(evaluation_results.head())
else:
    for metric, value in evaluation_results.items():
        if isinstance(value, pd.DataFrame):
            print(f"{metric}:")
            print(f"  Shape: {value.shape}")
            print("  Columns:")
            for column in value.columns:
                print(f"  - {column}")
        elif isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    if 'detailed_results' in evaluation_results:
        detailed_results = evaluation_results['detailed_results']
        print("\nDetailed Results Summary:")
        print(f"Number of samples: {len(detailed_results)}")
        print("First few rows:")
        print(detailed_results.head())


