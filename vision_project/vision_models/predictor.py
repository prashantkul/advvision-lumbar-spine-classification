import tensorflow as tf
import numpy as np
import constants
from densenetpredict import DenseNetModelPredictor
from densenetmodel import DenseNetVisionModel
from dataset import Dataset
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from tfidgenerator import TFSampleIDGenerator

# Usage example:
weights_path = "best_model.weights.h5"
num_classes = 25
input_shape = (200, 224, 224, 3)
batch_size = 12

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
        
# Initialize your dataset
dataset = load_data(constants.TEST)

# Take exactly 4 samples
dataset_samples = dataset.take(100)

dataset_4_samples = dataset.take(4)

# # Initialize the ID generator with the 4-sample dataset
id_generator = TFSampleIDGenerator(dataset_4_samples)

# # Generate IDs for the 4 samples
sample_ids = id_generator.get_ids(4)

predictor = DenseNetModelPredictor(weights_path, num_classes, input_shape)

# Make predictions on the 4 samples
predictions = predictor.predict(dataset_4_samples)

# Interpret predictions
interpreted_predictions = predictor.interpret_predictions(predictions)

# Save predictions for the 4 samples
predictor.save_predictions(predictions, "predictions_4_samples.csv", sample_ids=sample_ids)

# If you want to visualize the predictions (assuming you have this method)
#predictor.visualize_predictions(predictions, sample_ids=sample_ids)

# Print out some information
print(f"Processed {len(sample_ids)} samples.")
print("Sample IDs:", sample_ids)
print("Predictions shape:", predictions.shape)
print("Interpreted predictions:", interpreted_predictions)

# Evaluate on the entire test dataset
evaluation_results = predictor.evaluate(dataset_samples)

print("\nEvaluation Results:")
print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
print(f"Micro-averaged F1 Score: {evaluation_results['f1_score_micro']:.4f}")
print("F1 Scores per class:")
for class_name, f1 in evaluation_results['f1_score_per_class'].items():
    print(f"  {class_name}: {f1:.4f}")

# # For a batch of samples
# batch_input = np.random.rand(10, 200, 224, 224, 3)  # Replace with your actual batch input
# batch_predictions = predictor.predict_batch(batch_input)
# interpreted_batch_predictions = predictor.interpret_predictions(batch_predictions)

