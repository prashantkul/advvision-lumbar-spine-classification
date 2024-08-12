import os
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage
from keras.models import load_model
import vision_models.constants as constants
from vision_models.densenetmodel import DenseNetVisionModel  # Import your model definition
from vision_models.dataset import Dataset
from vision_models.constants import TEST_DATA_PATH, TRAIN_LABEL_PATH, IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, TRAIN, TEST, VAL, DISEASE_THRESHOLD

def init():
    # Set up Google Cloud Storage client
    client = storage.Client()
    bucket_name = 'models_output_234324'  # Replace with your bucket name
    model_path = 'vision_models/stage_1/Densenet/best_model.weights_08112024_1414.h5'  # Replace with the desired model path

    def load_model_from_gcs(bucket_name, model_path, labels):
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(model_path)
        local_model_path = '/tmp/model.h5'
        blob.download_to_filename(local_model_path)

        # Check or set the expected input shape based on the model's design
        expected_input_shape = (200, 224, 224, 3)  # This should match what the model expects
        
        # Define your model architecture here
        model = DenseNetVisionModel(input_shape=expected_input_shape, num_classes=len(labels))
    
        # Define your model architecture here
        # model = DenseNetVisionModel(input_shape=(200, IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, 3), num_classes=len(labels))
        # model = DenseNetVisionModel(input_shape=(224, 224, 3), num_classes=len(labels))
        
        # Load the weights
        model.load_weights(local_model_path)
        return model

    # Initialize Dataset
    dataset = Dataset(batch_size=constants.BATCH_SIZE)

    # Extract labels from the dataset
    labels = dataset.label_list

    def load_data(data_type):
        if data_type in [VAL, TEST]:
            data = dataset.load_data(data_type)
        else:
            raise ValueError("data_type must be 'val' or 'test'")
        return data

    val_data = load_data(VAL)
    test_data = load_data(TEST)

    def preprocess_input(original_input):
        # Example: Reshape the input to match the expected input shape for the model
        processed_input = tf.reshape(original_input, [-1, 224, 224, 3])
        return processed_input

    def predict_and_save(model, data, threshold=0.7, output_prefix='output'):
        # Preprocess the input data to match the expected shape
        data = data.map(lambda x, y: (preprocess_input(x), y))

        # Predict probabilities
        predictions = model.predict(data)

        # Flattening data to match the original shape
        predictions = np.vstack([pred for pred in predictions])

        # Save probabilities to CSV
        probs_csv_path = f'{output_prefix}_probabilities.csv'
        np.savetxt(probs_csv_path, predictions, delimiter=",")
        print(f'Saved probabilities to {probs_csv_path}')

        # Convert to binary based on threshold
        binary_predictions = (predictions > threshold).astype(int)
        binary_csv_path = f'{output_prefix}_binary.csv'
        np.savetxt(binary_csv_path, binary_predictions, delimiter=",")
        print(f'Saved binary predictions to {binary_csv_path}')

    return load_model_from_gcs, val_data, test_data, predict_and_save, bucket_name, model_path, labels

def main():
    load_model_from_gcs, val_data, test_data, predict_and_save, bucket_name, model_path, labels = init()

    # Load the model
    model = load_model_from_gcs(bucket_name, model_path, labels)

    # Choose data type to run predictions on
    data_type = VAL  # Change to TEST if needed
    data = val_data if data_type == VAL else test_data

    # Run predictions and save the results
    predict_and_save(model, data, threshold=DISEASE_THRESHOLD, output_prefix=f'{data_type}_set')

if __name__ == '__main__':
    main()