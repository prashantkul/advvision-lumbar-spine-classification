import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import vision_models.constants as constants
from vision_models.densenetmodel import DenseNetVisionModel
from vision_models.dataset import Dataset

def load_model_with_weights(weights_path, num_classes, input_shape):

    # model = tf.keras.models.load_model('linearclassifier.h5')
    # print(model.layers)
    # model.evaluate(X_test, y_test)

    # Create the model architecture
    model = DenseNetVisionModel(num_classes=num_classes, input_shape=input_shape, weights=None)
    
    # Load the model weights from the specified file
    try:
        model.load_weights(weights_path)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error when loading weights: {e}")
        return None
    
    # Compile the model (necessary if you want to evaluate it)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    
    print(model.summary())  # Optional: print the model summary
    return model

def prepare_data():
    # Initialize the dataset
    dataset = Dataset(batch_size=constants.BATCH_SIZE)
    
    # Extract labels from the dataset
    labels = dataset.label_list
    
    # Load validation and test data
    val_data = dataset.load_data(constants.VAL)
    test_data = dataset.load_data(constants.TEST)
    
    return labels, val_data, test_data

def predict_and_save(model, data, slices, output_prefix='output', threshold=0.7):
    # Make predictions
    predictions = model.predict(data)
    
    # Save the probabilities to CSV
    probs_csv_path = f'{output_prefix}_probabilities.csv'
    np.savetxt(probs_csv_path, predictions, delimiter=",")
    print(f'Saved probabilities to {probs_csv_path}')
    
    # Convert to binary predictions based on threshold
    binary_predictions = (predictions > threshold).astype(int)
    binary_csv_path = f'{output_prefix}_binary.csv'
    np.savetxt(binary_csv_path, binary_predictions, delimiter=",")
    print(f'Saved binary predictions to {binary_csv_path}')

def main():
    # Define the path to your weights file in the Git folder
    weights_path = 'vision_projects/vision_models_stage_1_Densenet_best_model.weights_08112024_1414.h5'
    
    # Initialize the dataset
    dataset = Dataset(batch_size=constants.BATCH_SIZE)
    
    # Extract labels from the dataset
    labels = dataset.label_list
    
    # Define model parameters
    slices = 200  # Number of slices per input sequence
    input_shape = (slices, 224, 224, 3)  # Adjust according to your model
    num_classes = len(labels)  # Define this as per your setup

    # Load the model with the weights
    model = load_model_with_weights(weights_path, num_classes=num_classes, input_shape=input_shape)
    
    if model is None:
        print("Model could not be loaded. Exiting.")
        return
    
    # Prepare data
    labels, val_data, test_data = prepare_data()
    
    # Choose the data type (validation or test) and make predictions
    data_type = constants.VAL  # or constants.TEST
    data = val_data if data_type == constants.VAL else test_data
    
    # Make predictions and save the results
    predict_and_save(model, data, slices, output_prefix=f'{data_type}_set')

if __name__ == '__main__':
    main()
