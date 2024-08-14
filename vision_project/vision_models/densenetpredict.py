import tensorflow as tf
import numpy as np
import os
from vision_models.densenetmodel import DenseNetVisionModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class DenseNetModelPredictor:
    def __init__(self, weights_path, num_classes, input_shape):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self.load_model(weights_path)
    

    def load_model(self, weights_path):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        # Create an instance of your model
        model = DenseNetVisionModel(self.num_classes, self.input_shape)
        
        # Build the model by calling it on a dummy input
        dummy_input = tf.zeros((1,) + self.input_shape)
        _ = model(dummy_input)
        
        # Load the weights
        try:
            model.load_weights(weights_path)
            print(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            raise

        return model

    def preprocess_input(self, input_data):
        # Implement any preprocessing steps required for your model
        # This might include resizing, normalization, etc.
        # For this example, we'll assume the input is already in the correct format
        return input_data

    def predict(self, input_data):
        # Preprocess the input
        preprocessed_input = self.preprocess_input(input_data)
        
        # Make prediction
        predictions = self.model.predict(preprocessed_input)
        
        return predictions

    def predict_batch(self, batch_data):
        # Preprocess the batch
        preprocessed_batch = np.array([self.preprocess_input(sample) for sample in batch_data])
        
        # Make predictions on the batch
        batch_predictions = self.model.predict(preprocessed_batch)
        
        return batch_predictions

    def interpret_predictions(self, predictions, threshold=0.2):
        """
        Interpret raw predictions into binary format.
        
        :param predictions: numpy array of raw predictions, shape (n_samples, n_classes)
        :param threshold: threshold for positive prediction (default 0.5)
        :return: binary predictions
        """
        return (predictions >= threshold).astype(int)

    def save_predictions(self, predictions, file_path, sample_ids=None):
        """
        Save both raw and interpreted predictions to a CSV file.
        
        :param predictions: numpy array of raw predictions, shape (n_samples, n_classes)
        :param file_path: path to save the CSV file
        :param sample_ids: list of sample identifiers (optional)
        """
        # Ensure predictions is a numpy array
        predictions = np.array(predictions)
        
        # Interpret predictions
        interpreted_predictions = self.interpret_predictions(predictions)
        
        # Create DataFrames for raw and interpreted predictions
        df_raw = pd.DataFrame(predictions, 
                              columns=[f'Class_{i}_Raw' for i in range(self.num_classes)])
        df_interpreted = pd.DataFrame(interpreted_predictions, 
                                      columns=[f'Class_{i}_Interpreted' for i in range(self.num_classes)])
        
        # Combine raw and interpreted predictions
        df = pd.concat([df_raw, df_interpreted], axis=1)
        
        # Add sample IDs if provided
        if sample_ids is not None:
            df.insert(0, 'Sample_ID', sample_ids)
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        print(f"Raw and interpreted predictions saved to {file_path}")

    def visualize_predictions(self, predictions, file_path, sample_ids=None):
        """
        Visualize raw and interpreted predictions using heatmaps and save as a file.
        
        :param predictions: numpy array of raw predictions, shape (n_samples, n_classes)
        :param file_path: path to save the visualization file
        :param sample_ids: list of sample identifiers (optional)
        """
        interpreted_predictions = self.interpret_predictions(predictions)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        # Raw predictions heatmap
        sns.heatmap(predictions, annot=True, fmt='.2f', cmap='YlOrRd', 
                    xticklabels=[f'Class_{i}' for i in range(self.num_classes)],
                    yticklabels=sample_ids if sample_ids else range(predictions.shape[0]),
                    ax=ax1)
        ax1.set_title('Raw Predictions Heatmap')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Samples')
        
        # Interpreted predictions heatmap
        sns.heatmap(interpreted_predictions, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[f'Class_{i}' for i in range(self.num_classes)],
                    yticklabels=sample_ids if sample_ids else range(predictions.shape[0]),
                    ax=ax2)
        ax2.set_title('Interpreted Predictions Heatmap')
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Samples')
        
        plt.tight_layout()
        
        # Save the figure instead of showing it
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory
        print(f"Visualization saved to {file_path}")
    

    def evaluate(self, dataset_samples):
        all_predictions = []
        all_labels = []
        
        for batch in dataset_samples:
            features, labels = batch
            predictions = self.predict(features)
            
            
            # Get the index (class) with the highest probability
            max_prob_indices = np.argmax(predictions, axis=1)
            
            # Create a one-hot encoded array where the highest probability is 1 and others are 0
            interpreted_predictions = np.eye(predictions.shape[1])[max_prob_indices]
            print("Labels: ", labels)
            print("Predictions: ", interpreted_predictions)
            
            all_predictions.append(interpreted_predictions)
            all_labels.append(labels.numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        # Debug information
        print("Predictions shape:", predictions.shape)
        print("Labels shape:", labels.shape)
        print("Predictions dtype:", predictions.dtype)
        print("Labels dtype:", labels.dtype)
        print("Unique prediction values:", np.unique(predictions))
        print("Unique label values:", np.unique(labels))
        
        # Read test_split.csv
        test_df = pd.read_csv('test_split.csv')
        
        # Get unique class values from test_split.csv
        label_list = sorted(test_df['class'].unique())
        
        # Ensure the number of predictions matches the number of samples
        assert predictions.shape[0] == labels.shape[0], f"Number of predictions ({predictions.shape[0]}) doesn't match number of labels ({labels.shape[0]})"
        
        # Create a DataFrame with detailed results, using only the number of samples we have predictions for
        num_samples = predictions.shape[0]
        results_df = pd.DataFrame({
            'study_id': test_df['study_id'].iloc[:num_samples].values,  # Only use the first num_samples study_ids
        })
        
        # Add columns for each class prediction and true label
        for i, class_name in enumerate(label_list):
            results_df[f'{class_name}_pred'] = predictions[:, i]
            results_df[f'{class_name}_true'] = labels[:, i]
        
        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        
        try:
            precision_micro = precision_score(labels, predictions, average='micro', zero_division=0)
            recall_micro = recall_score(labels, predictions, average='micro', zero_division=0)
            f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        except Exception as e:
            print(f"Error computing metrics: {str(e)}")
            precision_micro = recall_micro = f1_micro = None
        
        results = {
            'accuracy': accuracy,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_score_micro': f1_micro,
            'detailed_results': results_df
        }
        
        # # Save detailed results to CSV
        # csv_filename = f"evaluation_results_samples_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        # results_df.to_csv(csv_filename, index=False)
        # print(f"\nDetailed evaluation results saved to {csv_filename}")
        
        return results
# weights_path = "best_model.weights.h5"
# num_classes = 25
# input_shape = (192, 224, 224, 3)
# 
# predictor = ModelPredictor(weights_path, num_classes, input_shape)
# 
# # For a single sample
# sample_input = np.random.rand(1, 192, 224, 224, 3)  # Replace with your actual input
# predictions = predictor.predict(sample_input)
# interpreted_predictions = predictor.interpret_predictions(predictions)
# 
# # For a batch of samples
# batch_input = np.random.rand(10, 192, 224, 224, 3)  # Replace with your actual batch input
# batch_predictions = predictor.predict_batch(batch_input)
# interpreted_batch_predictions = predictor.interpret_predictions(batch_predictions)