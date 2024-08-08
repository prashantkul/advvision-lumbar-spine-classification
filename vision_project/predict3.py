import os
import tensorflow as tf
from vision_models.utils import VisionUtils
from vision_models.dataset import Dataset
from vision_models.densenetmodel import DenseNetVisionModel
import vision_models.constants as constants
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

class VisionModelPredictor:
    def __init__(self, model_path, mode='val'):
        self.vutil = VisionUtils()
        self.batch_size = constants.BATCH_SIZE
        self.dataset = Dataset(
            image_dir=constants.TRAIN_DATA_PATH if mode in ['val', 'test'] else constants.TEST_DATA_PATH,
            label_coordinates_csv=constants.TRAIN_LABEL_CORD_PATH if mode in ['val', 'test'] else None,
            labels_csv=constants.TRAIN_LABEL_PATH if mode in ['val', 'test'] else None,
            batch_size=self.batch_size
        )
        self.input_shape = (192, constants.IMAGE_SIZE_HEIGHT, constants.IMAGE_SIZE_WIDTH, 3)
        self.num_classes = 25
        self.model = self.load_model(model_path)
        self.mode = mode

    def load_model(self, model_path):
        model = DenseNetVisionModel(num_classes=self.num_classes, input_shape=self.input_shape, weights=None)
        model.load_weights(model_path)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def evaluate(self, dataset):
        results = self.model.evaluate(dataset)
        loss, accuracy = results[0], results[1]
        return loss, accuracy

    def save_predictions(self, predictions, output_csv, study_ids, series_ids):
        predictions_df = pd.DataFrame(predictions, columns=[f'class_{i}' for i in range(predictions.shape[1])])
        predictions_df.insert(0, 'study_id', study_ids)
        predictions_df.insert(0, 'series_id', series_ids)
        predictions_df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")

    def convert_to_binary_and_aggregate(self, predictions, study_ids, series_ids, threshold, binary_output_csv, aggregated_output_csv):
        binary_predictions = (predictions > threshold).astype(int)
        binary_predictions_df = pd.DataFrame(binary_predictions, columns=[f'class_{i}' for i in range(predictions.shape[1])])
        binary_predictions_df.insert(0, 'study_id', study_ids)
        binary_predictions_df.insert(1, 'series_id', series_ids)
        binary_predictions_df.to_csv(binary_output_csv, index=False)
        print(f"Binary predictions saved to {binary_output_csv}")
        aggregated_predictions_df = binary_predictions_df.groupby('study_id').max().drop(columns={'series_id'}).reset_index()
        aggregated_predictions_df.to_csv(aggregated_output_csv, index=False)
        print(f"Aggregated binary predictions saved to {aggregated_output_csv}")

    def predict(self, dataset, output_csv):
        y_pred = []
        study_ids = []
        series_ids = []
        predictions = []
        for images, labels in dataset:
            predictions_batch = self.model.predict(images)
            y_pred.extend(np.argmax(predictions_batch, axis=1))
            study_ids.extend(labels['study_id'].numpy())
            series_ids.extend(labels['series_id'].numpy())
            predictions.append(predictions_batch)
        predictions = np.concatenate(predictions, axis=0)
        self.save_predictions(predictions, output_csv, study_ids, series_ids)
        return study_ids, series_ids, predictions

def main():
    model_path = constants.DENSENET_MODEL
    eval_output_csv = 'data_visual_outputs/evaluation_predictions.csv'
    pred_val_output_csv = 'data_visual_outputs/predictions_val.csv'
    pred_test_output_csv = 'data_visual_outputs/predictions_test.csv'
    pred_test_images_output_csv = 'data_visual_outputs/predictions_test_images.csv'
    binary_eval_output_csv = 'data_visual_outputs/binary_evaluation_predictions.csv'
    aggregated_eval_output_csv = 'data_visual_outputs/aggregated_binary_evaluation_predictions.csv'
    binary_val_output_csv = 'data_visual_outputs/binary_predictions_val.csv'
    aggregated_val_output_csv = 'data_visual_outputs/aggregated_binary_predictions_val.csv'
    binary_test_output_csv = 'data_visual_outputs/binary_predictions_test.csv'
    aggregated_test_output_csv = 'data_visual_outputs/aggregated_binary_predictions_test.csv'
    threshold = constants.DISEASE_THRESHOLD

    mode = 'val'
    predictor = VisionModelPredictor(model_path, mode=mode)
    
    val_dataset, _ = predictor.dataset.load_data('val')
    print("Evaluating on validation dataset")
    loss, accuracy = predictor.evaluate(val_dataset)

    evaluation_results = {"loss": loss, "accuracy": accuracy}
    evaluation_df = pd.DataFrame([evaluation_results])
    evaluation_df.to_csv(eval_output_csv, index=False)
    print(f"Evaluation results saved to {eval_output_csv}")

    # Prediction on validation dataset
    print("Running predictions on validation dataset")
    study_ids, series_ids, predictions = predictor.predict(val_dataset, pred_val_output_csv)
    predictor.convert_to_binary_and_aggregate(predictions, study_ids, series_ids, threshold, binary_val_output_csv, aggregated_val_output_csv)

    # Prediction on test dataset (using train data split)
    # test_dataset, _ = predictor.dataset.load_data('test').take(1)
    test_dataset, _ = predictor.dataset.load_data('test')  # Unpack the tuple
    test_dataset = test_dataset.take(1)  # Apply .take(5) on the dataset
    print("Running predictions on test dataset")
    study_ids, series_ids, predictions = predictor.predict(test_dataset, pred_test_output_csv)
    predictor.convert_to_binary_and_aggregate(predictions, study_ids, series_ids, threshold, binary_test_output_csv, aggregated_test_output_csv)

    # Prediction on test images dataset
    print("Running predictions on test images dataset")
    predictor.predict2(pred_test_images_output_csv)

if __name__ == "__main__":
    main()
