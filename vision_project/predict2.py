import os
import tensorflow as tf
from vision_models.utils import VisionUtils
from vision_models.imageloader2 import ImageLoader
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
        
        if mode not in ['val', 'test', 'predict']:
            raise ValueError("Invalid mode. Mode should be one of ['val', 'test', 'predict']")
        
        if mode in ['val', 'test']:
            self.image_loader = ImageLoader(
                image_dir=constants.TRAIN_DATA_PATH,
                label_coordinates_csv=constants.TRAIN_LABEL_CORD_PATH,
                labels_csv=constants.TRAIN_LABEL_PATH,
                roi_size=(224, 224),
                batch_size=self.batch_size,
                mode=mode
            )
        elif mode == 'predict':
            self.image_loader = ImageLoader(
                image_dir=constants.TEST_DATA_PATH,
                label_coordinates_csv=None,
                labels_csv=None,
                roi_size=(224, 224),
                batch_size=self.batch_size,
                mode='predict'
            )
        
        self.input_shape = (self.batch_size, 192, 224, 224, 3)
        self.num_classes = 25
        self.model = self.load_model(model_path)
        self.human_readable_labels = self.image_loader.get_human_readable_labels()
        self.mode = mode

    def load_model(self, model_path):
        model = DenseNetVisionModel(num_classes=self.num_classes, input_shape=self.input_shape, weights=None)
        model.load_weights(model_path)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def prepare_data(self, split):
        if split in ['val', 'test']:
            return self.image_loader.load_data(split)
        else:
            return self.image_loader.load_test_data(constants.TEST_DATA_PATH)

    def evaluate(self, dataset):
        results = self.model.evaluate(dataset)
        loss, accuracy = results[0], results[1]
        return loss, accuracy

    def save_predictions(self, predictions, output_csv, study_ids, series_ids):
        # predictions_df = pd.DataFrame(predictions, columns=[f'class_{i}' for i in range(predictions.shape[1])])
        predictions_df = pd.DataFrame(predictions, columns=self.human_readable_labels)
        predictions_df.insert(0, 'study_id', study_ids)
        predictions_df.insert(0, 'series_id', series_ids)
        predictions_df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")

    def convert_to_binary_and_aggregate(self, predictions, study_ids, series_ids, threshold, binary_output_csv, aggregated_output_csv):
        binary_predictions = (predictions > threshold).astype(int)
        binary_predictions_df = pd.DataFrame(binary_predictions, columns=self.human_readable_labels)
        binary_predictions_df.insert(0, 'study_id', study_ids)
        binary_predictions_df.insert(1, 'series_id', series_ids)

        binary_predictions_df.to_csv(binary_output_csv, index=False)
        print(f"Binary predictions saved to {binary_output_csv}")

        aggregated_predictions_df = binary_predictions_df.groupby('study_id').max().drop(columns = {'series_id'}).reset_index()
        aggregated_predictions_df.to_csv(aggregated_output_csv, index=False)
        print(f"Aggregated binary predictions saved to {aggregated_output_csv}")

    def predict(self, dataset, output_csv):
        y_pred = []
        study_ids = []
        series_ids = []
        predictions = []

        metadata_gen = self.image_loader.create_metadata()

        for images, labels in dataset:
            predictions_batch = self.model.predict(images)
            y_pred.extend(np.argmax(predictions_batch, axis=1))
            
            # Retrieve study_id and series_id from metadata generator
            for _ in range(len(images)):
                study_id, series_id = next(metadata_gen)
                study_ids.append(study_id)
                series_ids.append(series_id)
                
            predictions.append(predictions_batch)

        predictions = np.concatenate(predictions, axis=0)
        self.save_predictions(predictions, output_csv, study_ids, series_ids)
        return study_ids, series_ids, predictions



    def predict2(self, output_csv):
        test_images_dir = constants.TEST_DATA_PATH
        self.image_loader.image_dir = test_images_dir  # Set the image loader to use the test data path
        test_study_ids = os.listdir(test_images_dir)
        predictions = []
        study_ids = []
        series_ids = []
        counter = 0

        for study_id in test_study_ids:
            study_dir = os.path.join(test_images_dir, study_id)
            for series_id in os.listdir(study_dir):
                series_dir = os.path.join(study_dir, series_id)
                images = sorted([os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith(".dcm")])
                if not images:
                    continue
                img_tensor = self.image_loader._preprocess_image_test(study_id, series_id)
                prediction = self.model.predict(np.expand_dims(img_tensor, axis=0))
                predictions.append(prediction)
                study_ids.append(study_id)
                series_ids.append(series_id)
                counter += 1

                if counter % 10 == 0:
                    temp_output_csv = f'{output_csv.split(".")[0]}_part_{counter // 10}.csv'
                    self.save_predictions(np.concatenate(predictions, axis=0), temp_output_csv, study_ids, series_ids)
                    predictions = []
                    study_ids = []
                    series_ids = []

        if predictions:
            self.save_predictions(np.concatenate(predictions, axis=0), output_csv, study_ids, series_ids)



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

    # Mode can be 'val', 'test', or 'predict'
    mode = 'val'

    predictor = VisionModelPredictor(model_path, mode=mode)

    # Evaluation on validation dataset
    val_dataset = predictor.prepare_data('val').take(5)
    print("Evaluating on validation dataset")
    loss, accuracy = predictor.evaluate(val_dataset)

    # Save evaluation results to CSV
    evaluation_results = {
        "loss": loss,
        "accuracy": accuracy
    }
    evaluation_df = pd.DataFrame([evaluation_results])
    evaluation_df.to_csv(eval_output_csv, index=False)
    print(f"Evaluation results saved to {eval_output_csv}")

    # # Prediction on validation dataset
    # print("Running predictions on validation dataset")
    # study_ids, series_ids, predictions = predictor.predict(val_dataset)
    # predictor.save_predictions(predictions, pred_val_output_csv, study_ids, series_ids)
    # predictor.convert_to_binary_and_aggregate(predictions, study_ids, series_ids, threshold, binary_val_output_csv, aggregated_val_output_csv)

    # # Prediction on test dataset (using train data split)
    # test_dataset = predictor.prepare_data('test').take(2)
    # print("Running predictions on test dataset")
    # study_ids, series_ids, predictions = predictor.predict(test_dataset)
    # predictor.save_predictions(predictions, pred_test_output_csv, study_ids, series_ids)
    # predictor.convert_to_binary_and_aggregate(predictions, study_ids, series_ids, threshold, binary_test_output_csv, aggregated_test_output_csv)

    print("Running predictions on validation dataset")
    study_ids, series_ids, predictions = predictor.predict(val_dataset, pred_val_output_csv)
    predictor.convert_to_binary_and_aggregate(predictions, study_ids, series_ids, threshold, binary_val_output_csv, aggregated_val_output_csv)

    test_dataset = predictor.prepare_data('test').take(5)
    print("Running predictions on test dataset")
    study_ids, series_ids, predictions = predictor.predict(test_dataset, pred_test_output_csv)
    predictor.convert_to_binary_and_aggregate(predictions, study_ids, series_ids, threshold, binary_test_output_csv, aggregated_test_output_csv)

    # Prediction on test images dataset
    print("Running predictions on test images dataset")
    predictor.predict2(pred_test_images_output_csv)

if __name__ == "__main__":
    main()