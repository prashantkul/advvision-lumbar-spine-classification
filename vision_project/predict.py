import os
import tensorflow as tf
from vision_models.utils import VisionUtils
from vision_models.imageloader import ImageLoader
from vision_models.densenetmodel import DenseNetVisionModel
import vision_models.constants as constants
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

class VisionModelPredictor:
    def __init__(self, model_path):
        self.vutil = VisionUtils()
        self.batch_size = constants.BATCH_SIZE
        self.val_image_loader = ImageLoader(
            image_dir=constants.TRAIN_DATA_PATH,
            label_coordinates_csv=constants.TRAIN_LABEL_CORD_PATH,
            labels_csv=constants.TRAIN_LABEL_PATH,
            roi_size=(constants.IMAGE_SIZE_HEIGHT, constants.IMAGE_SIZE_WIDTH),
            batch_size=self.batch_size
        )
        self.test_image_loader = ImageLoader(
            image_dir=constants.TEST_DATA_PATH,
            label_coordinates_csv=None,
            labels_csv=None,
            roi_size=(constants.IMAGE_SIZE_HEIGHT, constants.IMAGE_SIZE_WIDTH),
            batch_size=self.batch_size
        )
        self.input_shape = (self.batch_size, 192, constants.IMAGE_SIZE_HEIGHT, constants.IMAGE_SIZE_WIDTH, constants.IMAGE_CHANNELS)
        self.num_classes = 25
        self.model = self.load_model(model_path)
        self.human_readable_labels = self.val_image_loader.get_human_readable_labels()

    def load_model(self, model_path):
        model = DenseNetVisionModel(num_classes=self.num_classes, input_shape=self.input_shape, weights=None)
        model.load_weights(model_path)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
        return model

    def prepare_data(self, split):
        if split == 'val':
            return self.val_image_loader.load_data(split)
        else:
            return self.test_image_loader.load_test_data(constants.TEST_DATA_PATH)
        
    def save_evaluation_results(self, evaluation_results, output_csv):
        evaluation_df = pd.DataFrame([evaluation_results], columns=['loss', 'accuracy'])
        # evaluation_df.to_csv(output_csv, index=False)
        print(f"Evaluation results saved to {output_csv}")

    def evaluate(self, val_dataset):
        y_true = []
        y_pred = []

        for images, labels in val_dataset:
            predictions = self.model.predict(images)
            y_true.extend(labels.numpy())
            y_pred.extend(predictions)

        # Convert lists to numpy arrays for metric calculations
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Compute additional metrics
        report = classification_report(y_true, y_pred > 0.5, output_dict=True)
        confusion = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
        roc_auc = roc_auc_score(y_true, y_pred, average='macro')

        # Print evaluation results
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(confusion)
        print("ROC AUC Score:")
        print(roc_auc)

        # Save evaluation results to CSV
        evaluation_results = {
            "loss": report["weighted avg"]["precision"],  # Example, replace with actual loss
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
            "roc_auc": roc_auc
        }

        evaluation_df = pd.DataFrame([evaluation_results])
        evaluation_df.to_csv("evaluation_results.csv", index=False)
        print("Evaluation results saved to evaluation_results.csv")

        # Save confusion matrix as an image
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()
        print("Confusion matrix saved to confusion_matrix.png")

        return report, confusion, roc_auc

    def run_predictions(self, test_dataset):
        predictions = self.model.predict(test_dataset)
        return predictions

    def save_predictions(self, predictions, output_csv, study_ids, series_ids):
        predictions_df = pd.DataFrame(predictions, columns=self.human_readable_labels)
        predictions_df.insert(0, 'study_id', study_ids)
        predictions_df.insert(0, 'series_id', series_ids)
        predictions_df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")

    def predict(self, output_csv):
        test_images_dir = constants.TEST_DATA_PATH
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
                img_tensor = self.test_image_loader._preprocess_image(study_id, series_id, x=0, y=0)  # Dummy x, y for prediction
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

        # Save any remaining predictions
        if predictions:
            self.save_predictions(np.concatenate(predictions, axis=0), output_csv, study_ids, series_ids)

def count_studies_series_images(image_dir, split=None, label_coordinates_csv=None):
    study_count = 0
    series_count = 0
    image_count = 0

    if split:
        # df = pd.read_csv(label_coordinates_csv)
        if 'split' not in df.columns:
            image_loader = ImageLoader(image_dir=image_dir, label_coordinates_csv=label_coordinates_csv, labels_csv=None, roi_size=(constants.IMAGE_SIZE_HEIGHT, constants.IMAGE_SIZE_WIDTH), batch_size=constants.BATCH_SIZE)
            df = image_loader._create_split(df)
        df = df[df["split"] == split]
        unique_studies = df["study_id"].unique()
    else:
        unique_studies = os.listdir(image_dir)

    for study_id in unique_studies:
        study_count += 1
        study_dir = os.path.join(image_dir, str(study_id))  # Convert study_id to string
        for series_id in os.listdir(study_dir):
            series_count += 1
            series_dir = os.path.join(study_dir, series_id)
            images = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith(".dcm")]
            image_count += len(images)

    return study_count, series_count, image_count

def main():
    # Specify the path to your model weights
    model_path = '/opt/models/densenet_weights.h5'  # Update this to your actual model path
    output_csv = 'predictions.csv'
    evaluation_csv = 'evaluation_results.csv'

    predictor = VisionModelPredictor(model_path)

    # Evaluation
    val_dataset = predictor.prepare_data('val')  # Using 'val' split for evaluation
    print("Evaluating on validation dataset")
    report, confusion, roc_auc = predictor.evaluate(val_dataset)

    # Prediction
    print("Running predictions on test dataset")
    predictor.predict(output_csv)

if __name__ == "__main__":
    main()
