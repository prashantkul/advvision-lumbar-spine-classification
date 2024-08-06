import os
import tensorflow as tf
from vision_models.utils import VisionUtils
from vision_models.imageloader import ImageLoader
from vision_models.densenetmodel import DenseNetVisionModel
import vision_models.constants as constants
import pandas as pd
import numpy as np

class VisionModelPredictor:
    def __init__(self, model_path):
        self.vutil = VisionUtils()
        self.batch_size = constants.BATCH_SIZE
        self.val_image_loader = ImageLoader(
            image_dir=constants.TRAIN_DATA_PATH,
            label_coordinates_csv=constants.TRAIN_LABEL_CORD_PATH,
            labels_csv=constants.TRAIN_LABEL_PATH,
            roi_size=(224, 224),
            batch_size=self.batch_size
        )
        self.test_image_loader = ImageLoader(
            image_dir=constants.TEST_DATA_PATH,
            label_coordinates_csv=None,
            labels_csv=None,
            roi_size=(224, 224),
            batch_size=self.batch_size
        )
        self.input_shape = (self.batch_size, 192, 224, 224, 3)
        self.num_classes = 25
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = DenseNetVisionModel(num_classes=self.num_classes, input_shape=self.input_shape, weights=None)
        model.load_weights(model_path)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def prepare_data(self, split):
        if split == 'val':
            return self.val_image_loader.load_data(split)
        else:
            return self.test_image_loader.load_data(split)

    def evaluate(self, val_dataset):
        results = self.model.evaluate(val_dataset)
        print(f"Evaluation results: {results}")
        return results

    def run_predictions(self, test_dataset):
        predictions = self.model.predict(test_dataset)
        return predictions

    def save_predictions(self, predictions, output_csv, study_ids):
        predictions_df = pd.DataFrame(predictions, columns=[f'class_{i}' for i in range(predictions.shape[1])])
        predictions_df.insert(0, 'study_id', study_ids)
        predictions_df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")

    def predict(self, output_csv):
        test_images_dir = constants.TEST_DATA_PATH
        test_study_ids = os.listdir(test_images_dir)
        predictions = []
        study_ids = []

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

        predictions = np.concatenate(predictions, axis=0)
        self.save_predictions(predictions, output_csv, study_ids)

def main():
    model_path = constants.DENSENET_MODEL
    output_csv = 'predictions.csv'
    predictor = VisionModelPredictor(model_path)

    # Evaluation
    val_dataset = predictor.prepare_data('val')  # Using 'val' split for evaluation
    predictor.evaluate(val_dataset)

    # Prediction
    predictor.predict(output_csv)

if __name__ == "__main__":
    main()