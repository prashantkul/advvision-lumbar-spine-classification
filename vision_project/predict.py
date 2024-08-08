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
        # model.compile(optimizer='adam',  # Using Adam optimizer with default settings
        #     loss="binary_crossentropy",
        #     metrics=["binary_accuracy",                 
        #              tf.keras.metrics.AUC(multi_label=True, num_labels=self.model.num_classes),
        #             ])
        # model.compile_model()
        return model

    def prepare_data(self, split):
        if split in ['val', 'test']:
            return self.image_loader.load_data(split)
        else:
            return self.image_loader.load_test_data(constants.TEST_DATA_PATH)
        
    # def save_evaluation_results(self, evaluation_results, output_csv):
        
    #     evaluation_df = evaluation_results
    #     print(f"Evaluation results: {evaluation_df}")

    #     evaluation_df2 = pd.DataFrame([evaluation_results], columns=['loss', 'accuracy'])
    #     evaluation_df2.to_csv(output_csv, index=False)
    #     print(f"Evaluation results saved to {output_csv}")

    # def evaluate(self, val_dataset, output_csv):
    #     # Count the number of records in the validation dataset
    #     # val_count = sum(1 for _ in val_dataset)
    #     # print(f"Number of records in the validation dataset: {val_count}")

    #     results = self.model.evaluate(val_dataset)
    #     print(f"Evaluation results: {results}")
    #     self.save_evaluation_results(results, output_csv)
    #     return results

    def evaluate_and_predict(self, dataset, output_csv=None, is_evaluation=False):
        if is_evaluation:
            evaluation_metrics = self.model.evaluate(dataset)
            loss, accuracy = evaluation_metrics[0], evaluation_metrics[1]
        else:
            loss, accuracy = None, None

        study_ids = []
        series_ids = []
        predictions = []

        for images, labels in dataset:
            predictions_batch = self.model.predict(images)
            study_ids.extend(labels['study_id'])  # Assuming study_id is part of the labels
            series_ids.extend(labels['series_id'])  # Assuming series_id is part of the labels
            predictions.append(predictions_batch)

        predictions = np.concatenate(predictions, axis=0)
        
        if output_csv:
            self.save_predictions(predictions, output_csv, study_ids, series_ids)

        return loss, accuracy, study_ids, series_ids, predictions

    # def evaluate(self, val_dataset):
    #     # Initial evaluation to get loss and accuracy
    #     results = self.model.evaluate(val_dataset)
    #     loss, accuracy = results[0], results[1]

    #     y_true = []
    #     y_pred = []
    #     study_ids = []
    #     predictions = []

    #     for images, labels in val_dataset:
    #         predictions_batch = self.model.predict(images)
    #         y_true.extend(np.argmax(labels.numpy(), axis=1))
    #         y_pred.extend(np.argmax(predictions_batch, axis=1))
    #         study_ids.extend(labels['study_id'])  # Assuming study_id is part of the labels

    #         # Collect predictions for further processing
    #         predictions.append(predictions_batch)

    #     # Flatten the list of predictions
    #     predictions = np.concatenate(predictions, axis=0)

    #     # Convert probabilities to binary values and aggregate by study_id
    #     threshold = constants.DISEASE_THRESHOLD
    #     binary_predictions = (predictions > threshold).astype(int)
    #     binary_predictions_df = pd.DataFrame(binary_predictions, columns=self.human_readable_labels)
    #     binary_predictions_df.insert(0, 'study_id', study_ids)

    #     # Save the binary predictions
    #     binary_output_csv = 'data_visual_outputs/binary_evaluation_predictions.csv'
    #     binary_predictions_df.to_csv(binary_output_csv, index=False)
    #     print(f"Binary predictions saved to {binary_output_csv}")

    #     # Aggregate predictions by study_id and take the max for each condition
    #     aggregated_predictions_df = binary_predictions_df.groupby('study_id').max().reset_index()
    #     aggregated_output_csv = 'data_visual_outputs/aggregated_binary_evaluation_predictions.csv'
    #     aggregated_predictions_df.to_csv(aggregated_output_csv, index=False)
    #     print(f"Aggregated binary predictions saved to {aggregated_output_csv}")

    #     # Compute additional metrics
    #     report = classification_report(y_true, y_pred, output_dict=True)
    #     confusion = confusion_matrix(y_true, y_pred)
    #     roc_auc = roc_auc_score(tf.keras.utils.to_categorical(y_true, num_classes=self.num_classes), 
    #                             tf.keras.utils.to_categorical(y_pred, num_classes=self.num_classes), multi_class='ovr')

    #     # Print evaluation results
    #     print("Classification Report:")
    #     print(report)
    #     print("Confusion Matrix:")
    #     print(confusion)
    #     print("ROC AUC Score:")
    #     print(roc_auc)

    #     # Save evaluation results to CSV
    #     evaluation_results = {
    #         "loss": loss,
    #         "accuracy": accuracy,
    #         "precision": report["weighted avg"]["precision"],
    #         "recall": report["weighted avg"]["recall"],
    #         "f1_score": report["weighted avg"]["f1-score"],
    #         "roc_auc": roc_auc
    #     }

    #     # Convert classification report to DataFrame
    #     report_df = pd.DataFrame(report).transpose()
    #     report_df.to_csv("data_visual_outputs/report.csv", index=True)
        
    #     evaluation_df = pd.DataFrame([evaluation_results])
    #     evaluation_df.to_csv("data_visual_outputs/evaluation_results.csv", index=False)
    #     print("Evaluation results saved to evaluation_results.csv")

    #     # Save confusion matrix as an image
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    #     plt.xlabel("Predicted")
    #     plt.ylabel("Actual")
    #     plt.title("Confusion Matrix")
    #     plt.savefig("data_visual_outputs/confusion_matrix.png")
    #     plt.close()
    #     print("Confusion matrix saved to confusion_matrix.png")

    #     return report, confusion, roc_auc
        # return loss, accuracy


    # def evaluate(self, val_dataset):
    #     y_true = []
    #     y_pred = []

    #     for images, labels in val_dataset:
    #         predictions = self.model.predict(images)
    #         y_true.extend(np.argmax(labels.numpy(), axis=1))
    #         y_pred.extend(np.argmax(predictions, axis=1))

    #     # Compute additional metrics
    #     report = classification_report(y_true, y_pred, output_dict=True)
    #     confusion = confusion_matrix(y_true, y_pred)
    #     roc_auc = roc_auc_score(tf.keras.utils.to_categorical(y_true, num_classes=self.num_classes), 
    #                             tf.keras.utils.to_categorical(y_pred, num_classes=self.num_classes), multi_class='ovr')

    #     # Print evaluation results
    #     print("Classification Report:")
    #     print(report)
    #     print("Confusion Matrix:")
    #     print(confusion)
    #     print("ROC AUC Score:")
    #     print(roc_auc)

    #     # Save evaluation results to CSV
    #     evaluation_results = {
    #         "loss": report["weighted avg"]["precision"],  # Example, replace with actual loss
    #         "accuracy": report["accuracy"],
    #         "precision": report["weighted avg"]["precision"],
    #         "recall": report["weighted avg"]["recall"],
    #         "f1_score": report["weighted avg"]["f1-score"],
    #         "roc_auc": roc_auc
    #     }

    #     # Convert classification report to DataFrame
    #     report_df = pd.DataFrame(report).transpose()
    #     report_df.to_csv("data_visual_outputs/report.csv", index=True)

    #     evaluation_df = pd.DataFrame([evaluation_results])
    #     evaluation_df.to_csv("data_visual_outputs/evaluation_results.csv", index=False)
    #     print("Evaluation results saved to evaluation_results.csv")

    #     # Save confusion matrix as an image
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    #     plt.xlabel("Predicted")
    #     plt.ylabel("Actual")
    #     plt.title("Confusion Matrix")
    #     plt.savefig("data_visual_outputs/confusion_matrix.png")
    #     plt.close()
    #     print("Confusion matrix saved to confusion_matrix.png")

    #     return report, confusion, roc_auc

    # def evaluate(self, val_dataset):
    #     y_true = []
    #     y_pred = []

    #     for images, labels in val_dataset:
    #         predictions = self.model.predict(images)
    #         y_true.extend(np.argmax(labels.numpy(), axis=1))
    #         y_pred.extend(np.argmax(predictions, axis=1))

    #     # Compute additional metrics
    #     report = classification_report(y_true, y_pred, output_dict=True)
    #     confusion = confusion_matrix(y_true, y_pred)
    #     roc_auc = roc_auc_score(tf.keras.utils.to_categorical(y_true, num_classes=self.num_classes), 
    #                             tf.keras.utils.to_categorical(y_pred, num_classes=self.num_classes), multi_class='ovr')

    #     # Print evaluation results
    #     print("Classification Report:")
    #     print(report)
    #     print("Confusion Matrix:")
    #     print(confusion)
    #     print("ROC AUC Score:")
    #     print(roc_auc)

    #     # Save evaluation results to CSV
    #     evaluation_results = {
    #         "loss": report["weighted avg"]["precision"],  # Example, replace with actual loss
    #         "accuracy": report["accuracy"],
    #         "precision": report["weighted avg"]["precision"],
    #         "recall": report["weighted avg"]["recall"],
    #         "f1_score": report["weighted avg"]["f1-score"],
    #         "roc_auc": roc_auc
    #     }

    #     evaluation_df = pd.DataFrame([evaluation_results])
    #     evaluation_df.to_csv("evaluation_results.csv", index=False)
    #     print("Evaluation results saved to evaluation_results.csv")

    #     # Save confusion matrix as an image
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    #     plt.xlabel("Predicted")
    #     plt.ylabel("Actual")
    #     plt.title("Confusion Matrix")
    #     plt.savefig("confusion_matrix.png")
    #     plt.close()
    #     print("Confusion matrix saved to confusion_matrix.png")

    #     return report, confusion, roc_auc

    # def evaluate(self, val_dataset):
    #     y_true = []
    #     y_pred = []
    #     losses = []
    #     accuracies = []

    #     for images, labels in val_dataset:
    #         results = self.model.evaluate(images, labels, verbose=0)
    #         loss, accuracy = results[0], results[1]
    #         predictions = self.model.predict(images)
    #         y_true.extend(np.argmax(labels.numpy(), axis=1))
    #         y_pred.extend(np.argmax(predictions, axis=1))
    #         losses.append(loss)
    #         accuracies.append(accuracy)

    #     # Compute additional metrics
    #     report = classification_report(y_true, y_pred, output_dict=True)
    #     confusion = confusion_matrix(y_true, y_pred)
    #     roc_auc = roc_auc_score(tf.keras.utils.to_categorical(y_true, num_classes=self.num_classes), 
    #                             tf.keras.utils.to_categorical(y_pred, num_classes=self.num_classes), multi_class='ovr')

    #     # Print evaluation results
    #     print("Classification Report:")
    #     print(report)
    #     print("Confusion Matrix:")
    #     print(confusion)
    #     print("ROC AUC Score:")
    #     print(roc_auc)

    #     # Save evaluation results to CSV
    #     evaluation_results = {
    #         "loss": np.mean(losses),
    #         "accuracy": np.mean(accuracies),
    #         "precision": report["weighted avg"]["precision"],
    #         "recall": report["weighted avg"]["recall"],
    #         "f1_score": report["weighted avg"]["f1-score"],
    #         "roc_auc": roc_auc
    #     }

    #     evaluation_df = pd.DataFrame([evaluation_results])
    #     evaluation_df.to_csv("evaluation_results.csv", index=False)
    #     print("Evaluation results saved to evaluation_results.csv")

    #     # Save confusion matrix as an image
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    #     plt.xlabel("Predicted")
    #     plt.ylabel("Actual")
    #     plt.title("Confusion Matrix")
    #     plt.savefig("confusion_matrix.png")
    #     plt.close()
    #     print("Confusion matrix saved to confusion_matrix.png")

    #     return report, confusion, roc_auc



    # def run_predictions(self, test_dataset):
    #     predictions = self.model.predict(test_dataset)
    #     return predictions

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

        aggregated_predictions_df = binary_predictions_df.groupby('study_id').max().reset_index()
        aggregated_predictions_df.to_csv(aggregated_output_csv, index=False)
        print(f"Aggregated binary predictions saved to {aggregated_output_csv}")


    def predict(self, data_loader, output_csv):
        data_images_dir = data_loader.image_dir
        data_study_ids = os.listdir(data_images_dir)
        predictions = []
        study_ids = []
        series_ids = []
        counter = 0

        for study_id in data_study_ids:
            study_dir = os.path.join(data_images_dir, study_id)
            for series_id in os.listdir(study_dir):
                series_dir = os.path.join(study_dir, series_id)
                images = sorted([os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith(".dcm")])
                if not images:
                    continue
                img_tensor = data_loader._preprocess_image(study_id, series_id)
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


    # def predict(self, output_csv):
    #     test_images_dir = constants.TEST_DATA_PATH
    #     test_study_ids = os.listdir(test_images_dir)
    #     predictions = []
    #     study_ids = []

    #     total_studies = len(test_study_ids)
    #     study_counter = 0

    #     for study_id in test_study_ids:
    #         study_counter += 1
    #         study_dir = os.path.join(test_images_dir, study_id)
    #         for series_id in os.listdir(study_dir):
    #             series_dir = os.path.join(study_dir, series_id)
    #             images = sorted([os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith(".dcm")])
    #             if not images:
    #                 continue
    #             img_tensor = self.test_image_loader._preprocess_image(study_id, series_id, x=0, y=0)  # Dummy x, y for prediction
    #             prediction = self.model.predict(np.expand_dims(img_tensor, axis=0))
    #             predictions.append(prediction)
    #             study_ids.append(study_id)

    #         # Print progress
    #         if study_counter % 10 == 0:  # Print progress every 10 studies
    #             print(f"Processed {study_counter}/{total_studies} studies ({study_counter/total_studies:.2%})")

    #     predictions = np.concatenate(predictions, axis=0)
    #     self.save_predictions(predictions, output_csv, study_ids)
    #     print(f"Total records processed: {len(study_ids)}")

    # def predict(self, output_csv):
    #     test_images_dir = constants.TEST_DATA_PATH
    #     test_study_ids = os.listdir(test_images_dir)
    #     predictions = []
    #     study_ids = []
    #     counter = 0

    #     for study_id in test_study_ids:
    #         study_dir = os.path.join(test_images_dir, study_id)
    #         for series_id in os.listdir(study_dir):
    #             series_dir = os.path.join(study_dir, series_id)
    #             images = sorted([os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith(".dcm")])
    #             if not images:
    #                 continue
    #             img_tensor = self.test_image_loader._preprocess_image(study_id, series_id, x=0, y=0)  # Dummy x, y for prediction
    #             prediction = self.model.predict(np.expand_dims(img_tensor, axis=0))
    #             predictions.append(prediction)
    #             study_ids.append(study_id)
    #             counter += 1

    #             if counter % 10 == 0:
    #                 temp_output_csv = f'{output_csv.split(".")[0]}_part_{counter // 10}.csv'
    #                 self.save_predictions(np.concatenate(predictions, axis=0), temp_output_csv, study_ids)
    #                 predictions = []
    #                 study_ids = []
    #                 series_ids = []

    #     # Save any remaining predictions
    #     if predictions:
    #         self.save_predictions(np.concatenate(predictions, axis=0), output_csv, study_ids)

    # def predict(self, output_csv):
    #     test_images_dir = constants.TEST_DATA_PATH
    #     test_study_ids = os.listdir(test_images_dir)
    #     predictions = []
    #     study_ids = []
    #     series_ids = []
    #     counter = 0

    #     for study_id in test_study_ids:
    #         study_dir = os.path.join(test_images_dir, study_id)
    #         for series_id in os.listdir(study_dir):
    #             series_dir = os.path.join(study_dir, series_id)
    #             images = sorted([os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith(".dcm")])
    #             if not images:
    #                 continue
    #             img_tensor = self.test_image_loader._preprocess_image(study_id, series_id, x=0, y=0)  # Dummy x, y for prediction
    #             prediction = self.model.predict(np.expand_dims(img_tensor, axis=0))
    #             predictions.append(prediction)
    #             study_ids.append(study_id)
    #             series_ids.append(series_id)
    #             counter += 1

    #             if counter % 10 == 0:
    #                 temp_output_csv = f'{output_csv.split(".")[0]}_part_{counter // 10}.csv'
    #                 self.save_predictions(np.concatenate(predictions, axis=0), temp_output_csv, study_ids, series_ids)
    #                 predictions = []
    #                 study_ids = []
    #                 series_ids = []

    #     # Save any remaining predictions
    #     if predictions:
    #         self.save_predictions(np.concatenate(predictions, axis=0), output_csv, study_ids, series_ids)

    def predict2(self, output_csv):
        test_images_dir = constants.TEST_DATA_PATH
        test_study_ids = os.listdir(test_images_dir)
        predictions = []
        study_ids = []
        series_ids = []
        counter = 0

        for study_id in test_study_ids:
            study_dir = os.path.join(test_images_dir, study_id)
            for series_id in os.listdir(study_dir):
                series_dir = os.path.join(series_dir, series_id)
                images = sorted([os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith(".dcm")])
                if not images:
                    continue
                # Skip Gaussian attention by not providing x and y coordinates
                img_tensor = self.test_image_loader._preprocess_image_test(study_id, series_id)
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



# def count_studies_series_images(image_dir, split=None, label_coordinates_csv=None):
#     study_count = 0
#     series_count = 0
#     image_count = 0

#     if split:
#         df = pd.read_csv(label_coordinates_csv)
#         if 'split' not in df.columns:
#             image_loader = ImageLoader(image_dir=image_dir, label_coordinates_csv=label_coordinates_csv, labels_csv=None, roi_size=(224, 224), batch_size=1)
#             df = image_loader._create_split(df)
#         df = df[df["split"] == split]
#         unique_studies = df["study_id"].unique()
#     else:
#         unique_studies = os.listdir(image_dir)

#     for study_id in unique_studies:
#         study_count += 1
#         study_dir = os.path.join(image_dir, str(study_id))  # Convert study_id to string
#         for series_id in os.listdir(study_dir):
#             series_count += 1
#             series_dir = os.path.join(study_dir, series_id)
#             images = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith(".dcm")]
#             image_count += len(images)

#     return study_count, series_count, image_count

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
    val_dataset = predictor.prepare_data('val')
    print("Evaluating on validation dataset")
    loss, accuracy, study_ids, series_ids, predictions = predictor.evaluate_and_predict(val_dataset, is_evaluation=True)

    # Save evaluation results to CSV
    evaluation_results = {
        "loss": loss,
        "accuracy": accuracy
    }
    evaluation_df = pd.DataFrame([evaluation_results])
    evaluation_df.to_csv(eval_output_csv, index=False)
    print(f"Evaluation results saved to {eval_output_csv}")

    # Convert validation predictions to binary and aggregate by study_id
    predictor.convert_to_binary_and_aggregate(predictions, study_ids, series_ids, threshold, binary_eval_output_csv, aggregated_eval_output_csv)

    # Prediction on validation dataset
    print("Running predictions on validation dataset")
    _, _, study_ids, series_ids, predictions = predictor.evaluate_and_predict(val_dataset, pred_val_output_csv)

    # Convert validation predictions to binary and aggregate by study_id
    predictor.convert_to_binary_and_aggregate(predictions, study_ids, series_ids, threshold, binary_val_output_csv, aggregated_val_output_csv)

    # Prediction on test dataset (using train data split)
    test_dataset = predictor.prepare_data('test')
    print("Running predictions on test dataset")
    _, _, study_ids, series_ids, predictions = predictor.evaluate_and_predict(test_dataset, pred_test_output_csv)

    # Convert test predictions to binary and aggregate by study_id
    predictor.convert_to_binary_and_aggregate(predictions, study_ids, series_ids, threshold, binary_test_output_csv, aggregated_test_output_csv)

    # Prediction on test images dataset
    print("Running predictions on test images dataset")
    predictor.predict2(pred_test_images_output_csv)

if __name__ == "__main__":
    main()