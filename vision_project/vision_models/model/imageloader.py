import pandas as pd
import numpy as np
import pydicom
import cv2
import tensorflow as tf
from tensorflow.data import Dataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class ImageLoader:
    def __init__(
        self, image_dir, label_coordinates_csv, labels_csv, roi_size, batch_size
    ):
        self.image_dir = image_dir
        self.label_coordinates_csv = label_coordinates_csv
        self.labels_csv = labels_csv
        self.study_id_to_labels = self._load_labels()
        self.roi_size = roi_size
        self.batch_size = batch_size

    def _load_labels(self):
        labels_df = pd.read_csv(self.labels_csv)
        study_id_to_labels = {}
        for index, row in labels_df.iterrows():
            study_id = row["study_id"]
            labels = row[1:].values
            study_id_to_labels[study_id] = labels
        return study_id_to_labels

    def _read_dicom(self, file_path):
        dicom = pydicom.dcmread(file_path)
        image = dicom.pixel_array
        image = image / np.max(image)
        return image

    def _preprocess_image(self, study_id, series_id, instance_number, x, y):
        file_path = f"{self.image_dir}/{study_id}/{series_id}/{instance_number}.dcm"
        img = self._read_dicom(file_path)
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = tf.expand_dims(img, axis=-1)  # Add channel dimension
        img = tf.image.resize(img, self.roi_size)
        return img, x, y

    def _extract_patch(self, image, x, y, width, height):
        x = tf.round(x)
        y = tf.round(y)
        x1 = x - width // 2
        y1 = y - height // 2
        patch = image[y1 : y1 + height, x1 : x1 + width]
        return patch


    def _combine_labels(self, labels):
        return tf.reduce_sum(labels, axis=1)

    def feature_generator(self):
        label_coordinates_df = pd.read_csv(self.label_coordinates_csv)
        for index, row in label_coordinates_df.iterrows():
            study_id = row['study_id']
            series_id = row['series_id']
            instance_number = row['instance_number']
            x = row['x']
            y = row['y']
            img, _, _ = self._preprocess_image(study_id, series_id, instance_number, x, y)
            yield img


    # def label_generator(self):
    #     labels_df = pd.read_csv(self.labels_csv)
    #     labels_df = labels_df.drop(columns=['study_id'])
    #     label_map = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2, 'Unknown': 3}
    #     for index, row in labels_df.iterrows():
    #         label = row[1:].values
    #         label = [label_map[l] for l in label]  # Map each value to its integer label
    #         label = tf.one_hot(label, depth=4)  # 4 classes: Normal/Mild, Moderate, Severe, Unknown
    #         label = tf.reduce_sum(label, axis=0)  # Use axis=0 instead of axis=1
    #         yield label.numpy().astype(np.float32)

    def label_generator(self):
        labels_df = pd.read_csv(self.label_coordinates_csv)
        labels_df['label'] = labels_df.apply(lambda row: row['condition'].replace(' ', '_') + '_' + row['level'].replace(' ', '_'), axis=1)
        self.label_list = labels_df['label'].unique().tolist()
        for index, row in labels_df.iterrows():
            study_id = row['study_id']
            label = self.label_list.index(row['label'])
            label_vector = [0.0] * 25
            label_vector[label] = 1.0
            yield np.array(label_vector, dtype=np.float32)
            

    def create_dataset(self):
        feature_dataset = tf.data.Dataset.from_generator(
            self.feature_generator,
            output_signature=tf.TensorSpec(shape=(self.roi_size[0], self.roi_size[1], 1), dtype=tf.float32)
        )

        label_dataset = tf.data.Dataset.from_generator(
            self.label_generator,
            output_signature=tf.TensorSpec(shape=(25,), dtype=tf.float32)
        )

        dataset = tf.data.Dataset.zip((feature_dataset, label_dataset))
        return dataset

    def _add_random_value(self, x, y):
        """Add a random value to each dataset element for splitting."""
        return x, y, tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)

    def _remove_random_value(self, x, y, _):
        """Remove the random value from dataset elements."""
        return x, y

    def _filter_dataset(self, dataset, threshold, is_training=True):
        """Filter dataset based on random value."""
        if is_training:
            return dataset.filter(lambda _, __, z: z >= threshold)
        else:
            return dataset.filter(lambda _, __, z: z < threshold)

    def _split_dataset(self, dataset, val_split=0.2):
        """Split a dataset into training and validation sets."""
        dataset_with_random = dataset.map(self._add_random_value)

        train_dataset = self._filter_dataset(
            dataset_with_random, val_split, is_training=True
        )
        val_dataset = self._filter_dataset(
            dataset_with_random, val_split, is_training=False
        )

        train_dataset = train_dataset.map(self._remove_random_value)
        val_dataset = val_dataset.map(self._remove_random_value)

        return train_dataset, val_dataset

    def load_data(self):
        # Create the dataset
        dataset = self.create_dataset()
        print("Splitting dataset")
        train_dataset, val_dataset = self._split_dataset(dataset)

        if self.batch_size:
            train_dataset = train_dataset.batch(self.batch_size)
            val_dataset = val_dataset.batch(self.batch_size)

        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset


# image_loader = ImageLoader(label_coordinates_csv='label_coordinates.csv', labels_csv='labels.csv',
#                               image_dir='images', roi_size=(224, 224), batch_size=32)
# train_dataset, val_dataset, test_dataset = image_loader.load_data()
