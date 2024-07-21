import sys
print(sys.executable)

import numpy as np
print(f"Numpy version: {np.__version__}")
import pandas as pd
import os

import tensorflow as tf
print(f"Tensorflow version: {tf.__version__}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pydicom

from vision_models import constants

##################################################################
# This class creates a training and test tensorflow dataset object
# for the vision models.
#
##################################################################


class VisionDataset:
    """Dataset class for vision models"""

    def __init__(self):
        """ Creates the VisionDataset """
        self.train_dataset_path = constants.TRAIN_DATA_PATH
        self.train_dataset_path = constants.TEST_DATA_PATH
        self.batch_size = constants.BATCH_SIZE
        self.image_size = (constants.IMAGE_SIZE_HEIGHT*constants.IMAGE_SIZE_WIDTH)
        self.random_seed = constants.RANDOM_SEED
    
    def _read_dicom(file_path):
        dicom = pydicom.dcmread(file_path)
        image = dicom.pixel_array
        # Normalize the pixel values (0 to 1)
        image = image / np.max(image)
        return image

    def _preprocess_image(image, target_size=(224, 224)):
        # Resize image
        image = tf.image.resize(image, target_size)
        image = tf.expand_dims(image, axis=-1)  # Add channel dimension if necessary
        return image

    def _normalize_image(self, image):
        """Normalize image"""
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        return image

    def _load_data_from_csv(self, csv_path, label_coords_path, images_dir):
        """Load images and labels from local file system and associate them."""
        df = pd.read_csv(csv_path)
        coord_df = pd.read_csv(label_coords_path)
        images = []
        labels = []

        for index, row in df.iterrows():
            study_id = str(row["study_id"])
            label_values = row[1:].dropna().values.tolist()

            for condition_level, severity in zip(df.columns[1:], label_values):
                condition, level = condition_level.rsplit("_", 1)
                matching_coords = coord_df[
                    (coord_df["study_id"] == study_id)
                    & (coord_df["condition"] == condition)
                    & (coord_df["level"] == level)
                ]

                for _, coord_row in matching_coords.iterrows():
                    series_id = str(coord_row["series_id"])
                    instance_number = str(coord_row["instance_number"])
                    image_path = os.path.join(
                        images_dir, study_id, series_id, f"{instance_number}.dcm"
                    )

                    if os.path.exists(image_path):
                        image = self._read_dicom(image_path)
                        image = self._preprocess_image(image, self.image_size)
                        images.append(image)
                        labels.append(f"{condition_level}_{severity}")

        return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)

    def _create_tf_dataset(self, images, labels, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(images))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def _train_val_split_dataset(self, images, labels, split_ratio=0.2):
        """split dataset into training and validation sets"""
        images_train, images_val, labels_train, labels_val = train_test_split(
            images, labels, test_size=split_ratio, random_state=self.random_seed
        )
        return images_train, images_val, labels_train, labels_val

    def create_train_val_dataset(
        self, batch_size: int = 32) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """create training and validation datasets"""
        images, labels = self._load_data_from_csv(
            self.train_dataset_path, constants.LABEL_COORDS_PATH, constants.IMAGES_DIR
        )
        
        if batch_size:
            self.batch_size = batch_size

        # Encode labels using scikit-learn LabelEncoder
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        # Split dataset into training and validation sets
        images_train, images_val, labels_train, labels_val = (
            self._train_val_split_dataset(images, labels)
        )

        train_dataset = self._create_tf_dataset(
            images_train, labels_train, shuffle=True
        )
        val_dataset = self._create_tf_datasets(images_val, labels_val)

        return train_dataset, val_dataset

    def create_test_dataset(self):
        pass
    
    def take_n_from_dataset(self, dataset, n):
        itr = next(iter(dataset.take(n)))
        for image, label in itr:
            return image, label