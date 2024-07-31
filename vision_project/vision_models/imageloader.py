import pandas as pd
import numpy as np
import pydicom
import cv2
import tensorflow as tf
from tensorflow.data import Dataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import vision_models.constants as constants

class ImageLoader:
    """
    A class for loading and preprocessing medical image data for machine learning tasks.
    
    This class handles loading DICOM images, extracting relevant patches, and creating
    TensorFlow datasets for training and validation.
    """

    def __init__(
        self, image_dir, label_coordinates_csv, labels_csv, roi_size, batch_size
    ):
        """
        Initialize the ImageLoader.

        Args:
            image_dir (str): Directory containing the DICOM images.
            label_coordinates_csv (str): Path to CSV file with label coordinates.
            labels_csv (str): Path to CSV file with study labels.
            roi_size (tuple): Size of the region of interest (height, width).
            batch_size (int): Batch size for the dataset.
        """
        self.image_dir = image_dir
        self.label_coordinates_csv = label_coordinates_csv
        self.labels_csv = labels_csv
        self.study_id_to_labels = self._load_labels()
        self.roi_size = roi_size
        self.batch_size = batch_size

    def _load_labels(self):
        """
        Load labels from CSV file into a dictionary.

        Returns:
            dict: A dictionary mapping study_id to labels.
        """
        labels_df = pd.read_csv(self.labels_csv)
        study_id_to_labels = {}
        for index, row in labels_df.iterrows():
            study_id = row["study_id"]
            labels = row[1:].values
            study_id_to_labels[study_id] = labels
        return study_id_to_labels

    def _read_dicom(self, file_path):
        """
        Read a DICOM file and normalize the pixel array.

        Args:
            file_path (str): Path to the DICOM file.

        Returns:
            np.array: Normalized pixel array of the DICOM image.
        """
        dicom = pydicom.dcmread(file_path)
        image = dicom.pixel_array
        image = image / np.max(image)
        return image

    def _preprocess_image(self, study_id, series_id, instance_number, x, y):
        """
        Preprocess a single image.

        Args:
            study_id (str): Study ID.
            series_id (str): Series ID.
            instance_number (str): Instance number.
            x (float): X-coordinate.
            y (float): Y-coordinate.

        Returns:
            tuple: Preprocessed image tensor, x, and y coordinates.
        """
        file_path = f"{self.image_dir}/{study_id}/{series_id}/{instance_number}.dcm"
        img = self._read_dicom(file_path)
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = tf.expand_dims(img, axis=-1)  # Add channel dimension
        img = tf.image.resize(img, self.roi_size)
        return img, x, y

    def _extract_patch(self, image, x, y, width, height):
        """
        Extract a patch from the image centered at (x, y).

        Args:
            image (tf.Tensor): Input image.
            x (float): X-coordinate of the center.
            y (float): Y-coordinate of the center.
            width (int): Width of the patch.
            height (int): Height of the patch.

        Returns:
            tf.Tensor: Extracted patch.
        """
        x = tf.round(x)
        y = tf.round(y)
        x1 = x - width // 2
        y1 = y - height // 2
        patch = image[y1 : y1 + height, x1 : x1 + width]
        return patch

    def _combine_labels(self, labels):
        """
        Combine multiple labels into a single label.

        Args:
            labels (tf.Tensor): Input labels.

        Returns:
            tf.Tensor: Combined label.
        """
        return tf.reduce_sum(labels, axis=1)

    def feature_generator(self):
        """
        Generator function for image features.

        Yields:
            tf.Tensor: Preprocessed image tensor.
        """
        label_coordinates_df = pd.read_csv(self.label_coordinates_csv)
        for index, row in label_coordinates_df.iterrows():
            study_id = row['study_id']
            series_id = row['series_id']
            instance_number = row['instance_number']
            x = row['x']
            y = row['y']
            img, _, _ = self._preprocess_image(study_id, series_id, instance_number, x, y)
            
            # convert to RGB since many pretrained model use RGB
            img = tf.image.grayscale_to_rgb(img) 
            yield img

    def label_generator(self):
        """
        Generator function for labels.

        Yields:
            np.array: One-hot encoded label vector.
        """
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
        """
        Create a TensorFlow dataset from feature and label generators.

        Returns:
            tf.data.Dataset: Combined dataset of features and labels.
        """
        feature_dataset = tf.data.Dataset.from_generator(
            self.feature_generator,
            output_signature=tf.TensorSpec(shape=(self.roi_size[0], self.roi_size[1], 3), dtype=tf.float32)
        )

        label_dataset = tf.data.Dataset.from_generator(
            self.label_generator,
            output_signature=tf.TensorSpec(shape=(25,), dtype=tf.float32)
        )

        dataset = tf.data.Dataset.zip((feature_dataset, label_dataset))
        return dataset

    def _add_random_value(self, x, y):
        """
        Add a random value to each dataset element for splitting.

        Args:
            x (tf.Tensor): Feature tensor.
            y (tf.Tensor): Label tensor.

        Returns:
            tuple: Feature tensor, label tensor, and random value.
        """
        return x, y, tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)

    def _remove_random_value(self, x, y, _):
        """
        Remove the random value from dataset elements.

        Args:
            x (tf.Tensor): Feature tensor.
            y (tf.Tensor): Label tensor.
            _ (tf.Tensor): Random value to be removed.

        Returns:
            tuple: Feature tensor and label tensor.
        """
        return x, y

    def _filter_dataset(self, dataset, threshold, is_training=True):
        """
        Filter dataset based on random value.

        Args:
            dataset (tf.data.Dataset): Input dataset.
            threshold (float): Threshold for splitting.
            is_training (bool): Flag to indicate if it's the training set.

        Returns:
            tf.data.Dataset: Filtered dataset.
        """
        if is_training:
            return dataset.filter(lambda _, __, z: z >= threshold)
        else:
            return dataset.filter(lambda _, __, z: z < threshold)

    def _split_dataset(self, dataset, val_split=0.2):
        """
        Split a dataset into training and validation sets.

        Args:
            dataset (tf.data.Dataset): Input dataset.
            val_split (float): Fraction of data to use for validation.

        Returns:
            tuple: Training dataset and validation dataset.
        """
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
        """
        Load and prepare the data for training and validation.

        Returns:
            tuple: Training dataset and validation dataset.
            Each dataset will have image and label tensors.
        """
        # Create the dataset
        dataset = self.create_dataset()
        print("Splitting dataset")
        train_dataset, val_dataset = self._split_dataset(dataset)

        if self.batch_size:
            train_dataset = train_dataset.batch(self.batch_size)
            val_dataset = val_dataset.batch(self.batch_size)
        else:
            train_dataset = train_dataset.batch(constants.BATCH_SIZE)
            val_dataset = val_dataset.batch(constants.BATCH_SIZE)

        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        print("Dataset created, you can now iterate over the dataset")

        return train_dataset, val_dataset

# Usage example (commented out):
# image_loader = ImageLoader(label_coordinates_csv='label_coordinates.csv', labels_csv='labels.csv',
#                               image_dir='images', roi_size=(224, 224), batch_size=32)
# train_dataset, val_dataset = image_loader.load_data()
# for img, labels in train_dataset.take(1):
#     print(img.shape)
#     print(labels.shape)