import os
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

    def _preprocess_image(self, study_id, series_id):
        """
        Preprocess all images in a series.

        Args:
            study_id (str): Study ID.
            series_id (str): Series ID.

        Yields:
            tuple: Preprocessed image tensor, study_id, series_id.
        """
        series_dir = f"{self.image_dir}/{study_id}/{series_id}"
        images = []
        for filename in os.listdir(series_dir):
            if filename.endswith(".dcm"):
                file_path = f"{series_dir}/{filename}"
                img = self._read_dicom(file_path)
                img = tf.convert_to_tensor(img, dtype=tf.float32)
                img = tf.expand_dims(img, axis=-1)  # Add channel dimension
                img = tf.image.resize(img, self.roi_size)
                img = tf.image.grayscale_to_rgb(img)  # Convert to RGB
                images.append(img)

        # Pad images to 192 if necessary, we're padding to max number of dicom images in a series
        # we precalculated that train_images/2508151528/1550325930 contains the largest number of
        # images in a series with 192 images. THe model expects tensors of the same size, so we pad
        # the images to 192. 
        if len(images) < 192:
            padding = tf.zeros((192 - len(images), *images[0].shape), dtype=tf.float32)
            images = tf.concat([images, padding], axis=0)

        return tf.stack(images)

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


    def feature_label_generator(self):
        label_coordinates_df = pd.read_csv(self.label_coordinates_csv)
        labels_df = pd.read_csv(self.label_coordinates_csv)

        # Create a combined label column, this should match the header of the labels.csv file
        labels_df['label'] = labels_df.apply(
            lambda row: row['condition'].replace(' ', '_') + '_' + row['level'].replace(' ', '_'), axis=1
        )

        # Extract unique labels and store them from train.csv, we will match generated labels against this list
        self.label_list = pd.read_csv(self.labels_csv).columns[1:].tolist()

        row = label_coordinates_df.sample(n=1) # randomly select one row from the dataframe to return
        
        study_id = row['study_id'].values[0]
        series_id = row['series_id'].values[0]
        condition = row['condition'].values[0]
        level = row['level'].values[0]
        x = row['x'].values[0]
        y = row['y'].values[0]
        
        print(f"Going to generate feature for study_id: {study_id}, series_id: {series_id}, condition: {condition}, level: {level}")
        img_tensor = self._preprocess_image(study_id, series_id)
        
        # Create a unique label for the combination of study_id, series_id, condition, and level
        label = f"{row['condition'].values[0].replace(' ', '_').lower()}_{row['level'].values[0].replace('/', '_').lower()}"
        try:
            label_vector = self.label_list.index(label)
        except ValueError:
            raise ValueError(f"Label {label} not found in the label list")

        # Create a one-hot encoded vector
        one_hot_vector = [0.0] * len(self.label_list)
        one_hot_vector[label_vector] = 1.0

        yield img_tensor, np.array(one_hot_vector, dtype=np.float32)

    def create_dataset(self):
        """
        Create a TensorFlow dataset from feature and label generators.

        Returns:
            tf.data.Dataset: Combined dataset of features and labels.
        """

        dataset = tf.data.Dataset.from_generator(
            self.feature_label_generator,
            output_signature=(
                tf.TensorSpec(shape=(192, self.roi_size[0], self.roi_size[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(25,), dtype=tf.float32)
            )
        )
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

    def _filter_dataset(self, dataset, threshold, offset=0.0, is_training=True):
        """
        Filter dataset based on random value.

        Args:
            dataset (tf.data.Dataset): Input dataset.
            threshold (float): Threshold for splitting.
            offset (float): Offset for splitting. Defaults to 0.0.
            is_training (bool): Flag to indicate if it's the training set.

        Returns:
            tf.data.Dataset: Filtered dataset.
        """
        if is_training:
            return dataset.filter(lambda _, __, z: z >= offset and z < offset + threshold)
        else:
            return dataset.filter(lambda _, __, z: z < offset or z >= offset + threshold)

    def _split_dataset(self, dataset, val_split=0.2, test_split=0.1):
        """
        Split a dataset into training, validation, and test sets.

        Args:
            dataset (tf.data.Dataset): Input dataset.
            val_split (float): Fraction of data to use for validation.
            test_split (float): Fraction of data to use for testing.

        Returns:
            tuple: Training dataset, validation dataset, and test dataset.
        """
        dataset_with_random = dataset.map(self._add_random_value)

        train_dataset = self._filter_dataset(
            dataset_with_random, val_split + test_split, is_training=True
        )
        val_dataset = self._filter_dataset(
            dataset_with_random, val_split, is_training=False, offset=val_split
        )
        test_dataset = self._filter_dataset(
            dataset_with_random, test_split, is_training=False, offset=val_split + test_split
        )

        train_dataset = train_dataset.map(self._remove_random_value)
        val_dataset = val_dataset.map(self._remove_random_value)
        test_dataset = test_dataset.map(self._remove_random_value)

        return train_dataset, val_dataset, test_dataset

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
        train_dataset, val_dataset, test_dataset = self._split_dataset(dataset)

        if self.batch_size:
            train_dataset = train_dataset.batch(self.batch_size)
            val_dataset = val_dataset.batch(self.batch_size)
            test_dataset = test_dataset.batch(self.batch_size)
        else:
            train_dataset = train_dataset.batch(constants.BATCH_SIZE)
            val_dataset = val_dataset.batch(constants.BATCH_SIZE)
            test_dataset = test_dataset.batch(constants.BATCH_SIZE)

        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
        print("Dataset created, you can now iterate over the dataset")

        return train_dataset, val_dataset, test_dataset

# Usage example (commented out):
# image_loader = ImageLoader(label_coordinates_csv='label_coordinates.csv', labels_csv='labels.csv',
#                               image_dir='images', roi_size=(224, 224), batch_size=32)
# train_dataset, val_dataset = image_loader.load_data()
# for img, labels in train_dataset.take(1):
#     print(img.shape)
#     print(labels.shape)