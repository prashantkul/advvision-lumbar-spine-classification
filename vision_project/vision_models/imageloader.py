import os
import pandas as pd
import numpy as np
import random
import pydicom
import cv2
import tensorflow as tf
from tensorflow.data import Dataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import vision_models.constants as constants
from typing import Any, Iterator, Tuple

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

        # Set the random seed for reproducibility
        self.random_seed = constants.RANDOM_SEED
        self._set_random_seed(self.random_seed)

        # Load label data
        self.label_df = self._load_labels_df()
        self.label_coordinates_df = pd.read_csv(self.label_coordinates_csv)

        # Split the data into train, val, test
        self.train_ids, self.val_ids, self.test_ids = self.split_dataset()

    def _set_random_seed(self, seed):
        """Set the random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

    def _load_labels_df(self):
        """Load the label dataframe from the CSV file."""
        return pd.read_csv(self.labels_csv)
    
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
        print("Preprocessing images")
        series_dir = f"{self.image_dir}/{study_id}/{series_id}"
        print(f"Reading images from {series_dir}")
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
        print(f"Number of images in series: {len(images)}")
        if len(images) < 192:
            print(f"Padding tensor to 192 images")
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

    def split_dataset(self):
        """
        Split the data into train, validation, and test sets.
        """
        # self._set_random_seed(self.random_seed)  # Set the random seed

        # train_size = int(constants.TRAIN_DATASET * len(self.label_df))
        # val_size = int(constants.VAL_DATASET * len(self.label_df))
        # test_size = int(constants.TEST_DATASET * len(self.label_df))

        # dataset = self.create_dataset()
        # dataset = dataset.shuffle(buffer_size=len(self.label_df), seed=self.random_seed)

        # train_dataset = dataset.take(train_size)
        # val_dataset = dataset.skip(train_size).take(val_size)
        # test_dataset = dataset.skip(train_size + val_size).take(test_size)

        # return train_dataset, val_dataset, test_dataset

        self._set_random_seed(self.random_seed)  # Set the random seed

        unique_ids = self.label_df['study_id'].unique()
        np.random.shuffle(unique_ids)

        train_end = int(constants.TRAIN_DATASET * len(unique_ids))
        val_end = train_end + int(constants.VAL_DATASET * len(unique_ids))

        train_ids = unique_ids[:train_end]
        val_ids = unique_ids[train_end:val_end]
        test_ids = unique_ids[val_end:]

        return train_ids, val_ids, test_ids

    # def feature_label_generator(self, split) -> Iterator[Tuple[tf.Tensor, tf.Tensor]]:
    #     label_coordinates_df = pd.read_csv(self.label_coordinates_csv)

    #     # Filter based on split
    #     if split == 'train':
    #         split_ids = self.train_ids
    #     elif split == 'val':
    #         split_ids = self.val_ids
    #     elif split == 'test':
    #         split_ids = self.test_ids
    #     else:
    #         raise ValueError(f"Unknown split: {split}")

    #     label_coordinates_df = label_coordinates_df[label_coordinates_df['study_id'].isin(split_ids)]

    #     # Create a combined label column, this should match the header of the labels.csv file
    #     label_coordinates_df['label'] = label_coordinates_df.apply(
    #         lambda row: row['condition'].replace(' ', '_') + '_' + row['level'].replace(' ', '_'), axis=1
    #     )

    #     # Extract unique labels and store them from train.csv, we will match generated labels against this list
    #     if self.label_list is None:
    #         self.label_list = pd.read_csv(self.labels_csv).columns[1:].tolist()

    #     # Shuffle the dataframe
    #     label_coordinates_df = label_coordinates_df.sample(frac=1).reset_index(drop=True)

    #     if len(label_coordinates_df) == 0:
    #         raise ValueError("All study_id's have been exhausted.")
        
    #     while len(label_coordinates_df) > 0:
    #         print("*" * 100)
    #         row = label_coordinates_df.sample(n=1) # randomly select one row from the dataframe to return
            
    #         study_id = row['study_id'].values[0]
    #         series_id = row['series_id'].values[0]
    #         condition = row['condition'].values[0]
    #         level = row['level'].values[0]
    #         x = row['x'].values[0]
    #         y = row['y'].values[0]
            
    #         print(f"Fetching data for study_id: {study_id}, batch size: {self.batch_size}")
    #         print(f"Going to generate feature for study_id: {study_id}, series_id: {series_id}, condition: {condition}, level: {level}")
    #         img_tensor = self._preprocess_image(study_id, series_id)
    #         print(f"Feature tensor generated, size: {img_tensor.shape}, now generating label")
    #         # Create a unique label for the combination of study_id, series_id, condition, and level
    #         label = f"{row['condition'].values[0].replace(' ', '_').lower()}_{row['level'].values[0].replace('/', '_').lower()}"
    #         try:
    #             label_vector = self.label_list.index(label)
    #         except ValueError:
    #             raise ValueError(f"Label {label} not found in the label list")
    #         print(f"Label generated")
    #         # Create a one-hot encoded vector
    #         one_hot_vector = [0.0] * len(self.label_list)
    #         one_hot_vector[label_vector] = 1.0
    #         print("Returning feature and label tensors")
    #         yield img_tensor, np.array(one_hot_vector, dtype=np.float32)

    def feature_label_generator(self) -> Iterator[Tuple[tf.Tensor, tf.Tensor]]:
            label_coordinates_df = pd.read_csv(self.label_coordinates_csv)

            # Create a combined label column, this should match the header of the labels.csv file
            label_coordinates_df['label'] = label_coordinates_df.apply(
                lambda row: row['condition'].replace(' ', '_') + '_' + row['level'].replace(' ', '_'), axis=1
            )

            # Extract unique labels and store them from train.csv, we will match generated labels against this list
            self.label_list = pd.read_csv(self.labels_csv).columns[1:].tolist()

            # shuffle the dataframe
            label_coordinates_df = label_coordinates_df.sample(frac=1).reset_index(drop=True)

            if len(label_coordinates_df) == 0:
                raise ValueError("All study_id's have been exhausted.")
            
            while len(label_coordinates_df) > 0:
                print("*"*100)
                row = label_coordinates_df.sample(n=1) # randomly select one row from the dataframe to return
                
                study_id = row['study_id'].values[0]
                series_id = row['series_id'].values[0]
                condition = row['condition'].values[0]
                level = row['level'].values[0]
                x = row['x'].values[0]
                y = row['y'].values[0]
                
                print(f"Fetching data for study_id: {study_id}, batch size: {self.batch_size}")
                print(f"Going to generate feature for study_id: {study_id}, series_id: {series_id}, condition: {condition}, level: {level}")
                img_tensor = self._preprocess_image(study_id, series_id)
                print(f"Feature tensor generated, size: {img_tensor.shape}, now generating label")
                # Create a unique label for the combination of study_id, series_id, condition, and level
                label = f"{row['condition'].values[0].replace(' ', '_').lower()}_{row['level'].values[0].replace('/', '_').lower()}"
                try:
                    label_vector = self.label_list.index(label)
                except ValueError:
                    raise ValueError(f"Label {label} not found in the label list")
                print(f"Label generated")
                # Create a one-hot encoded vector
                one_hot_vector = [0.0] * len(self.label_list)
                one_hot_vector[label_vector] = 1.0
                print("Returning feature and label tensors")
                yield img_tensor, np.array(one_hot_vector, dtype=np.float32)


    def create_dataset(self, split):
        """
        Create a TensorFlow dataset from feature and label generators.

        Returns:
            tf.data.Dataset: Combined dataset of features and labels.
        """

        # dataset = tf.data.Dataset.from_generator(
        #     self.feature_label_generator,
        #     output_signature=(
        #         tf.TensorSpec(shape=(192, self.roi_size[0], self.roi_size[1], 3), dtype=tf.float32),
        #         tf.TensorSpec(shape=(25,), dtype=tf.float32)
        #     )
        # )
        # return dataset

        dataset = tf.data.Dataset.from_generator(
            # lambda: self.feature_label_generator(split),
            self.feature_label_generator,
            output_signature=(
                tf.TensorSpec(shape=(192, self.roi_size[0], self.roi_size[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(25,), dtype=tf.float32)
            )
        )

        if self.batch_size:
            dataset = dataset.batch(self.batch_size)
        else:
            dataset = dataset.batch(constants.BATCH_SIZE)

        return dataset

    def load_data(self, split):
        """
        Load and prepare the data for training and validation.

        Returns:
            tuple: Training dataset and validation dataset.
            Each dataset will have image and label tensors.
        """
        # # # Create the dataset
        # dataset = self.create_dataset(split)
        # print(f"Dataset for {split} is created, setting batch size")

        # if self.batch_size:
        #     print(f"Batching dataset to: {self.batch_size}")
        #     dataset = dataset.batch(self.batch_size)
        # else:
        #     print(f"Batching dataset to default defined in constants: {constants.BATCH_SIZE}")
        #     dataset = dataset.batch(constants.BATCH_SIZE)

        # print(f"Dataset for {split} created, you can now iterate over the dataset")

        # return dataset
        
        # Filter the label dataframe based on the split
        if split == 'train':
            split_ids = self.train_ids
        elif split == 'val':
            split_ids = self.val_ids
        elif split == 'test':
            split_ids = self.test_ids
        else:
            raise ValueError(f"Unknown split: {split}")

        # Filter the label dataframe based on the split_ids
        self.label_df = self.label_df[self.label_df['study_id'].isin(split_ids)]

        # Create the dataset
        dataset = self.create_dataset()
        print(f"Dataset for {split} is created, setting batch size")

        if self.batch_size:
            print(f"Batching dataset to: {self.batch_size}")
            dataset = dataset.batch(self.batch_size)
        else:
            print(f"Batching dataset to default defined in constants: {constants.BATCH_SIZE}")
            dataset = dataset.batch(constants.BATCH_SIZE)

        print(f"Dataset for {split} created, you can now iterate over the dataset")

        return dataset

    def analyze_split(self):
        """
        Analyze the split of the dataset.
        """
        # self.label_df['split'] = pd.cut(
        #     self.label_df.index,
        #     bins=[0, int(constants.TRAIN_DATASET * len(self.label_df)), 
        #         int((constants.TRAIN_DATASET + constants.VAL_DATASET) * len(self.label_df)), 
        #         len(self.label_df)],
        #     labels=['train', 'val', 'test']
        # )

        # split_counts = self.label_df['split'].value_counts()
        # print(f"Number of unique study_ids in each split:\n{self.label_df.groupby('split')['study_id'].nunique()}")
        # print(f"\nNumber of images in each split:\n{split_counts}")

        # label_coordinates_df = pd.read_csv(self.label_coordinates_csv)

        # label_coordinates_df['split'] = label_coordinates_df['study_id'].apply(
        #     lambda study_id: 'train' if study_id in self.train_ids else 'val' if study_id in self.val_ids else 'test'
        # )

        # split_counts = label_coordinates_df['split'].value_counts()
        # unique_studies_per_split = label_coordinates_df.groupby('split')['study_id'].nunique()

        # print(f"Number of unique study_ids in each split:\n{unique_studies_per_split}")
        # print(f"\nNumber of instances in each split:\n{split_counts}")

        # return unique_studies_per_split, split_counts

        self.label_coordinates_df['split'] = self.label_coordinates_df['study_id'].apply(
            lambda study_id: 'train' if study_id in self.train_ids else 'val' if study_id in self.val_ids else 'test'
        )

        split_counts = self.label_coordinates_df['split'].value_counts()
        unique_studies_per_split = self.label_coordinates_df.groupby('split')['study_id'].nunique()

        print(f"Number of unique study_ids in each split:\n{unique_studies_per_split}")
        print(f"\nNumber of instances in each split:\n{split_counts}")

        # Show the first 5 rows of the label_coordinates_df with the split column
        print("\nFirst 5 rows of the label_coordinates_df with the split column:")
        print(self.label_coordinates_df.head())

        # Show the first 5 rows of the label_coordinates_df with the split column
        print("\nFirst 5 rows of the label_df with the split column:")
        print(self.label_df.head())

        return unique_studies_per_split, split_counts
    
    def save_split_labels(self, filename='label_coordinates_with_split.csv'):
        """
        Save the label_coordinates_df with the split column to a CSV file.

        Args:
            filename (str): The name of the file to save the DataFrame to.
        """
        self.label_coordinates_df.to_csv(filename, index=False)
        print(f"Saved the DataFrame with the split column to {filename}")


# Usage example (commented out):
# image_loader = ImageLoader(label_coordinates_csv='label_coordinates.csv', labels_csv='labels.csv',
#                               image_dir='images', roi_size=(224, 224), batch_size=32)
# train_dataset, val_dataset = image_loader.load_data()
# for img, labels in train_dataset.take(1):
#     print(img.shape)
#     print(labels.shape)