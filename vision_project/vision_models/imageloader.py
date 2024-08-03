import os
import pandas as pd
import numpy as np
import pydicom
import cv2
import tensorflow as tf
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
        self.split = None
        self.study_ids: list[str] = []
        
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

    def _create_split(self, df):
        """
        Create a split of the data based on the split column.

        Args:
            dataframe (pd.DataFrame): Input dataframe.

        Returns:
            tuple: Train and validation dataframes.
        """
        # given a dataframe, split it into train, test and validation dataframes
        # USING TRAIN_SPLIT = 0.80 VAL_SPLIT = 0.10 TEST_SPLIT = 0.10
        # Calculate split indices
        train_end = int(len(df) * constants.TRAIN_SPLIT)
        val_end = train_end + int(len(df) * constants.VAL_SPLIT)

        # Shuffle the DataFrame
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Calculate split indices
        train_end = int(len(df) * constants.TRAIN_SPLIT)
        val_end = train_end + int(len(df) * constants.VAL_SPLIT)

        # Initialize the 'split' column
        df["split"] = None

        # Assign split labels
        df.loc[:train_end, "split"] = "train"
        df.loc[train_end:val_end, "split"] = "val"
        df.loc[val_end:, "split"] = "test"

        return df

    def _analyze_splits(self):
        label_coordinates_df = pd.read_csv(self.label_coordinates_csv)
        # split the dataframe for train, test and validation
        label_coordinates_df = self._create_split(label_coordinates_df)

    def _create_human_readable_label(self, label_vector, labels_df):
        # Convert label_vector to numpy array if it's not already
        label_vector = np.array(label_vector)
        # Remove the [0] indexing if label_vector is 1D
        indices_with_ones = np.where(label_vector == 1)[0]
        # Make sure labels_df has an index that matches the indices_with_ones
        human_readable_labels = labels_df.iloc[indices_with_ones]["label"].tolist()
        # Output the labels
        print("Human-readable labels:", human_readable_labels)
    
    def _filter_df(self, df, study_ids):
        print(f"Label coordinates DF will be filtered based on study_ids: {study_ids}")
        self_study_ids = list(map(int, self.study_ids))
        df = df[df["study_id"].isin(self_study_ids)]
        print(f"Label coordinates DF filtered based on study_ids, new shape: {df.shape}")
        return df

    def _feature_label_generator(self) -> Iterator[Tuple[tf.Tensor, tf.Tensor]]:
        """ Generate features and labels for the dataset. 
            This method implements a method to return feature and label tensors for the dataset.
            
            The method reads the label_coordinates_csv file and reads a random row from the file.
            It then generates a feature tensor by calling the _preprocess_image method and generates a label tensor
            by creating a one-hot encoded vector for the label. The method then returns the feature and label tensors.
            
            The while loop returns unlimited number of feature and label tensors until all rows from the label_coordinates_csv
            file are exhausted.
        
        """
        # read the label_coordinates.csv file       
        label_coordinates_df = pd.read_csv(self.label_coordinates_csv)
        
        # filter the label_corrdinates_df based on the study_ids
        if self.study_ids:
            print("Available Study IDs in DataFrame:", len(label_coordinates_df['study_id'].unique()))
            label_coordinates_df = self._filter_df(label_coordinates_df, self.study_ids)
            if len(label_coordinates_df) == 0:
                print("No rows found for the study_ids provided")
        
        labels_df = pd.read_csv(self.label_coordinates_csv)

        # Create a combined label column, this should match the header of the labels.csv file
        labels_df["label"] = labels_df.apply(
            lambda row: (
                row["condition"].replace(" ", "_")
                + "_"
                + row["level"].replace(" ", "_").replace("/", "_")
            ).lower(),
            axis=1,
        )

        # Extract unique labels and store them from train.csv, we will match generated labels against this list
        self.label_list = pd.read_csv(self.labels_csv).columns[1:].tolist()

        # split the dataframe for train, test and validation
        label_coordinates_df = self._create_split(label_coordinates_df)

        # This loop iterates until all the rows have beene exahused for the generator
        while len(label_coordinates_df) > 0:
            print("*" * 100)

            # Filter the dataframe based on the split requested in the generator
            if self.split in ["train", "val", "test"]:
                label_coordinates_df = label_coordinates_df[
                    label_coordinates_df["split"] == self.split]
            
            # randomly select one row from the dataframe to return
            row = label_coordinates_df.sample(n=1)  

            study_id = row["study_id"].values[0]
            series_id = row["series_id"].values[0]
            condition = row["condition"].values[0]
            level = row["level"].values[0]
            x = row["x"].values[0]
            y = row["y"].values[0]

            print(
                f"Going to generate feature for study_id: {study_id}, series_id: {series_id}, condition: {condition}, level: {level}"
            )

            # Preprocess the image before supplying to the generator.
            # TBD: This is the place for Gaussian attention mask
            img_tensor = self._preprocess_image(study_id, series_id)
            print(
                f"Feature tensor generated, size: {img_tensor.shape}, now generating label"
            )
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

            print(f"One hot vector generated: {one_hot_vector}")
            self._create_human_readable_label(one_hot_vector, labels_df)

            one_hot_vector_array = np.array(one_hot_vector, dtype=np.float32)
            print("Returning feature and label tensors")
            yield img_tensor, one_hot_vector_array

    def create_dataset(self):
        """
        Create a TensorFlow dataset from feature and label generators.

        Returns:
            tf.data.Dataset: Combined dataset of features and labels.
        """

        dataset = tf.data.Dataset.from_generator(
            self._feature_label_generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(192, self.roi_size[0], self.roi_size[1], 3), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(25,), dtype=tf.float32),
            ),
        )
        return dataset

    def load_data(self, split, 
                  study_ids: list[str] = None):
        """
        Load and prepare the data for training and validation.

        Returns:
            tuple: Training dataset and validation dataset.
            Each dataset will have image and label tensors.
        """
        # Filter the label dataframe based on the split
        if split == "train" or split == "val" or split == "test":
            self.split = split
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if study_ids is not None:
            self.study_ids = study_ids

        # Create the dataset
        dataset = self.create_dataset()
        print("Dataset is created, setting batch size")

        if self.batch_size:
            print("Batching dataset to :", self.batch_size)
            dataset = dataset.batch(self.batch_size)

        else:
            print(
                "Batching dataset to default defined in constants:",
                constants.BATCH_SIZE,
            )
            dataset = dataset.batch(constants.BATCH_SIZE)

        print("Dataset created, you can now iterate over the dataset")

        return dataset


# Usage example (commented out):
# image_loader = ImageLoader(label_coordinates_csv='label_coordinates.csv', labels_csv='labels.csv',
#                               image_dir='images', roi_size=(224, 224), batch_size=32)
# train_dataset, val_dataset = image_loader.load_data()
# for img, labels in train_dataset.take(1):
#     print(img.shape)
#     print(labels.shape)
