import os
import pandas as pd
import numpy as np
import pydicom
import cv2
import tensorflow as tf
import vision_models.constants
from typing import Any, Iterator, Tuple
import time
from collections import Counter
from instance import InstanceCoordinates
import logging
from constants import constants

from sklearn import train_test_split

class Dataset:
    """
    A class for loading and preprocessing medical image data for machine learning tasks.

    This class handles loading DICOM images, extracting relevant patches, and creating
    TensorFlow datasets for training and validation.
    """

    def __init__(
        self, image_dir, label_coordinates_csv, labels_csv, batch_size
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
        self.batch_size = batch_size
        self.label_list = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.split_data = None
        
        # get ROI size from constants.IMAGE_SIZE_HEIGHT and constants.IMAGE_SIZE_WIDTH
        self.roi_size = (constants.IMAGE_SIZE_HEIGHT, constants.IMAGE_SIZE_WIDTH)
        
        # Read train_label_cord CSV file and create a DataFrame
        # Read train.csv and create a list of labels
        self._prepare_data()

        
    def _create_train_label_cord_dataframe(self):
        """
        Create a DataFrame from the label coordinates CSV file.

        Returns:
            pd.DataFrame: DataFrame with study_id, series_id, instance_number, x, y, condition, and level columns.
        """
        df = pd.read_csv(self.label_coordinates_csv)
        df["study_id"] = df["study_id"].astype(str)
        df["series_id"] = df["series_id"].astype(str)
        
        # create a new column for condition - its a combination of condition and level, lowercase
        # first remove / from level
        df["level"] = df["level"].str.replace("/", "_")
        df["class"] = df["condition"].str.lower() + "_" + df["level"].str.lower()
        
        # create series directory column
        df["series_dir"] = self.image_dir + "/" + df["study_id"] + "/" + df["series_id"] + "/"
        
        print(f"Created train_label_cord dataframe with shape: {df.shape}")
        print("Columns: ", df.columns)
        print(df.head(5))
            
        return df    

    def _create_split(self, df):
        df['composite_key'] = df['study_id'].astype(str) + '_' + \
                              df['series_id'].astype(str) + '_' + \
                              df['class'] 

        class_distribution = df['composite_key'].value_counts()

        train_ids, test_ids = train_test_split(
            class_distribution.index,
            test_size=0.2,
            stratify=class_distribution.values,
            random_state=42
        )

        train_ids, val_ids = train_test_split(
            train_ids,
            test_size=0.25,
            stratify=class_distribution[train_ids].values,
            random_state=42
        )

        train_split = df[df['composite_key'].isin(train_ids)].copy()
        val_split = df[df['composite_key'].isin(val_ids)].copy()
        test_split = df[df['composite_key'].isin(test_ids)].copy()

        train_split['split'] = 'train'
        val_split['split'] = 'val'
        test_split['split'] = 'test'

        split_data = pd.concat([train_split, val_split, test_split])
        split_data.drop(columns=['composite_key'], inplace=True)

        print("Total Data Shape:", split_data.shape)
        print("Train Set Shape:", train_split.shape)
        print("Validation Set Shape:", val_split.shape)
        print("Test Set Shape:", test_split.shape)

        return train_split, val_split, test_split, split_data

    def _create_instance_coordinates(self, df, study_id, series_id):
        """ Create an InstanceCoordinates object from the DataFrame for the given study_id and series_id. """
        instances = InstanceCoordinates(study_id, series_id)
        series_df = df[(df["study_id"] == study_id) & (df["series_id"] == series_id)]
        for _, row in series_df.iterrows():
            instances.add_coordinates(row["instance_number"], row["x"], row["y"])
        return instances
        

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
    
    
    def _apply_gaussian_attention(self, instances: InstanceCoordinates):
        """Apply Gaussian attention to images based on the provided InstanceCoordinates object."""
        
        # Directory containing the images
        series_dir = f"{self.image_dir}/{instances.study_id}/{instances.series_id}"
        
        # Initialize a list to store the mask-applied images
        masked_images = []
        
        # Read and process each DICOM image from the file system
        for instance_number in instances.data.keys():
            filename = f"{instance_number}.dcm"
            file_path = os.path.join(series_dir, filename)
            
            if os.path.exists(file_path):
                dicom_image = pydicom.dcmread(file_path)
                image = dicom_image.pixel_array
                image_shape = image.shape
                combined_mask = np.zeros(image_shape)
                
                coordinates = instances.get_coordinates(instance_number)
                for x, y in coordinates:
                    mask = self.create_gaussian_mask(image_shape, x, y)
                    combined_mask = np.maximum(combined_mask, mask)
                
                # Stack the original image and the mask
                masked_image = np.stack([image, combined_mask], axis=-1)
                masked_images.append(masked_image)
            else:
                print(f"File {file_path} not found.")
        
        return masked_images
    

    def create_gaussian_mask(image_shape, x, y, sigma=10):
        """ Create a 2D Gaussian mask centered at (x, y) with standard deviation sigma. """
        xv, yv = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        mask = np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * sigma ** 2))
        return mask
    
    
    def _preprocess_image(self, study_id, series_id):
        """ Preprocess images for a given study and series. """
        
        # Get series_dir from self.train_df
        series_dir = self.train_df[(self.train_df["study_id"] == study_id) & (self.train_df["series_id"] == series_id)]["series_dir"].values[0]
        
        # Create instances object
        instances = self._create_instance_coordinates(self.train_df, study_id, series_id)
        
        print(f"Reading images from: {series_dir}")
        
        # Get sorted list of DICOM files
        dicom_files = sorted([f for f in os.listdir(series_dir) if f.endswith(".dcm")])
        
        images = []

        # Read and process each DICOM image from the file system
        for idx, filename in enumerate(dicom_files):
            file_path = os.path.join(series_dir, filename)
            
            dicom_image = pydicom.dcmread(file_path)
            image = dicom_image.pixel_array
            image_shape = image.shape

            # Apply Gaussian attention mask if the instance_number matches
            instance_number = int(filename.split('.')[0])
            if instance_number in instances.data:
                combined_mask = self._apply_gaussian_attention(instances, image_shape, instance_number)
                
                # Stack the original image and the mask
                attended_img = np.stack([image, combined_mask], axis=-1)
                images.append(attended_img)
            else:
                images.append(image)
        
        # Convert the list of images to tensors and preprocess
        processed_images = []
        for img in images:
            img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
            img_tensor = tf.expand_dims(img_tensor, axis=-1)  # Add channel dimension
            img_tensor = tf.image.resize(img_tensor, self.roi_size)
            img_tensor = tf.image.grayscale_to_rgb(img_tensor)  # Convert to RGB
            processed_images.append(img_tensor)

        # Pad images to 192 if necessary
        print(f"Number of images in series: {len(processed_images)}")
        if len(processed_images) < 192:
            print(f"Padding tensor to 192 images")
            padding = tf.zeros((192 - len(processed_images), *self.roi_size, 3), dtype=tf.float32)
            processed_images = tf.concat([processed_images, padding], axis=0)

        result_tensor = tf.stack(processed_images)
        print(f"Resulting preprocessed image tensor shape: {result_tensor.shape}")
        return result_tensor

    
    def _base_generator(self, df: pd.DataFrame, split: str, repeat: bool = False) -> Iterator[Tuple[tf.Tensor, tf.Tensor]]:
        total_rows = len(df)
        count = 0
        unique_study_ids = set()
        unique_labels = set()
        start_time = time.time()

        while True:
            df_copy = df.copy()
            df_copy = df_copy.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the data
            
            while len(df_copy) > 0:
                count += 1
                if count % 1000 == 0:
                    elapsed_time = time.time() - start_time
                    print(f" Generated {count}/{total_rows} samples for {split}")
                    print(f" Time elapsed: {elapsed_time:.2f} seconds")
                    print(f" Remaining rows in df: {len(df_copy)}")

                row = df_copy.iloc[0]
                df_copy = df_copy.iloc[1:]

                study_id = row["study_id"]
                series_id = row["series_id"]
    
                unique_study_ids.add(study_id)

                img_tensor = self._preprocess_image(study_id, series_id)

                label = row['class']
                
                try:
                    label_vector = self.label_list.index(label)
                    unique_labels.add(label)
                except ValueError:
                        print(f"Error: Label '{label}' not found in the label list")
                        
                one_hot_vector = tf.one_hot(label_vector, depth=len(self.label_list))

                yield img_tensor, one_hot_vector
            
            if not repeat:
                break

        self._print_generator_stats(count, total_rows, unique_study_ids, unique_labels, start_time, split)

    def _train_generator(self) -> Iterator[Tuple[tf.Tensor, tf.Tensor]]:
        yield from self._base_generator(self.train_df, 'train', repeat=False)

    def _val_generator(self) -> Iterator[Tuple[tf.Tensor, tf.Tensor]]:
        yield from self._base_generator(self.val_df, 'val', repeat=True)

    def _test_generator(self) -> Iterator[Tuple[tf.Tensor, tf.Tensor]]:
        yield from self._base_generator(self.test_df, 'test', repeat=True)


    def _print_generator_stats(self, count, total_rows, unique_study_ids, unique_labels, start_time, split):
        total_time = time.time() - start_time
        print(f"\nGenerator completed for {split} split")
        print(f"Total samples generated: {count}")
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Average time per sample: {total_time/count:.4f} seconds")
        print(f"Unique study IDs: {len(unique_study_ids)}")
        print(f"Unique labels: {len(unique_labels)}")

        if count != total_rows:
            print(f"Warning: Generated {count} samples, but expected {total_rows}")

        print("\nDistribution of labels:")
        label_counts = Counter(unique_labels)
        for label, count in label_counts.most_common(10):  # Show top 10 labels
            print(f"  {label}: {count}")

    def _prepare_data(self):
        # Read the label coordinates CSV file and create a DataFrame
        df = self._create_train_label_cord_dataframe()

        # Create the splits
        self.train_df, self.val_df, self.test_df, self.split_data = self._create_split(df)
        
        # Extract unique labels and store them from labels.csv
        self.label_list = pd.read_csv(self.labels_csv).columns[1:].tolist()
    
    
    def create_dataset(self, split: str):
        if split == 'train':
            generator = self._train_generator
            dataset_size = len(self.train_df)
        elif split == 'val':
            generator = self._val_generator
            dataset_size = len(self.val_df)
        elif split == 'test':
            generator = self._test_generator
            dataset_size = len(self.test_df)
        else:
            raise ValueError(f"Invalid split: {split}")

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(192, self.roi_size[0], self.roi_size[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(len(self.label_list),), dtype=tf.float32),
            ),
        )

        return dataset, dataset_size

    def load_data(self, split: str, batch_size: int = None) -> Tuple[Any, int]:
        """ Main method to load data for a given split """
        print(f"load_data called for *{split}* split")
        
        dataset, dataset_size = self.create_dataset(split)
        
        if batch_size is not None:
            self.batch_size = batch_size
        
        print("Batching the dataset to batch_size:", self.batch_size)
        dataset = dataset.batch(self.batch_size)
        
        if split in ["val"]:
            dataset = dataset.repeat()

        steps_per_epoch = dataset_size // self.batch_size

        return dataset, steps_per_epoch