import os
import pandas as pd
import numpy as np
import pydicom
import cv2
import tensorflow as tf
from typing import Any, Iterator, Tuple
import time
from collections import Counter

from vision_models.instance import InstanceCoordinates
import vision_models.constants as constants

from sklearn.model_selection import train_test_split

class Dataset:
    """
    A class for loading and preprocessing medical image data for machine learning tasks.

    This class handles loading DICOM images, extracting relevant patches, and creating
    TensorFlow datasets for training and validation.
    """

    def __init__(
        self, batch_size: None
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
        self.image_dir = constants.TRAIN_DATA_PATH
        self.label_coordinates_csv = constants.TRAIN_LABEL_CORD_PATH
        self.labels_csv = constants.TRAIN_LABEL_PATH
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
        
    def _prepare_data(self):
        # Read the label coordinates CSV file and create a DataFrame
        df = self._create_train_label_cord_dataframe()

        # Create the splits
        self.train_df, self.val_df, self.test_df, self.split_data = self._create_split(df)
        
        # Extract unique labels and store them from labels.csv
        self.label_list = pd.read_csv(self.labels_csv).columns[1:].tolist()
        print("#"* 100)
        print("Dataset splits sizes:", self.get_df_sizes())

    def _create_human_readable_label(self, label_vector, label_list):
        """ Create human-readable labels from one-hot encoded vector. """
        # Convert label_vector to numpy array if it's not already
        label_vector = np.array(label_vector)
        # Remove the [0] indexing if label_vector is 1D
        indices_with_ones = np.where(label_vector == 1)[0]
        # Select the corresponding labels from label_list
        human_readable_labels = [label_list[i] for i in indices_with_ones]
        # Output the labels
        print("Human-readable labels:", human_readable_labels)
        
    def _create_train_label_cord_dataframe(self):
        """
        Create a DataFrame from the label coordinates CSV file.

        Returns:
            pd.DataFrame: DataFrame with study_id, series_id, instance_number, x, y, condition, and level columns.
        """
        df = pd.read_csv(self.label_coordinates_csv)
        
        # Convert to string types
        df["study_id"] = df["study_id"].astype(str)
        df["series_id"] = df["series_id"].astype(str)
        
        # Transform the data
        df["level"] = df["level"].str.replace("/", "_")
        df["class"] = df["condition"].str.replace(" ", "_").str.lower() + "_" + df["level"].str.lower()
        
        # Create series directory column
        df["series_dir"] = self.image_dir + "/" + df["study_id"] + "/" + df["series_id"] + "/"
        
        return df  

    def _create_split(self, df):
        df['composite_key'] = df['study_id'].astype(str) + '_' + df['series_id'].astype(str) + '_' + df['class'] 
        class_distribution = df['composite_key'].value_counts()

        # Split into train and test, then further split train into train and validation
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

        # Create splits
        train_split = df[df['composite_key'].isin(train_ids)].copy()
        val_split = df[df['composite_key'].isin(val_ids)].copy()
        test_split = df[df['composite_key'].isin(test_ids)].copy()

        return train_split, val_split, test_split, df

    def get_df_sizes(self):
        size = {}
        {size.update({split: len(df)}) for split, df in zip(['train', 'val', 'test'], [self.train_df, self.val_df, self.test_df])}
        return size

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
        masked_images = {}
        
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
                    mask = Dataset._create_gaussian_mask(image_shape, (x, y))                    
                    combined_mask = np.maximum(combined_mask, mask)
                    
                # Convert the single-channel image and mask to three-channel
                image = np.expand_dims(image, axis=-1)
                combined_mask = np.expand_dims(combined_mask, axis=-1)
                combined_image = np.concatenate([image, combined_mask, combined_mask], axis=-1)
                
                masked_images[instance_number] = combined_image
            else:
                print(f"File {file_path} not found.")
        
        return masked_images

 
    @staticmethod
    def _create_gaussian_mask(image_shape, center, sigma=10):
        """ Create a 2D Gaussian mask centered at center with standard deviation sigma. """
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        center_x, center_y = center
        mask = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
        return mask
    
    def _create_instance_coordinates(self, df, study_id, series_id):
        instances = InstanceCoordinates(study_id, series_id)
        
        # Check before filtering
        subset_df = df[(df['study_id'] == study_id) & (df['series_id'] == series_id)]
       
        for _, row in subset_df.iterrows():
            instance_number = row['instance_number']
            x = row['x']
            y = row['y']
            instances.add_coordinates(instance_number, x, y)
            
        return instances
    
    def _preprocess_image(self, df, study_id, series_id):
        """ Preprocess images for a given study and series. """
        
        # Get series_dir from self.train_df
        series_dir = df[(df["study_id"] == study_id) & (df["series_id"] == series_id)]["series_dir"].values[0]
        # Get sorted list of DICOM files
        dicom_files = sorted([f for f in os.listdir(series_dir) if f.endswith(".dcm")])
        
        # Create instances object
        instances = self._create_instance_coordinates(df, study_id, series_id)
        
        # Apply Gaussian attention to relevant images
        attended_images = self._apply_gaussian_attention(instances)
        
        images = []

        for filename in dicom_files:
            instance_number = int(filename.split('.')[0])
            file_path = os.path.join(series_dir, filename)
            
            dicom_image = pydicom.dcmread(file_path)
            image = dicom_image.pixel_array
            
            # Check if the image needs attention
            if instance_number in attended_images:
                img = attended_images[instance_number]
            else:
                # Convert the single-channel image to three-channel by duplicating the original image
                image = np.expand_dims(image, axis=-1)
                img = np.concatenate([image, image, image], axis=-1)
            
            img = tf.convert_to_tensor(img, dtype=tf.float32)
            
            # Resize the image to the desired ROI size
            img = tf.image.resize(img, self.roi_size)
            
            images.append(img)

        # Pad images to 192 if necessary
        if len(images) < 192:
            padding = tf.zeros((192 - len(images), *self.roi_size, 3), dtype=tf.float32)
            images = tf.concat([tf.stack(images), padding], axis=0)
        else:
            images = tf.stack(images[:192])  # Truncate to 192 if more

        return images
    
    def _base_generator(self, df: pd.DataFrame, split: str, repeat: bool = False) -> Iterator[Tuple[tf.Tensor, tf.Tensor]]:
        total_rows = len(df)
        count = 0
        unique_study_ids = set()
        unique_labels = set()
        start_time = time.time()

        while True:
            df_copy = df.copy()
            
            while not df_copy.empty:
                count += 1
                if count % 1000 == 0:
                    elapsed_time = time.time() - start_time
                    print(f" Generated {count}/{total_rows} samples for {split}")
                    print(f" Time elapsed: {elapsed_time:.2f} seconds")
                    print(f" Remaining rows in df: {len(df_copy)}")

                # Get the first row and drop it from df_copy
                row = df_copy.iloc[0]
                df_copy = df_copy.drop(df_copy.index[0])

                study_id = row["study_id"]
                series_id = row["series_id"]

                unique_study_ids.add(study_id)

                img_tensor = self._preprocess_image(df, study_id, series_id)

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
        # shuffle train_df before passing to _base_generator
        self.train_df = self.train_df.sample(frac=1, random_state=42).reset_index(drop=True)
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

    def create_dataset(self, split: str) -> Tuple[Any, int]:
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

        return dataset

    def load_data(self, split: str):
        """ Main method to load data for a given split """
        print(f"\n ----- Creating dataset for *{split}* ------ \n")
        
        dataset = self.create_dataset(split)       
        if self.batch_size:
            batch_size = self.batch_size
        else:
            batch_size = constants.BATCH_SIZE
        
        print("Batching the dataset to batch_size:", batch_size)
        dataset = dataset.batch(self.batch_size)
        
        if split in ["val"]:
            dataset = dataset.repeat()

        return dataset