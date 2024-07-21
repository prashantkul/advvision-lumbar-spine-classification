from typing import Iterator, Optional, Tuple
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pydicom

from vision_models import constants

##################################################################
# This class creates a training, validation and test tensorflow dataset object
# for the vision models.
##################################################################

class VisionDataset:
    """Dataset class for vision models"""

    def __init__(self):
        """Creates the VisionDataset"""
        self.train_dataset_path = constants.TRAIN_DATA_PATH
        self.test_dataset_path = constants.TEST_DATA_PATH
        self.train_label_path = constants.TRAIN_LABEL_PATH
        self.train_label_cord_path = constants.TRAIN_LABEL_CORD_PATH
        self.train_series_desc_path = constants.TRAIN_SERIES_DESC_PATH
        
        self.batch_size = constants.BATCH_SIZE
        self.image_size = constants.IMAGE_SIZE_HEIGHT * constants.IMAGE_SIZE_WIDTH
        self.random_seed = constants.RANDOM_SEED
        self.label_path = constants.TRAIN_LABEL_PATH
        self.label_cord_path = constants.TRAIN_LABEL_CORD_PATH
        self.shuffle_buffer_size = constants.SHUFFLE_BUFFER_SIZE
        
        self.df = self._read_csv(self.train_label_path)
        self.series_desc_df = self._read_csv(self.train_series_desc_path)
        self.label_coord_df = self._read_csv(self.train_label_cord_path)
        self.label_columns = self.df.columns[1:] # study_id is the label column
        self.encoders = self._create_label_encoders()

        self.max_subdirs = self._find_max_subdirs()

    def _read_dicom(self, file_path):
        #print(f"Reading file {file_path}")
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

    def analyze_nan_values(self):
        print("Analyzing NaN values in the dataset:")
        
        # Count total NaN values
        total_nan = self.df.isna().sum().sum()
        print(f"Total NaN values in the dataset: {total_nan}")
        
        # Count NaN values per column
        nan_counts = self.df.isna().sum()
        print("\nNaN counts per column:")
        for column, count in nan_counts.items():
            if count > 0:
                print(f"{column}: {count}")
        
        # Calculate percentage of NaN values per column
        nan_percentages = (self.df.isna().sum() / len(self.df)) * 100
        print("\nPercentage of NaN values per column:")
        for column, percentage in nan_percentages.items():
            if percentage > 0:
                print(f"{column}: {percentage:.2f}%")
        
        # Create a DataFrame with rows containing NaN values
        rows_with_nan = self.df[self.df.isna().any(axis=1)]
        
        if not rows_with_nan.empty:
            print(f"\nFound {len(rows_with_nan)} rows with NaN values.")
            print("Sample of rows with NaN values:")
            print(rows_with_nan.head())
        else:
            print("\nNo rows with NaN values found in the dataset.")
        
        return rows_with_nan
    
    def _read_csv(self, file_path):
        """Read the CSV file containing the labels."""
        print(f"Reading CSV file {file_path}")
        return pd.read_csv(file_path)

    def _find_max_subdirs(self):
        """Find the maximum number of subdirectories across all studies."""
        max_subdirs = 0
        for study_id in self.df['study_id']:
            study_dir = f"{self.train_dataset_path}{os.path.sep}{study_id}"
            subdirs = [d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))]
            max_subdirs = max(max_subdirs, len(subdirs))
        print(f"Maximum number of subdirectories across all studies: {max_subdirs}")
        return max_subdirs

    def _load_and_preprocess_image(self, study_id):
        """ 
        Load and preprocess the images for a study.
        Input: study_id
            train.csv: This file provides the labels for each image. The file contains a study_id, that corresponds to the directory name in the
            train_images folder, and the labels for each condition. The labels are 'Normal/Mild', 'Moderate', and 'Severe'.
            Each study_id folder contains 3 or more subdirectories, each of which contains a series of DICOM images.
            [condition]_[level] - The target labels, such as spinal_canal_stenosis_l1_l2, with the severity levels of Normal/Mild, Moderate, 
            or Severe. Some entries have incomplete labels.
            The subdir_id corresponds to the series_id in the train_series_descriptions.csv file. 
        
        train_series_descriptions.csv: This file provides information about the type of MRI sequence for each series in a study. 
            We can use this to potentially filter or categorize our images. 
            study_id, series_id, series_description (The scan's orientation.)
        
        train_label_coordinates.csv: This file provides the coordinates of the labeled conditions in each image. This could be very 
            useful for creating attention mechanisms or for image segmentation tasks.
            The file contains a 
            study_id, 
            series_id, 
            instance_number (The image's order number within the 3D stack)
            condition - There are three core conditions: spinal canal stenosis, neural_foraminal_narrowing, and subarticular_stenosis. 
            The latter two are considered for each side of the spine.
            level - The relevant vertebrae, such as l3_l4
            x,y - The coordinates are in the format of a bounding box. The x/y coordinates for the center of the area that defined the label.
            
        Output: A Dict containing following keys:
            'images': subdir_tensors,
            'series_descriptions': series_descs_tensor,
            'label_coordinates': label_coords_tensor
        
        Shapes: 
        Image tensor shapes: Example: [1, 6, 43, 224, 224, 1] for images would mean: 
            1 batch, 6 subdirectories, 43 slices per subdir, 224x224 image size, 1 channel.
            We always scale the tensors to the same size. The number of slices can vary per subdir, so we pad with zeros.
            The maximum number of slices across all subdirs is used as the number of slices. In the training dataset it is 6. 
        
        Series tensor shape: [1, 6] for series descriptions and label coordinates would mean: 1 batch, 6 items (one for each subdir).
        
        label coords shape: [1, 25] for labels would mean: 1 batch, 25 label coordinates values.
        
        """
        study_id = tf.strings.as_string(study_id).numpy().decode('utf-8')
        study_dir = f"{self.train_dataset_path}{os.path.sep}{study_id}"
        subdirs = [d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))]

        subdir_tensors = []
        series_descs = []
        label_coords = []

        max_slices = 0

        # First pass: determine the maximum number of slices
        for subdir in sorted(subdirs):
            subdir_path = os.path.join(study_dir, subdir)
            dicom_files = [f for f in os.listdir(subdir_path) if f.endswith('.dcm')]
            max_slices = max(max_slices, len(dicom_files))

        # Second pass: load and preprocess images, padding as necessary
        for subdir in sorted(subdirs):
            subdir_path = os.path.join(study_dir, subdir)
            dicom_files = [f for f in os.listdir(subdir_path) if f.endswith('.dcm')]
            
            subdir_images = []
            for dicom_file in sorted(dicom_files):
                file_path = os.path.join(subdir_path, dicom_file)
                img = self._read_dicom(file_path)
                
                img = tf.convert_to_tensor(img, dtype=tf.float32)
                img = tf.expand_dims(img, axis=-1)  # Add channel dimension
                img = tf.image.resize(img, [224, 224])
                
                subdir_images.append(img)

            # Pad with zero tensors if necessary
            while len(subdir_images) < max_slices:
                subdir_images.append(tf.zeros([224, 224, 1], dtype=tf.float32))

            subdir_tensor = tf.stack(subdir_images)
            subdir_tensors.append(subdir_tensor)

            # Get series description
            series_desc = self.series_desc_df[(self.series_desc_df['study_id'] == int(study_id)) & 
                                            (self.series_desc_df['series_id'] == int(subdir))]['series_description'].values
            series_descs.append(series_desc[0] if len(series_desc) > 0 else '')

            # Get label coordinates
            coords = self.label_coord_df[(self.label_coord_df['study_id'] == int(study_id)) & 
                                        (self.label_coord_df['series_id'] == int(subdir))]
            label_coords.append(coords.to_json())  # Convert DataFrame to JSON string

        # Ensure we have exactly max_subdirs items for each category
        subdir_tensors = subdir_tensors[:self.max_subdirs]
        series_descs = series_descs[:self.max_subdirs]
        label_coords = label_coords[:self.max_subdirs]

        # Pad if necessary
        while len(subdir_tensors) < self.max_subdirs:
            subdir_tensors.append(tf.zeros([max_slices, 224, 224, 1], dtype=tf.float32))
            series_descs.append('')
            label_coords.append('{}')

        # Create and return a dictionary
        return {
            'images': subdir_tensors,
            'series_descriptions': series_descs,
            'label_coordinates': label_coords
        }

    def _generate_label_tensor(self, study_id):
        # Convert study_id tensor to string
        study_id_str = tf.strings.as_string(study_id).numpy().decode('utf-8')
        
        # Find the matching row
        matching_rows = self.df[self.df['study_id'].astype(str) == study_id_str]
        
        if matching_rows.empty:
            print(f"Warning: No matching study_id found for {study_id_str}")
            # Return a tensor of zeros if no match is found
            return tf.zeros([len(self.label_columns)], dtype=tf.int32)
        
        row = matching_rows.iloc[0]
        labels = []
        for col in self.label_columns:
            label_value = row[col]
            encoded_label = self.encoders[col].transform([label_value])[0]
            labels.append(encoded_label)
       
       
        encoded_labels = tf.convert_to_tensor(labels, dtype=tf.int32)

        # Uncomment this to see the dataframe
        
        # # Print the original DataFrame row
        # print(f"\nOriginal data for study_id {study_id_str}:")
        # print(row)
        
        # # Print the encoded labels
        # print("Encoded labels:")
        # print(encoded_labels.numpy())
        # print("\n" + "-"*50 + "\n")
    
        return encoded_labels
    
    def data_generator(self):
        for study_id in self.df['study_id']:
            image_data = self._load_and_preprocess_image(study_id)
            label_data = self._generate_label_tensor(study_id)
            yield image_data, label_data

    def _create_label_encoders(self):
        """Create a dictionary of LabelEncoders for each column with correct ordering and NaN handling."""
        encoders = {}
        for column in self.label_columns:
            le = LabelEncoder()
            # Define the correct order, including a representation for NaN
            ordered_labels = ['Normal/Mild','Moderate','Severe', 'Unknown']
            # Fit the encoder with the ordered labels
            le.fit(ordered_labels)
            encoders[column] = le
        return encoders

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
        
        train_dataset = self._filter_dataset(dataset_with_random, val_split, is_training=True)
        val_dataset = self._filter_dataset(dataset_with_random, val_split, is_training=False)
        
        train_dataset = train_dataset.map(self._remove_random_value)
        val_dataset = val_dataset.map(self._remove_random_value)
        
        return train_dataset, val_dataset
    
    def _create_initial_dataset(self):
        study_ids = self.df['study_id'].tolist()
        output_signature = (
        {
            'images': tf.TensorSpec(shape=(self.max_subdirs, None, 224, 224, 1), dtype=tf.float32),
            'series_descriptions': tf.TensorSpec(shape=(self.max_subdirs,), dtype=tf.string),
            'label_coordinates': tf.TensorSpec(shape=(self.max_subdirs,), dtype=tf.string)
        },
        tf.TensorSpec(shape=(len(self.label_columns),), dtype=tf.int32)
        )

        dataset = tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=output_signature
        )

        return dataset

    def create_dataset(self, batch_size, study_id: Optional[list] = None ):
        """Create and configure the TensorFlow dataset for RSNA dataset .
        Input: none

        Return: tf.data.Dataset object
                Encoder dic
                The encoders returned from create_dataset is a dictionary that contains LabelEncoder objects for each label column in your dataset.
                The keys are the names of the label columns (e.g., 'spinal_canal_stenosis_l1_l2', 'left_neural_foraminal_narrowing_l1_l2', etc.)
                The values are the corresponding LabelEncoder objects for each column

                Encoder can be used to:

                Transform labels: encoders['spinal_canal_stenosis_l1_l2'].transform(['Moderate'])
                Inverse transform (get original labels): encoders['spinal_canal_stenosis_l1_l2'].inverse_transform([1])
                See the classes: encoders['spinal_canal_stenosis_l1_l2'].classes_

                Labels are 'Normal/Mild', 'Moderate', and 'Severe', a typical encoding will be:

                'Normal/Mild' -> 0
                'Moderate' -> 1
                'Severe' -> 2
        """
        # print shape and colmns of df
        print(f"Shape of the DataFrame: {self.df.shape}")
        print(f"Columns of the DataFrame: {self.df.columns}")
        
        # Identify columns with NaN values
        nan_columns = self.df.columns[self.df.isna().any()].tolist()

        # Handle NaN values
        for col in nan_columns:
            self.df[col] = self.df[col].fillna('Unknown')
            
        print("Analyzing NaN values in the dataset")
        nan_df = self.analyze_nan_values()
        
        # Handle Nan values. We will create a category of "Unknown" for NaN values.
        self.df[col] = self.df[col].fillna('Unknown')
        
        if not nan_df.empty:
            print("ERROR: NaN values found in the dataset.")
            print(f"Number of rows with NaN values: {len(nan_df)}")
            
            # Optionally, you can save this DataFrame to a CSV file
            nan_csv_path = 'rows_with_nan_values.csv'
            nan_df.to_csv(nan_csv_path, index=False)
            print(f"Rows with NaN values have been saved to {nan_csv_path}")
            
            print("Dataset creation process stopped. Please handle NaN values before proceeding.")
            return None, None, None  # Return None for train_dataset, val_dataset, and encoders
        
        print("No NaN values found. Proceeding with dataset creation.")
            
        print("Going to encode labels")
        encoders = self._create_label_encoders()
        print("Creating initial dataset")
        dataset = self._create_initial_dataset()
        dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)  # Adjust buffer_size as needed
        print("Splitting dataset")
        train_dataset, val_dataset = self._split_dataset(dataset)
        
        if batch_size:
            train_dataset = train_dataset.batch(batch_size)
            val_dataset = val_dataset.batch(batch_size)
        
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        print("Dataset created")
        return train_dataset, val_dataset, encoders

    def create_test_dataset(self):
        pass
    