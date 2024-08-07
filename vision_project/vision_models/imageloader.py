import os
import pandas as pd
import numpy as np
import pydicom
import cv2
import tensorflow as tf
import vision_models.constants as constants
from typing import Any, Iterator, Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    
    def visualize_attention(self, original_img, attended_img, x, y, filename):
        """
        Visualize the original and attended images side by side.
        
        Args:
        original_img (np.array): Original image
        attended_img (np.array): Image after applying Gaussian attention
        x, y (float): Coordinates of the attention center
        filename (str): Name of the file for saving the visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        print(f"Original image shape: {original_img.shape}")
        print(f"Attended image shape: {attended_img.shape}")
        
        ax1.imshow(original_img, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        ax1.plot(x, y, 'r+', markersize=10)  # Mark the attention center
        
        ax2.imshow(attended_img, cmap='gray')
        ax2.set_title('Image with Gaussian Attention')
        ax2.axis('off')
        ax2.plot(x, y, 'r+', markersize=10)  # Mark the attention center
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def apply_gaussian_attention(self, image, x, y, attention_size_ratio=0.5, sigma_ratio=0.3):
        """
        Apply Gaussian attention to an image based on x, y coordinates.
        
        Args:
        image (np.array): Original image array
        x (float): x-coordinate in the original image space
        y (float): y-coordinate in the original image space
        attention_size_ratio (float): Ratio of image size to use for attention area
        sigma_ratio (float): Ratio of attention size to use for sigma
        
        Returns:
        np.array: Image with Gaussian attention applied
        """
        height, width = image.shape[:2]
        
        # Normalize coordinates
        norm_x = x / width
        norm_y = y / height
        
        # Calculate attention area size (increase this for a wider attention area)
        attention_size = int(min(height, width) * attention_size_ratio)
        sigma = attention_size * sigma_ratio
        
        # Create coordinate grid for the entire image
        y_grid, x_grid = np.ogrid[:height, :width]
        y_grid = (y_grid - y) / sigma
        x_grid = (x_grid - x) / sigma
        
        # Calculate Gaussian mask for the entire image
        mask = np.exp(-(x_grid**2 + y_grid**2) / 2)
        
        # Normalize the mask to [0.2, 1] range to maintain some visibility across the entire image
        mask = 0.2 + 0.8 * (mask - mask.min()) / (mask.max() - mask.min())
        
        # Apply mask to image
        attended_img = image * mask[:,:,np.newaxis] if len(image.shape) == 3 else image * mask
        
        return attended_img

    def _preprocess_image(self, study_id, series_id, x, y, print_images=False):
        """ Preprocess images for a given study and series. """
        
        #print(f"Preprocessing images for study_id: {study_id}, series_id: {series_id}")
        #print(f"Applying Gaussian attention at coordinates: ({x}, {y})")
        
        series_dir = f"{self.image_dir}/{study_id}/{series_id}"
        #print(f"Reading images from {series_dir}")
        images = []           
        
        # Get sorted list of DICOM files
        dicom_files = sorted([f for f in os.listdir(series_dir) if f.endswith(".dcm")])
        
        for idx, filename in enumerate(dicom_files):
            file_path = os.path.join(series_dir, filename)
            original_img = self._read_dicom(file_path)
            
            # Apply Gaussian attention to original image using the single x and y for all images
            attended_img = self.apply_gaussian_attention(original_img, x, y)
            
            # # Visualize (for the first 5 images in the series)
            # if print_images:
            #     if idx < 5:
            #         current_dir = os.getcwd()
            #         vis_dir = os.path.join(current_dir, "visualizations", f"{study_id}_{series_id}")
            #         os.makedirs(vis_dir, exist_ok=True)
            #         vis_filename = os.path.join(vis_dir, f"attention_vis_{idx}.png")
            #         self.visualize_attention(original_img, attended_img, x, y, vis_filename)
            
            # Convert to tensor and preprocess
            img = tf.convert_to_tensor(attended_img, dtype=tf.float32)
            img = tf.expand_dims(img, axis=-1)  # Add channel dimension
            img = tf.image.resize(img, self.roi_size)
            img = tf.image.grayscale_to_rgb(img)  # Convert to RGB
            
            images.append(img)

        # Pad images to 192 if necessary
        #print(f"Number of images in series: {len(images)}")
        if len(images) < 192:
            #print(f"Padding tensor to 192 images")
            padding = tf.zeros((192 - len(images), *self.roi_size, 3), dtype=tf.float32)
            images = tf.concat([images, padding], axis=0)

        result = tf.stack(images)
        #print(f"Resulting preprocessed image tensor shape: {result.shape}")
        return result


    def _create_split(self, data):
        """
        Create a split of the data based on the composite key.

        Args:
            data (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with a new 'split' column indicating the train, val, or test set.
        """        
        # Combine condition and level to form a unique class identifier
        data['class'] = data['condition'] + '_' + data['level']

        # Create a composite key for stratification
        data['composite_key'] = data['study_id'].astype(str) + '_' + \
                                data['series_id'].astype(str) + '_' + \
                                data['condition'] + '_' + data['level']

        # Calculate the class distribution based on the composite key
        class_distribution = data['composite_key'].value_counts()

        # Perform stratified sampling based on the composite key
        train_ids, test_ids = train_test_split(
            class_distribution.index,
            test_size=0.2,
            stratify=class_distribution.values,
            random_state=42
        )

        train_ids, val_ids = train_test_split(
            train_ids,
            test_size=0.25,  # 0.25 of the 0.8 train set size results in 0.2 validation set size
            stratify=class_distribution[train_ids].values,
            random_state=42
        )

        # Split the data based on these IDs
        train_split = data[data['composite_key'].isin(train_ids)].copy()
        val_split = data[data['composite_key'].isin(val_ids)].copy()
        test_split = data[data['composite_key'].isin(test_ids)].copy()

        # Assign split labels
        train_split['split'] = 'train'
        val_split['split'] = 'val'
        test_split['split'] = 'test'

        # Concatenate all splits
        split_data = pd.concat([train_split, val_split, test_split])

        # Drop the temporary columns
        split_data.drop(columns=['class', 'composite_key'], inplace=True)

        print("Train Set Shape:", train_split.shape)
        print("Validation Set Shape:", val_split.shape)
        print("Test Set Shape:", test_split.shape)
        
        # Save splits to CSV files
        train_split.to_csv(f'train_split.csv', index=False)
        val_split.to_csv(f'val_split.csv', index=False)
        test_split.to_csv(f'test_split.csv', index=False)

        return split_data
            
    def _analyze_splits(self):
        """ Analyze the splits of the data. """
        label_coordinates_df = pd.read_csv(self.label_coordinates_csv)
        # split the dataframe for train, test and validation
        label_coordinates_df = self._create_split(label_coordinates_df)

    def _create_human_readable_label(self, label_vector, label_list):
        """ Create human-readable labels from one-hot encoded vector. """
        # Convert label_vector to numpy array if it's not already
        label_vector = np.array(label_vector)
        # Remove the [0] indexing if label_vector is 1D
        indices_with_ones = np.where(label_vector == 1)[0]
        # Select the corresponding labels from label_list
        human_readable_labels = [label_list[i] for i in indices_with_ones]
        # Output the labels
        #print("Human-readable labels:", human_readable_labels)
    
    def _filter_df(self, df, study_ids):
        print(f"Label coordinates DF will be filtered based on study_ids: {study_ids}")
        self_study_ids = list(map(int, self.study_ids))
        df = df[df["study_id"].isin(self_study_ids)]
        print(f"Label coordinates DF filtered based on study_ids, new shape: {df.shape}")
        return df

    def _sample_dataframe(self, df, fraction=0.4):
        """
        Sample the dataframe while maintaining the proportions of each split.
        
        Args:
        df (pd.DataFrame): Input dataframe
        fraction (float): Fraction of data to sample (default: 0.4)
        
        Returns:
        pd.DataFrame: Sampled dataframe
        """
        print("\nBefore sampling:")
        for split in df['split'].unique():
            split_df = df[df['split'] == split]
            print(f"{split} split - Size: {len(split_df)}, Shape: {split_df.shape}")
        
        sampled_df = df.groupby('split', group_keys=False).apply(lambda x: x.sample(frac=fraction, random_state=42))
        
        print("\nAfter sampling:")
        for split in sampled_df['split'].unique():
            split_df = sampled_df[sampled_df['split'] == split]
            print(f"{split} split - Size: {len(split_df)}, Shape: {split_df.shape}")
    
        return sampled_df
    
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
        
        # Create a combined label column, this should match the header of the labels.csv file
        label_coordinates_df["label"] = label_coordinates_df.apply(
            lambda row: (
                row["condition"].replace(" ", "_")
                + "_"
                + row["level"].replace(" ", "_").replace("/", "_")
            ).lower(),
            axis=1,
        )
        
        # filter the label_corrdinates_df based on the study_ids
        if self.study_ids:
            print("Available Study IDs in DataFrame:", len(label_coordinates_df['study_id'].unique()))
            label_coordinates_df = self._filter_df(label_coordinates_df, self.study_ids)
            if len(label_coordinates_df) == 0:
                print("No rows found for the study_ids provided")

        # Extract unique labels and store them from train.csv, we will match generated labels against this list
        self.label_list = pd.read_csv(self.labels_csv).columns[1:].tolist()

        # Create a new split column in the dataframe for train, test and validation split.
        label_coordinates_df = self._create_split(label_coordinates_df)
        
        # Sample 40% of the data
        #label_coordinates_df = self._sample_dataframe(label_coordinates_df, fraction=constants.TRAIN_SAMPLE_RATE)
    
        # Filter the dataframe based on the split requested in the generator, only return those rows
        if self.split in ["train", "val", "test"]:
            label_coordinates_df = label_coordinates_df[
                label_coordinates_df["split"] == self.split]
        
        # Shuffle if the split is "train"
        if self.split == "train":
            label_coordinates_df = label_coordinates_df.sample(frac=1, random_state=42).reset_index(drop=True)
           
        # This loop iterates until all the rows have beene exahused for the generator
        while True:
            if len(label_coordinates_df) == 0:
                print(f"All {self.split} samples processed. Raising StopIteration.")
                raise StopIteration
            
            #print("*" * 100)
            
            # randomly select one row from the dataframe to return
            row = label_coordinates_df.sample(n=1)
            
            # Remove the selected row from the DataFrame
            label_coordinates_df = label_coordinates_df.drop(row.index) 

            study_id = row["study_id"].values[0]
            series_id = row["series_id"].values[0]
            condition = row["condition"].values[0]
            level = row["level"].values[0]
            x = row["x"].values[0]
            y = row["y"].values[0]

            # print(
            #     f"Going to generate feature for study_id: {study_id}, series_id: {series_id}, condition: {condition}, level: {level}"
            # )

            # Preprocess the image before supplying to the generator.
            img_tensor = self._preprocess_image(study_id, series_id, x , y)
            # print(
            #     f"Feature tensor generated, size: {img_tensor.shape}, now generating label"
            # )
            
            # Create a unique label for the combination of study_id, series_id, condition, and level
            label = f"{row['condition'].values[0].replace(' ', '_').lower()}_{row['level'].values[0].replace('/', '_').lower()}"
            try:
                label_vector = self.label_list.index(label)
            except ValueError:
                raise ValueError(f"Label {label} not found in the label list")
           # print(f"Label generated")
            # Create a one-hot encoded vector
            one_hot_vector = [0.0] * len(self.label_list)
            one_hot_vector[label_vector] = 1.0

            #print(f"One hot enocded label vector generated: {one_hot_vector}")
            self._create_human_readable_label(one_hot_vector, self.label_list)

            one_hot_vector_array = np.array(one_hot_vector, dtype=np.float32)
            #print("Returning feature and label tensors")
          
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
            raise ValueError(f"Unknown split: {split}, correct values are: train, val, test")
        
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
