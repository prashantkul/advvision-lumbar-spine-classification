
import sys
import tensorflow as tf

import vision_models.constants as constants

class VisionUtils:
   def set_seed(self):
      tf.random.set_seed(constants.RANDOM_SEED)
      
   def print_sys_path(self):
      print(sys.path)
      
   def print_python_version(self):
      print("Python version: " + sys.version)

   def print_tf_version(self):
      print("Tensorflow version: " + tf.__version__)

   def print_tf_gpu_support(self):
      print(tf.config.list_physical_devices('GPU'))

   def print_dataset_info(self, dataset, print_labels):
    """Print information about a dataset."""
    for data, labels in dataset.take(1):  # Take 1 batch
      print("Images shape:", tf.shape(data['images']))
      print("Series descriptions shape:", tf.shape(data['series_descriptions']))
      print("Label coordinates shape:", tf.shape(data['label_coordinates']))
      if print_labels:
         print("Labels shape:", tf.shape(labels))

    # If you want to print shapes for each subdir tensor in 'images'
    for i, img_tensor in enumerate(data['images']):
        print(f"Subdir {i} image tensor shape:", tf.shape(img_tensor))

   def print_encoder_info(self, encoders):
      """Print information about the label encoders."""
      for col, encoder in encoders.items():
         print(f"\nEncoding for {col}:")
         for class_name, class_code in zip(encoder.classes_, encoder.transform(encoder.classes_)):
               print(f"{class_name}: {class_code}")
   