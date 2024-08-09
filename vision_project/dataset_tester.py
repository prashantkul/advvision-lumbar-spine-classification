import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

from vision_models import dataset
import tensorflow as tf

# creat instance of the dataset class
dataset_instance = dataset.Dataset(batch_size = 1)

# load the dataset
dataset, steps_per_epoch  = dataset_instance.load_data("test")

# print

print("Steps per epoch: ", steps_per_epoch)

# Create an iterator
iterator = iter(dataset)

try:
    element = dataset.take(1)
    for img, label in element:
        print("Image tensor shape: ", img.shape)  
        print("Label tensor shape: ", label.shape)
        break  
except tf.errors.OutOfRangeError:
    print("End of dataset")