
import sys
import tensorflow as tf

class Utils:

   def print_sys_path():
      print(sys.path)
      
   def print_python_version():
      print("Python version: " + sys.version)

   def print_tf_version():
      print("Tensorflow version: " + tf.__version__)

   def check_gpu_support():
      print(tf.config.list_physical_devices('GPU'))

   