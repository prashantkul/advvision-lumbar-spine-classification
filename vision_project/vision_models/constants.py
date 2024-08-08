""" Constants definitins for the project """
DATASET_BASE_PATH = "/opt/dataset/"
TRAIN_DATA_PATH = "/opt/dataset/train_images/"
TRAIN_LABEL_PATH = "/opt/dataset/train.csv"
TRAIN_LABEL_CORD_PATH = "/opt/dataset/train_label_coordinates.csv"
TRAIN_SERIES_DESC_PATH = "/opt/dataset/train_series_descriptions.csv"
TEST_DATA_PATH = "/opt/dataset/test_images/"
TEST_SERIES_DESC_PATH = "/opt/dataset/test_series_descriptions.csv"
DENSENET_MODEL = "/opt/vision_models/stage_1/best_model.weights.h5"
RANDOM_SEED = 44
IMAGE_SIZE_HEIGHT = 224
IMAGE_SIZE_WIDTH = 224
IMAGE_CHANNELS = 3
EPOCHS = 1
BATCH_SIZE = 4
SHUFFLE_BUFFER_SIZE=10
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
TRAIN_SAMPLE_RATE = 0.4
DISEASE_THRESHOLD = 0.7

