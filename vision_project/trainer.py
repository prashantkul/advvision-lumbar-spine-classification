import tensorflow as tf
from vision_models.utils import VisionUtils
from vision_models import constants
from vision_models.imageloader import ImageLoader
from vision_models.densenetmodel import DenseNetVisionModel, ModelTrainer

class VisionModelPipeline:

    def __init__(self):
        self.vutil = VisionUtils()
        self.strategy = self._get_strategy()
        self.batch_size = 1
        with self.strategy.scope():
            self.image_loader = ImageLoader(
                label_coordinates_csv=constants.TRAIN_LABEL_CORD_PATH,
                labels_csv=constants.TRAIN_LABEL_PATH,
                image_dir=constants.TRAIN_DATA_PATH,
                roi_size=(224, 224),
                batch_size=self.batch_size
            )
            self.input_shape = (self.batch_size, 192, 224, 224, 3)  # Updated to include the slice dimension
            self.num_classes = 25
            self.weights = 'imagenet'
            self.epochs = 5

    def _get_strategy(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) > 1:
            print(f"Using MirroredStrategy with {len(gpus)} GPUs")
            return tf.distribute.MirroredStrategy()
        elif len(gpus) == 1:
            print("Using single GPU")
            return tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            print("Using CPU")
            return tf.distribute.OneDeviceStrategy(device="/cpu:0")

    def setup_environment(self):
        print("*" * 50)
        print("Environment:    \n")
        self.vutil.set_seed()
        self.vutil.print_python_version()
        self.vutil.print_tf_version()
        self.vutil.print_tf_gpu_support()
        print("*" * 50)

    def load_data(self, mode, study_ids: list[str] = None):
        print("Creating datasets...")
        dataset = self.image_loader.load_data(mode, study_ids)
        
        return dataset

    def build_model(self):
        with self.strategy.scope():
            model = DenseNetVisionModel(num_classes=self.num_classes, input_shape=self.input_shape, weights=self.weights)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
           # Build the model with a sample input
            sample_input = tf.keras.Input(shape=self.input_shape)
            model(sample_input)
            
        return model

    def train_model(self, model, train_dataset, val_dataset):
        with self.strategy.scope():
            trainer = ModelTrainer(model)
            history = trainer.train(train_dataset, val_dataset, epochs=self.epochs)
        return history

def main():
    pipeline = VisionModelPipeline()
    #pipeline.setup_environment()


if __name__ == "__main__":
    main()
