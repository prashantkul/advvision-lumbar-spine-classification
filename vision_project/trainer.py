import tensorflow as tf
from vision_models.utils import VisionUtils
from vision_models import constants
from vision_models.imageloader import ImageLoader
from vision_models.densenetmodel import DenseNetVisionModel, ModelTrainer

class VisionModelPipeline:

    def __init__(self):
        self.vutil = VisionUtils()
        self.strategy = self._get_strategy()
        with self.strategy.scope():
            self.image_loader = ImageLoader(
                label_coordinates_csv=constants.TRAIN_LABEL_CORD_PATH,
                labels_csv=constants.TRAIN_LABEL_PATH,
                image_dir=constants.TRAIN_DATA_PATH,
                roi_size=(224, 224),
                batch_size=32
            )
            self.input_shape = (224, 224, 3)
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
        print("   Environment    \n")
        self.vutil.set_seed()
        self.vutil.print_python_version()
        self.vutil.print_tf_version()
        self.vutil.print_tf_gpu_support()
        print("*" * 50)

    def load_data(self):
        print("Creating datasets...")
        train_dataset, val_dataset = self.image_loader.load_data()
        print("Printing dataset info...")
        for img, labels in train_dataset.take(1):
            print(img.shape)
            print(labels.shape)
        return train_dataset, val_dataset

    def build_model(self):
        with self.strategy.scope():
            model = DenseNetVisionModel(num_classes=self.num_classes, input_shape=self.input_shape, weights=self.weights)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            sample_input = tf.keras.Input(shape=self.input_shape)
            model(sample_input)  # Build the model with a sample input
        return model

    def train_model(self, model, train_dataset, val_dataset):
        with self.strategy.scope():
            trainer = ModelTrainer(model)
            history = trainer.train(train_dataset, val_dataset, epochs=self.epochs)
        return history

def main():
    pipeline = VisionModelPipeline()
    pipeline.setup_environment()
    train_dataset, val_dataset = pipeline.load_data()
    model = pipeline.build_model()
    history = pipeline.train_model(model, train_dataset, val_dataset)
    print(history)

if __name__ == "__main__":
    main()