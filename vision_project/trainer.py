import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

import tensorflow as tf
from vision_models.utils import VisionUtils
from vision_models import constants
from vision_models.dataset import Dataset
from vision_models.densenetmodel import DenseNetVisionModel, ModelTrainer
class VisionModelPipeline:

    def __init__(self):
        self.vutil = VisionUtils()
        self.strategy = self._get_strategy()
        self.batch_size = 12 # change batch size to 24 for training if using A100 40GB. On T4, set to 12.
        with self.strategy.scope():
            self.image_loader = Dataset(batch_size=self.batch_size)
            self.input_shape = (self.batch_size, 192, 224, 224, 3)  # Updated to include the slice dimension
            self.num_classes = 25
            self.weights = 'imagenet'
            self.epochs = 2

    def _get_strategy(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) > 1:
            print(f">> Using MirroredStrategy with {len(gpus)} GPUs \n")
            return tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
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
        dataset = self.image_loader.load_data(mode)

        if self.strategy is tf.distribute.OneDeviceStrategy(device="/cpu:0"):
            return dataset
        else:
            return self.strategy.experimental_distribute_dataset(dataset)

    def build_model(self):
        with self.strategy.scope():
            model = DenseNetVisionModel(
                num_classes=self.num_classes,
                input_shape=self.input_shape,
                weights=self.weights,
            )
            # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Build the model with a sample input
            sample_input = tf.keras.Input(shape=self.input_shape)
            model(sample_input)

        return model

    def train_model(self, model, train_dataset, val_dataset):
        with self.strategy.scope():
            print("Starting training...")
            trainer = ModelTrainer(model)
            history = trainer.train(train_dataset, val_dataset, epochs=self.epochs)
        return history

    def _print_distributed_dataset(self, dataset: tf.distribute.DistributedDataset 
                                , num_elements=1):
        iterator = iter(dataset)
        for _ in range(num_elements):
            try:
                distributed_element = next(iterator)
                elements = self.strategy.experimental_local_results(distributed_element)

                # Handle the case where elements is a tuple of tuples
                if isinstance(elements[0], tuple):
                    img, label = elements[0]
                else:
                    img, label = elements

                print("Image shape:", img.shape)
                print("Label shape:", label.shape)
                print("Label:", label)
                print("-" * 50)
            except StopIteration:
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Element structure:", elements)
                break

    def calculate_steps_per_epoch(self):
        return self.image_loader.get_df_sizes()

    def _get_strategy(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) > 1:
            print(f"Using MirroredStrategy with {len(gpus)} GPUs")
            return tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        elif len(gpus) == 1:
            print("Using single GPU")
            return tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            print("Using CPU")
            return tf.distribute.OneDeviceStrategy(device="/cpu:0")

    def train_model(self, model, train_dataset, val_dataset, 
                    train_steps_per_epoch, validation_steps, epochs = None):
        print(">> Running full training...")
        with self.strategy.scope():

            if epochs is None:
                epochs = self.epochs

            trainer = ModelTrainer(model)

            # Check for existing checkpoints in the current directory
            checkpoint_dir = os.getcwd()
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                                if f.startswith('best_model') and f.endswith('.weights.h5')]

            # Sort the checkpoint files by epoch number
            #checkpoint_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))

            # Load the latest checkpoint if it exists
            if checkpoint_files:
                latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
                print(f"Loading existing checkpoint from {latest_checkpoint}")
                model.load_weights(latest_checkpoint)

            history = trainer.train(train_dataset, val_dataset, epochs=self.epochs, 
                                    steps_per_epoch=train_steps_per_epoch, validation_steps=validation_steps)
        return history

    def _run_quick_training(self, model, train_dataset, val_datset):
        print(">> Running a quick training session...")
        # Small dataset generator
        small_train_generator = train_dataset.take(100)  # Take 100 samples for quick testing
        small_val_generator = val_datset.take(20)      # Take 20 samples for quick validation

        steps_per_epoch = 100 // self.batch_size
        validation_steps = 20 // self.batch_size

        # Run a quick training session
        self.train_model(model, 
            train_dataset = small_train_generator,
            val_dataset = small_val_generator,
            train_steps_per_epoch = 8,
            validation_steps= 2,
            epochs = 2   
        )


def main():
    pipeline = VisionModelPipeline()
    image_loader = Dataset(batch_size=12)

    
    #calculate step sizes
    dataset_size_dict = pipeline.calculate_steps_per_epoch()
    training_steps_per_epoch = dataset_size_dict[constants.TRAIN] // pipeline.batch_size
    validation_steps_per_epoch = dataset_size_dict[constants.VAL] // pipeline.batch_size
    print(f"\n Training steps per epoch:    {training_steps_per_epoch}")
    print(f"   Validation steps per epoch:  {validation_steps_per_epoch} \n")
    print("*" * 100)

    train_dataset = image_loader.load_data(constants.TRAIN)
    val_dataset =   image_loader.load_data(constants.VAL)

    # Uncomment code below for training the model
    model = pipeline.build_model()
    history = pipeline.train_model(model, train_dataset, val_dataset, training_steps_per_epoch, validation_steps_per_epoch)  
    #history = pipeline._run_quick_training(model, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
