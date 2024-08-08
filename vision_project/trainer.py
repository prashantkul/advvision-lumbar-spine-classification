import tensorflow as tf
from vision_models.utils import VisionUtils
from vision_models import constants
from vision_models.dataset import Dataset
from vision_models.densenetmodel import DenseNetVisionModel, ModelTrainer
import pickle
import os
class VisionModelPipeline:

    def __init__(self):
        self.vutil = VisionUtils()
        self.strategy = self._get_strategy()
        self.batch_size = 2 # change batch size to 24 for training. Batch greater than 24 will result in OOM error
        with self.strategy.scope():
            self.image_loader = Dataset(batch_size=self.batch_size)
            self.input_shape = (self.batch_size, 192, 224, 224, 3)  # Updated to include the slice dimension
            self.num_classes = 25
            self.weights = 'imagenet'
            self.epochs = 1

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
        
    def train_model(self, model, train_dataset, val_dataset, train_steps_per_epoch, validation_steps):
        with self.strategy.scope():
            trainer = ModelTrainer(model)
            
            # Check for existing checkpoints in the current directory
            checkpoint_dir = os.getcwd()
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.weights.h5')]
            
            # Sort the checkpoint files by epoch number
            checkpoint_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
            
            # Load the latest checkpoint if it exists
            if checkpoint_files:
                latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
                print(f"Loading existing checkpoint from {latest_checkpoint}")
                model.load_weights(latest_checkpoint)
        
            history = trainer.train(train_dataset, val_dataset, epochs=self.epochs, 
                                    steps_per_epoch=train_steps_per_epoch, validation_steps=validation_steps)
        return history
    
    
def main():
    pipeline = VisionModelPipeline()
    
    train_dataset = pipeline.load_data("train")
    val_dataset = pipeline.load_data("val")
    
    # calculate step sizes
    dataset_size_dict = pipeline.calculate_steps_per_epoch()
    training_steps_per_epoch = dataset_size_dict[constants.TRAIN] // pipeline.batch_size
    validation_steps_per_epoch = dataset_size_dict[constants.VAL] // pipeline.batch_size
    print(f"Training steps per epoch: {training_steps_per_epoch}")
    print(f"Validation steps per epoch: {validation_steps_per_epoch}")

    # Uncomment code below for training the model
    model = pipeline.build_model()
    history = pipeline.train_model(model, train_dataset, val_dataset, training_steps_per_epoch, validation_steps_per_epoch)            

if __name__ == "__main__":
    main()
