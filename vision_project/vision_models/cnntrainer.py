import tensorflow as tf

from vision_models.cnnmodel import LumbarStenosisCNN
from vision_models.dataset import Dataset

class Trainer:
    def __init__(self, model, dataset, batch_size=8):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.strategy = self._get_strategy()
        
        with self.strategy.scope():
            self.image_loader = Dataset(batch_size=self.batch_size)
            self.input_shape = (self.batch_size, 200, 224, 224, 3)  # Updated to include the slice dimension
            self.num_classes = 25
            self.epochs = 2
        
        self.dataset_sizes = self.dataset.get_df_sizes()

    
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
    
    def _calculate_steps(self, split):
        return self.dataset_sizes[split] // self.batch_size
    
    def load_data(self, mode, study_ids: list[str] = None):
        print("Creating datasets...")
        dataset = self.image_loader.load_data(mode)

        if self.strategy is tf.distribute.OneDeviceStrategy(device="/cpu:0"):
            return dataset
        else:
            return self.strategy.experimental_distribute_dataset(dataset)

    def train(self, epochs=1):
        train_data = self.dataset.load_data('train')
        val_data = self.dataset.load_data('val')
        
        # Calculate steps per epoch
        train_steps = self._calculate_steps('train')
        val_steps = self._calculate_steps('val')
        
        # Define callbacks
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, min_delta=0.001, verbose=1
        )

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath="best_model.weights.{epoch:03d}-{val_accuracy:.4f}.weights.h5",
            save_weights_only=True,  # Save only the weights (not the entire model)
            save_freq='epoch' # Save every epoch
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=0.00001, verbose=1
        )
        
        callbacks = [early_stopping, model_checkpoint, reduce_lr]
        
        # Train the model
        history = self.model.fit(
            train_data,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_data=val_data,
            validation_steps=val_steps,
            callbacks=callbacks
        )

        return history

    def evaluate(self):
        test_data = self.dataset.load_data('test')
        test_steps = len(self.dataset.test_df) // self.batch_size
        test_loss, test_accuracy = self.model.evaluate(test_data, steps=test_steps)
        print(f"Test accuracy: {test_accuracy:.4f}")
        return test_loss, test_accuracy

# Example usage:
if __name__ == "__main__":
    # Assuming Dataset class is imported and available
    dataset = Dataset(batch_size=12)
    
    # Create and compile the model
    cnn_model = LumbarStenosisCNN()
    cnn_model.compile()
    cnn_model.summary()

    # Create trainer and train the model
    trainer = Trainer(cnn_model, dataset)
    history = trainer.train(epochs=trainer.epochs)

    # Evaluate the model
    test_loss, test_accuracy = trainer.evaluate()