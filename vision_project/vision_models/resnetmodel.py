import tensorflow as tf
from keras.models import Model
from tqdm.keras import TqdmCallback
#from keras.applications.resnet101 import ResNet101
from keras.applications import ResNet101
from keras.optimizers import Adam

from tensorflow.keras.callbacks import ReduceLROnPlateau

class BatchReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, monitor='auc', factor=0.5, patience=100, verbose=1, mode='max', min_delta=1e-4, cooldown=50, min_lr=1e-6, **kwargs):
        super(BatchReduceLROnPlateau, self).__init__(
            monitor=monitor, factor=factor, patience=patience, verbose=verbose, mode=mode,
            min_delta=min_delta, cooldown=cooldown, min_lr=min_lr, **kwargs
        )
        self.batch_count = 0
    
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batch_count += 1
        
        batch_auc = logs.get(self.monitor)
        if batch_auc is None:
            return
        
        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0
        
        if self.monitor_op(batch_auc - self.min_delta, self.best):
            self.best = batch_auc
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose > 0:
                        print(f'\nBatch {self.batch_count}: reducing learning rate to {new_lr:.6f}.')
                    self.cooldown_counter = self.cooldown
                    self.wait = 0

class ResNetModelTrainer:
    def __init__(self, model):
        self.model = model
        # Compile the model in the constructor
        self.compile_model()

    def compile_model(self):
        # Separate method for model compilation
        print("Compiling the model...")
        self.model.compile(
            optimizer=Adam(),  # Using Adam optimizer with default settings
            loss="binary_crossentropy",
            metrics=["binary_accuracy",                 
                     tf.keras.metrics.AUC(multi_label=True, num_labels=self.model.num_classes),
                    ],
        )

    def train(
        self,
        train_generator,
        validation_generator,
        epochs=1,
        steps_per_epoch=None,
        validation_steps=None,
        class_balancing_weights=None,
        load_checkpoint=None
    ):
        print(f"{'*'*20} Training the model {'*'*20}")        
        # Define callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, min_delta=0.001, verbose=1
        )

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath="best_model.weights.h5",
            save_weights_only=True,  # Save only the weights (not the entire model)
            save_freq='epoch' # Save every epoch
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=0.00001, verbose=1
        )

        # Create the callback to monitor AUC within an epoch. 
        batch_reduce_lr = BatchReduceLROnPlateau(
            monitor='auc',  # Monitor the AUC metric
            factor=0.5,     # Reduce LR by half when triggered
            patience=300,    # Wait for 300 batches before reducing LR
            verbose=1,
            mode='max',     # We want to maximize AUC
            min_delta=1e-4, # Minimum change to qualify as an improvement
            cooldown=150,   # Wait for 150 batches after each LR reduction before resuming monitoring
            min_lr=1e-6     # Don't reduce LR below this value
        )

        self.callbacks = [early_stopping, model_checkpoint, reduce_lr, batch_reduce_lr]
        
        if load_checkpoint:
            self.model.load_weights(load_checkpoint)
            print(f"Loaded checkpoint from {load_checkpoint}")

       # Train the model
        fit_args = {
            'x': train_generator,
            'validation_data': validation_generator,
            'epochs': epochs,
            'callbacks': self.callbacks,
            'steps_per_epoch': steps_per_epoch,
            'validation_steps': validation_steps
        }

        if class_balancing_weights is not None:
            fit_args['class_weight'] = class_balancing_weights

        return self.model.fit(**fit_args)
    
class ResNetVisionModel(tf.keras.Model):
    def __init__(self, num_classes, input_shape, weights='imagenet'):
        super(ResNetVisionModel, self).__init__()
        print(f"Input shape received to the init method: {input_shape}")
        self.num_classes = num_classes
        
        if len(input_shape) == 5:  # (batch, 192, 224, 224, 3)
            self.input_shape = input_shape[1:]  # (192, 224, 224, 3)
        elif len(input_shape) == 4:  # (192, 224, 224, 3)
            self.input_shape = input_shape
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")
        
        self.slices = self.input_shape[0]
        self.image_shape = self.input_shape[1:]
        
        print(f"Input shape for base model: {self.image_shape}")

        self.base_model = tf.keras.applications.ResNet101(
            include_top=False, 
            weights=weights, 
            input_shape=self.image_shape
        )
        self.base_model.trainable = False

        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        
        # We'll create the prediction_layer in the build method
        self.prediction_layer = None

    def build(self, input_shape):
        # Call the base model on a sample input to get the output shape
        sample_input = tf.keras.Input(shape=self.image_shape)
        sample_output = self.base_model(sample_input)
        sample_output = self.global_average_layer(sample_output)
        
        # Now we know the shape of the flattened features
        feature_shape = sample_output.shape[-1]
        
        # Create the prediction layer with the known input shape
        self.prediction_layer = tf.keras.layers.Dense(self.num_classes, activation='sigmoid')
        # Build the prediction layer
        self.prediction_layer.build((None, feature_shape))

        super(ResNetVisionModel, self).build(input_shape)

    def call(self, inputs, training=False):
        # Get the shape of the input
        batch_size = tf.shape(inputs)[0]
        
        # Reshape to (batch_size * slices, height, width, channels)
        x = tf.reshape(inputs, [-1] + list(self.image_shape))
        
        x = self.base_model(x, training=training)
        x = self.global_average_layer(x)
        
        # Reshape back to (batch_size, slices, features)
        x = tf.reshape(x, [batch_size, self.slices, -1])
        
        # Global average pooling over the slices
        x = tf.reduce_mean(x, axis=1)
        
        return self.prediction_layer(x)

    def model(self):
        x = tf.keras.Input(shape=self.input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x, training=False))

# Usage example:
# input_shape = (None, 192, 224, 224, 3)  # None for batch size
# num_classes = 25

# model = DenseNetVisionModel(num_classes, input_shape, weights='imagenet')
# trainer = ModelTrainer(model)

# # train the mode using:
# # trainer.train(train_generator, validation_generator, epochs=10)