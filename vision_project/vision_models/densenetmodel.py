import tensorflow as tf
from keras.models import Model
from tqdm.keras import TqdmCallback
from keras.applications import DenseNet121
from keras.optimizers import Adam

class ModelTrainer:
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

        self.callbacks = [early_stopping, model_checkpoint, reduce_lr]
        
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
    
class DenseNetVisionModel(tf.keras.Model):
    def __init__(self, num_classes, input_shape, weights='imagenet'):
        super(DenseNetVisionModel, self).__init__()
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

        self.base_model = tf.keras.applications.DenseNet121(
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

        super(DenseNetVisionModel, self).build(input_shape)

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