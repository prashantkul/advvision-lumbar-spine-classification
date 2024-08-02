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
        self.model.compile(
            optimizer=Adam(),  # Using Adam optimizer with default settings
            loss="binary_crossentropy",
            metrics=["categorical_accuracy"],
        )

    def train(
        self,
        train_generator,
        validation_generator,
        epochs=1,
        class_balancing_weights=None,
    ):
        # Define callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, min_delta=0.001, verbose=1
        )

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath="best_model.keras",
            monitor="val_categorical_accuracy",  # Changed to match the metric name
            save_best_only=True,
            mode="max",
            verbose=1,
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=0.00001, verbose=1
        )

        self.callbacks = [early_stopping, model_checkpoint, reduce_lr]

        # Train the model
        if class_balancing_weights is not None:
            return self.model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                class_weight=class_balancing_weights,
                callbacks=self.callbacks,
            )
        else:
            return self.model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                callbacks=self.callbacks,
            )

class DenseNetVisionModel(tf.keras.Model):
    def __init__(self, num_classes, input_shape, weights='imagenet'):
        super(DenseNetVisionModel, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape[2:]  # (224, 224, 3)
        print(f"Input shape: {self.input_shape}")

        self.base_model = DenseNet121(
            include_top=False, 
            weights=weights, 
            input_shape=self.input_shape
        )
        self.base_model.trainable = False
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.prediction_layer = tf.keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        print(f"Input shape in call: {inputs.shape}")
        # Reshape to (batch_size * 192, 224, 224, 3)
        batch_size, num_images = inputs.shape[0], inputs.shape[1]
        x = tf.reshape(inputs, (-1,) + self.input_shape)
        print(f"Shape after reshape: {x.shape}")
        
        x = self.base_model(x)
        print(f"Shape after base_model: {x.shape}")
        x = self.global_average_layer(x)
        print(f"Shape after global_average_layer: {x.shape}")
        
        # Reshape back to (batch_size, 192, feature_dim)
        x = tf.reshape(x, (batch_size, num_images, -1))
        print(f"Shape after reshape back: {x.shape}")
        
        # Global average pooling over the 192 images
        x = tf.reduce_mean(x, axis=1)
        print(f"Shape after pooling over images: {x.shape}")
        
        x = self.prediction_layer(x)
        print(f"Shape after prediction_layer: {x.shape}")
        return x

    def build(self, input_shape):
        super(DenseNetVisionModel, self).build(input_shape)
        self.built = True

    def freeze_base_model(self):
        self.base_model.trainable = False

    def unfreeze_model(self, num_layers=20):
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-num_layers]:
            layer.trainable = False

    def evaluate(self, validation_generator):
        return super().evaluate(validation_generator)

    def predict(self, test_generator):
        return super().predict(test_generator)

# Usage example:
# input_shape = (None, 192, 224, 224, 3)  # None for batch size
# num_classes = 25

# model = DenseNetVisionModel(num_classes, input_shape, weights='imagenet')
# trainer = ModelTrainer(model)

# # Assuming you have your train_generator and validation_generator ready
# # trainer.train(train_generator, validation_generator, epochs=10)