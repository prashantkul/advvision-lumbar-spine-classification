import tensorflow as tf
from keras import layers, models


class LumbarStenosisCNN:
    def __init__(self, input_shape=(200, 224, 224, 3), num_classes=25):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        initializer = tf.keras.initializers.HeNormal()  # He initialization often works well with ReLU
        
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        # Initial 3D Convolutional layer
        x = layers.Conv3D(32, kernel_size=(3, 3, 3), padding='same', kernel_initializer=initializer)(inputs)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        
        # Convolutional blocks
        for filters in [64, 128, 256, 512]:
            x = self._conv_block(x, filters, initializer)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling3D()(x)
        
        # Dense layers
        x = layers.Dense(512, kernel_initializer=initializer)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, kernel_initializer=initializer)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', kernel_initializer=initializer)(x)
        
        return models.Model(inputs=inputs, outputs=outputs)

    def _conv_block(self, x, filters, initializer):
        x = layers.Conv3D(filters, kernel_size=(3, 3, 3), padding='same', kernel_initializer=initializer)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(filters, kernel_size=(3, 3, 3), padding='same', kernel_initializer=initializer)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        return x

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        return self.model.summary()

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)