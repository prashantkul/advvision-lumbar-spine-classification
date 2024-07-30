import tensorflow as tf
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DenseNetVisionModel:
    def __init__(self, num_classes, input_shape=(224, 224, 3), weights='imagenet'):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.weights = weights
        self.model = self._build_model()

    def _build_model(self):
        base_model = DenseNet169(weights=self.weights, include_top=False, input_shape=self.input_shape)
        x = GlobalAveragePooling2D()(base_model.output)
        output = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)
        return model

    def freeze_base_model(self):
        for layer in self.model.layers[:-2]:
            layer.trainable = False

    def unfreeze_model(self, num_layers=20):
        for layer in self.model.layers[-num_layers:]:
            layer.trainable = True

    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_generator, validation_generator, epochs=10, class_weights=None):
        return self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            epochs=epochs,
            class_weight=class_weights
        )

    def evaluate(self, test_generator):
        return self.model.evaluate(test_generator)

    def predict(self, input_data):
        return self.model.predict(input_data)
