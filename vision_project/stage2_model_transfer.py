import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model

class EfficientNetClassifier(Model):
    def _init_(self, num_classes=1000):
        super(EfficientNetClassifier, self)._init_()
        # Load the pre-trained EfficientNetB0 model
        self.efficientnet = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        
        # Add global average pooling and a dense layer for classification
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.efficientnet(x, training=False)  # Set training=False to avoid updating batch norm layers
        x = self.global_avg_pool(x)
        x = self.fc(x)
        return x

# Example usage
model = EfficientNetClassifier(num_classes=1000)
model.build((None, 224, 224, 3))  # Build the model with the input shape
model.summary()