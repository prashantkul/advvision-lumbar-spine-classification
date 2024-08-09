import tensorflow as tf
from tensorflow.keras import layers, models

def create_classification_model(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flattening
    model.add(layers.Flatten())
    
    # Fully connected layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(3, activation='sigmoid'))
    
    return model

# Create the model
input_shape = (height, width, channels)
classification_model = create_classification_model(input_shape)

classification_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classification_model.summary()