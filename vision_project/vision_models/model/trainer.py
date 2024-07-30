import tensorflow as tf

class DenseNetVisionModelTrainer:
    def __init__(self, model, train_generator, validation_generator, test_generator, epochs, learning_rate, batch_size):
        self.model = model
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.test_generator = test_generator
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def train(self):
        # Compile the model
        self.model.compile_model(learning_rate=self.learning_rate)

        # Train the model
        history = self.model.train(
            self.train_generator,
            self.validation_generator,
            epochs=self.epochs
        )

        return history

    def evaluate(self):
        # Evaluate the model
        loss, accuracy = self.model.evaluate(self.test_generator)
        print(f'Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}')

    def save_model(self, model_path):
        # Save the model
        self.model.model.save(model_path)

    def load_model(self, model_path):
        # Load the model
        self.model.model = tf.keras.models.load_model(model_path)

# Usage
# model = DenseNetVisionModel(num_classes=3, input_shape=(224, 224, 1), weights='imagenet')
# train_generator, validation_generator, test_generator = DenseNetVisionModel.create_data_generators(
#     train_dir='train_images',
#     validation_dir='validation_images',
#     test_dir='test_images',
#     target_size=(224, 224),
#     batch_size=32
# )


# trainer = Trainer(
#     model=model,
#     train_generator=train_generator,
#     validation_generator=validation_generator,
#     test_generator=test_generator,
#     epochs=10,
#     learning_rate=0.001,
#     batch_size=32
# )


# history = trainer.train()
# trainer.evaluate()
# trainer.save_model('trained_model.h5')