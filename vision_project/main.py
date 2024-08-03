from vision_models.imageloader import ImageLoader
import vision_models.constants as constants

# # # Initialize ImageLoader
# # image_loader = ImageLoader(
# #     image_dir=constants.TRAIN_DATA_PATH,
# #     label_coordinates_csv=constants.TRAIN_LABEL_CORD_PATH,
# #     labels_csv=constants.TRAIN_LABEL_PATH,
# #     roi_size=(constants.IMAGE_SIZE_HEIGHT, constants.IMAGE_SIZE_WIDTH),
# #     batch_size=constants.BATCH_SIZE
# # )

# # # Load dataset and split
# # train_dataset, val_dataset, test_dataset = image_loader.split_dataset()

# # # Analyze the split
# # image_loader.analyze_split()


import vision_models.constants as constants
from vision_models.imageloader import ImageLoader
from vision_models.densenetmodel import DenseNetVisionModel, ModelTrainer

# def main():
#     # Initialize the ImageLoader
#     image_loader = ImageLoader(
#         image_dir=constants.TRAIN_DATA_PATH,
#         label_coordinates_csv=constants.TRAIN_LABEL_CORD_PATH,
#         labels_csv=constants.TRAIN_LABEL_PATH,
#         roi_size=(constants.IMAGE_SIZE_HEIGHT, constants.IMAGE_SIZE_WIDTH),
#         batch_size=constants.BATCH_SIZE
#     )

#     # Split the data
#     train_dataset, val_dataset, test_dataset = image_loader.split_dataset()

#     # Initialize the model
#     input_shape = (1, 192, 224, 224, 3)  # None for batch size
#     num_classes = 25

#     model = DenseNetVisionModel(num_classes, input_shape, weights='imagenet')
#     trainer = ModelTrainer(model)

#     # Train the model using the train and validation datasets
#     trainer.train(train_dataset, val_dataset, epochs=constants.EPOCHS)

# if __name__ == "__main__":
#     main()


# Usage example:
input_shape = (None, 192, 224, 224, 3)  # None for batch size
num_classes = 25

model = DenseNetVisionModel(num_classes, input_shape, weights='imagenet')
trainer = ModelTrainer(model)

# Load the datasets
image_loader = ImageLoader(
    image_dir=constants.TRAIN_DATA_PATH,
    label_coordinates_csv=constants.TRAIN_LABEL_CORD_PATH,
    labels_csv=constants.TRAIN_LABEL_PATH,
    roi_size=(constants.IMAGE_SIZE_HEIGHT, constants.IMAGE_SIZE_WIDTH),
    batch_size=constants.BATCH_SIZE
)

# # Load train and validation datasets
train_dataset = image_loader.load_data('train')
val_dataset = image_loader.load_data('val')

# train the mode using:
trainer.train(train_dataset, val_dataset, epochs=10)