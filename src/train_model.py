import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Function to load and preprocess data
def load_data(train_dir, test_dir):
    try:
        # Data augmentation for training data
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            train_dir, target_size=(48, 48), batch_size=32, color_mode="grayscale", class_mode="categorical"
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir, target_size=(48, 48), batch_size=32, color_mode="grayscale", class_mode="categorical"
        )

        return train_generator, test_generator
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Function to create the model
def create_model():
    try:
        model = Sequential([
            Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),

            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(7, activation="softmax"),  # Assuming 7 emotion classes
        ])

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        raise

# Train the model
def train_model(model, train_generator, test_generator, epochs=10):
    try:
        model.fit(train_generator, epochs=epochs, validation_data=test_generator)
        model.save("../data/model/emotion_detector_model_1.h5")
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    try:
        # Directories for training and testing data
        train_dir = "../data/train"
        test_dir = "../data/test"

        # Load data
        train_gen, test_gen = load_data(train_dir, test_dir)

        # Create model
        model = create_model()

        # Train model
        train_model(model, train_gen, test_gen, epochs=10)

    except Exception as e:
        print(f"An error occurred in the main program: {e}")