"""
    Thomas: Program to create and train a model for handwriting identification using transfer learning.
    A lot of code repurposed from here: https://www.kaggle.com/tejasreddy/offline-handwriting-recognition-cnn/notebook
"""

import glob
import os
import random
import numpy as np
import tensorflow as tf
from keras import Model, layers
from keras.applications.mobilenet_v2 import MobileNetV2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
from plot_keras_history import plot_history
from matplotlib import pyplot as plt

# Run from repository root
DATA_DIR = "data"
FORMS_PATH = os.path.join(DATA_DIR, "forms_for_parsing.txt")

# Save generated images to
OUT_DIR = "out"
MODEL_PLOT_IMG = os.path.join(OUT_DIR, "model.png")
ACC_GRAPH_IMG = os.path.join(OUT_DIR, "accuracy.png")

# Dimensions of input images
# From default input dimensions for MobileNetV2
IMG_WIDTH = 224
IMG_HEIGHT = 224

BATCH_SIZE = 32


# Retrieve data for training
def get_data():
    key2writer = {}  # Map sample id to writer
    with open(FORMS_PATH, "r") as f:
        for line in f:
            parts = line.split()
            key = parts[0]
            writer = parts[1]
            key2writer[key] = writer

    filenames = []  # Inputs by filename
    writers = []  # Corresponding targets
    path_to_files = os.path.join(DATA_DIR, "data_subset", "data_subset", "*")
    for filename in sorted(glob.glob(path_to_files)):
        filenames.append(filename)
        image_name = filename.split("/")[-1]
        file, _ = os.path.splitext(image_name)
        parts = file.split("-")
        form = parts[0] + "-" + parts[1]
        writers.append(key2writer[form])

    img_files = np.asarray(filenames)
    img_targets = np.asarray(writers)
    return img_files, img_targets


# Split data for training, validation, and testing
def split_data(img_files, img_targets):
    # Create an encoder that gives each writer a unique id starting from zero.
    encoder = LabelEncoder()
    encoder.fit(img_targets)
    encoded_Y = encoder.transform(img_targets)

    # Split the dataset
    train_files, other_files, train_targets, other_targets = train_test_split(
        img_files, encoded_Y, train_size=0.66, random_state=52, shuffle=True)
    validation_files, test_files, validation_targets, test_targets = train_test_split(
        other_files, other_targets, train_size=0.5, random_state=25, shuffle=True)

    # Return the encoder in addition to the split dataset
    # so that it can be used by other parts of the program
    return encoder, train_files,  validation_files, test_files, train_targets, validation_targets, test_targets


# Generator function to select random portions of images and resize themto the required dimensions
def gen_data(samples, target_files, n_classes, batch_size=BATCH_SIZE):
    n_samples = len(samples)

    while True:
        for offset in range(0, n_samples - batch_size, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_targets = target_files[offset:offset+batch_size]

            images = []
            targets = []
            for i in range(batch_size):
                # Get the next image in the batch
                batch_sample = batch_samples[i]
                batch_target = batch_targets[i]
                img = Image.open(batch_sample)
                img_width = img.size[0]
                img_height = img.size[1]

                # Select a random section of the image of a random width and height
                # which will be no less than 25/36 the area of the original
                new_width = random.randint(5 * img_width // 6, img_width)
                new_height = random.randint(5 * img_height // 6, img_height)
                new_left = random.randint(0, img_width - new_width)
                new_up = random.randint(0, img_height - new_height)

                # Crop the image and resize it to the required input dimensions
                new_img = img.crop((new_left, new_up, new_left + new_width,
                                    new_up + new_height)).resize((IMG_WIDTH, IMG_HEIGHT))
                images.append(np.asarray(new_img))
                targets.append(batch_target)

            # Prepare the inputs and targets for the convolutional net
            X_train = np.array(images).reshape(
                len(images), IMG_WIDTH, IMG_HEIGHT, 1).repeat(3, axis=3).astype("float32") / 255
            y_train = tf.keras.utils.to_categorical(
                np.array(targets), n_classes)

            yield shuffle(X_train, y_train)


# Create the model to be trained
def gen_model(n_writers):
    # The MobileNetV2 image recognition model will be used as a base
    base_model = MobileNetV2(input_shape=(
        IMG_WIDTH, IMG_HEIGHT, 3), weights="imagenet", include_top=False)
    base_model.trainable = False
    flatten = layers.Flatten()(base_model.output)

    # Dropout layer to prevent overfitting
    dropout1 = layers.Dropout(0.1)(flatten)
    dense1 = layers.Dense(300, activation="relu")(dropout1)
    dropout2 = layers.Dropout(0.1)(dense1)
    dense2 = layers.Dense(500, activation="relu")(dropout2)
    dropout3 = layers.Dropout(0.1)(dense2)
    dense3 = layers.Dense(500, activation="relu")(dropout3)
    # The fingerprint array which will be used by the actual application
    fingerprint = layers.Dense(200, activation="relu")(dense3)
    output = layers.Dense(n_writers, activation="softmax")(fingerprint)
    model = Model(inputs=base_model.input, outputs=output)

    return model


if __name__ == "__main__":
    # Create output directory if not present
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    # Retrieve and split the dataset
    le, train_files,  validation_files, test_files, train_targets, validation_targets, test_targets = split_data(
        *get_data())

    n_writers = len(le.classes_)
    model = gen_model(n_writers)

    train_generator = gen_data(train_files, train_targets, n_writers)
    validation_generator = gen_data(
        validation_files, validation_targets, n_writers)
    test_generator = gen_data(test_files, test_targets, n_writers)

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    print(model.summary())
    tf.keras.utils.plot_model(model, to_file=MODEL_PLOT_IMG, show_shapes=True)

    # Train the model
    history = model.fit(train_generator, validation_data=validation_generator,
                        epochs=3, steps_per_epoch=250, validation_steps=20)

    # Plot training history
    plot_history(history, path=ACC_GRAPH_IMG)
    plt.close()

    scores = model.evaluate(test_generator, steps=200)
    print(f"Accuracy = {scores[1]}")
