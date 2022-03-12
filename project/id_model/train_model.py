"""
    Thomas: Program to create and train a model for handwriting identification using transfer learning.
    A lot of code repurposed from here: https://www.kaggle.com/tejasreddy/offline-handwriting-recognition-cnn/notebook

    Directory structure of data:
    data/
    -> data/
     -> [ids of contributors]
      -> [forms from each contributor]
    -> segments/
     -> paragraphs
      -> [ids of contributors]
       -> [paragraphs from corresponding forms]
     -> words
      -> [ids of contributors]
       -> [words from all forms from each contributor]
"""

import math
import os
import numpy as np
import tensorflow as tf
from typing import Iterator
from random import shuffle, randint
from keras import Model, layers
from keras.metrics import top_k_categorical_accuracy
from keras.applications.mobilenet import MobileNet
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from plot_keras_history import plot_history
from matplotlib import pyplot as plt

from segment_data import *


# Save generated images to
OUT_DIR = "out/"
MODEL_PLOT_IMG = os.path.join(OUT_DIR, "model.png")
ACC_GRAPH_IMG = os.path.join(OUT_DIR, "accuracy.png")
SAVED_MODEL = os.path.join(OUT_DIR, "saved_model.h5")

BATCH_SIZE = 12


def top_3_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_5_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


# Split data for training, validation, and testing
def split_data(writer2words: dict[str, str], encoder: LabelEncoder)\
        -> tuple[np.ndarray, ...]:
    # Split the dataset
    train_files, validation_files, test_files = [], [], []
    train_targets, validation_targets, test_targets = [], [], []
    for key, val in writer2words.items():
        n_train = math.ceil(len(val) * 0.7)
        for _ in range(n_train):
            i = randint(0, len(val) - 1)
            train_files.append(val.pop(i))
            train_targets.append(key)

        n_valid = math.ceil(len(val) * 0.5)
        for _ in range(n_valid):
            i = randint(0, len(val) - 1)
            validation_files.append(val.pop(i))
            validation_targets.append(key)

        n_test = len(val)
        for _ in range(n_test):
            i = randint(0, len(val) - 1)
            test_files.append(val.pop(i))
            test_targets.append(key)

    train_files, validation_files, test_files = np.asarray(
        train_files), np.asarray(validation_files), np.asarray(test_files)
    train_targets, validation_targets, test_targets = np.asarray(
        train_targets), np.asarray(validation_targets), np.asarray(test_targets)
    train_targets, validation_targets, test_targets = encoder.transform(
        train_targets), encoder.transform(validation_targets), encoder.transform(test_targets)

    # Return the encoder in addition to the split dataset
    # so that it can be used by other parts of the program
    return train_files, validation_files, test_files, train_targets, validation_targets, test_targets

# Generator function to get words from the dataset.


def gen_data(samples: np.ndarray, targets: np.ndarray, n_classes: int, batch_size: int = BATCH_SIZE, do_resize: bool = False)\
        -> Iterator[tuple[np.ndarray, np.ndarray]]:
    n_samples = len(samples)
    samples_targets = list(zip(samples, targets))

    while True:
        shuffle(samples_targets)
        for offset in range(0, n_samples - batch_size, batch_size):
            batch = samples_targets[offset:offset+batch_size]

            images = []
            targets = []
            for i in range(batch_size):
                # Get the next image in the batch
                batch_sample, batch_target = batch[i]
                with Image.open(batch_sample) as img:
                    if do_resize:
                        img = img.resize((IMG_WIDTH, IMG_HEIGHT))

                    images.append(np.asarray(img))
                    targets.append(batch_target)

            # Prepare the inputs and targets for the convolutional net
            X_train = transform_images(images)
            y_train = tf.keras.utils.to_categorical(
                np.array(targets), n_classes)

            yield X_train, y_train


# Create the model to be trained
def gen_model(n_writers: int) -> Model:
    # The MobileNet image recognition model will be used as a base
    base_model = MobileNet(input_shape=(
        IMG_WIDTH, IMG_HEIGHT, 3), weights="imagenet", include_top=False)
    base_model.trainable = False
    flatten = layers.Flatten()(base_model.output)

    # Dropout layer to prevent overfitting
    dropout = layers.Dropout(0.3)(flatten)
    dense = layers.Dense(400, activation="relu")(dropout)
    dense = layers.Dense(400, activation="relu")(dense)
    dense = layers.Dense(500, activation="relu")(dense)

    output = layers.Dense(n_writers, activation="softmax")(dense)
    model = Model(inputs=base_model.input, outputs=output)

    return model


def retrieve_split_dataset(writer2words: dict[str, str], encoder: LabelEncoder)\
        -> tuple[Model, tuple[Iterator[tuple[np.ndarray, np.ndarray]], ...]]:
    # Retrieve and split the dataset
    train_files,  validation_files, test_files, train_targets, validation_targets, test_targets =\
        split_data(writer2words, encoder)

    n_writers = len(encoder.classes_)
    model = gen_model(n_writers)

    train_generator = gen_data(train_files, train_targets, n_writers)
    validation_generator = gen_data(
        validation_files, validation_targets, n_writers)
    test_generator = gen_data(test_files, test_targets, n_writers)

    return model, (train_generator, validation_generator, test_generator)


if __name__ == "__main__":
    # Assume data has already been processed
    writer2words, encoder = get_segmented_data(
        WORDS, LE_SAVE_PATH, do_gen_encoder=True)

    model, (train_generator, validation_generator,
            test_generator) = retrieve_split_dataset(writer2words, encoder)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy", top_3_accuracy, top_5_accuracy])
    print(model.summary())
    tf.keras.utils.plot_model(model, to_file=MODEL_PLOT_IMG, show_shapes=True)

    # Train the model
    history = model.fit(train_generator, validation_data=validation_generator,
                        epochs=50, steps_per_epoch=175, validation_steps=50)

    # Plot training history
    plot_history(history, path=ACC_GRAPH_IMG)
    plt.close()

    scores = model.evaluate(test_generator, steps=500)

    model.save(SAVED_MODEL)
