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

import glob
import math
import os
import shutil
import sys
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
from tqdm import trange

from segment import get_paragraph, get_words


# Run from repository root
DATA_DIR = "data/"
DATASET_PATH = os.path.join(DATA_DIR, "data/")
SEGMENTS = os.path.join(DATA_DIR, "segments/")
PARAGRAPHS = os.path.join(SEGMENTS, "paragraphs/")
WORDS = os.path.join(SEGMENTS, "words/")

# Save generated images to
OUT_DIR = "out/"
MODEL_PLOT_IMG = os.path.join(OUT_DIR, "model.png")
ACC_GRAPH_IMG = os.path.join(OUT_DIR, "accuracy.png")
SAVED_MODEL = os.path.join(OUT_DIR, "saved_model.h5")

# Dimensions of input images
# From default input dimensions for MobileNet
IMG_WIDTH = 224
IMG_HEIGHT = 224

BATCH_SIZE = 16

# Min forms a contributor must have filled out to be included in set
MIN_FORMS_THRESHOLD = 5


def top_3_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_5_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


# Retrieve data for training
def get_data(dataset_path: str) -> tuple[list[str], list[str]]:
    writer_dirs = glob.glob(os.path.join(dataset_path, "*"))
    author2imgs = [(os.path.split(writer_dir)[1], glob.glob(os.path.join(writer_dir, "*")))
                   for writer_dir in writer_dirs]

    print("Getting data from dataset...")
    filenames = []
    writers = []
    for author, img_files in author2imgs:
        if len(img_files) >= MIN_FORMS_THRESHOLD:
            for img_file in img_files:
                filenames.append(img_file)
                writers.append(author)

    return filenames, writers


# Extract paragraphs and words from image files
# paragraph_dir and word_dir must exist
def segment_data(filenames: list[str], writers: list[str], paragraph_dir: str, word_dir: str) -> dict[str, str]:
    print("Finding words in forms...")
    writer2words = {}
    totalwords = 0
    for _, (filename, writer) in zip(trange(len(filenames)), zip(filenames, writers)):
        writer_dir = os.path.join(paragraph_dir, f"{writer}/")
        os.makedirs(writer_dir, exist_ok=True)

        _, tail = os.path.split(filename)
        paragraph_img_path = os.path.join(
            writer_dir, f"{tail}.para.png")
        get_paragraph(filename, paragraph_img_path)

        writer_word_dir = os.path.join(word_dir, f"{writer}/")
        paragraph_prefix = f"{tail}_"
        os.makedirs(writer_word_dir, exist_ok=True)
        word_filenames = get_words(paragraph_img_path, writer_word_dir, prefix=paragraph_prefix)
        if writer in writer2words:
            writer2words[writer].extend(word_filenames)
        else:
            writer2words[writer] = word_filenames
        
        totalwords += len(word_filenames)

    print(f"Found a total of {totalwords} words.")
    return writer2words


# Get segmented data, assuming it's already been processed
def get_segmented_data(word_dir: str) -> dict[str, str]:
    print("Getting segmented images...")
    writer2words = {}
    writer_dirs = glob.glob(os.path.join(word_dir, "*"))
    for writer_dir in writer_dirs:
        _, tail = os.path.split(writer_dir)
        writer2words[tail] = glob.glob(os.path.join(writer_dir, "*"))

    return writer2words


# Split data for training, validation, and testing
def split_data(writer2words: dict[str, str]) ->\
        tuple[LabelEncoder, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Create an encoder that gives each writer a unique id starting from zero.
    encoder = LabelEncoder()
    encoder.fit(np.asarray(list(writer2words.keys())))

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
    return encoder, train_files, validation_files, test_files, train_targets, validation_targets, test_targets


# Generator function to select random portions of images and resize them to the required dimensions
def gen_data(samples: np.ndarray, targets: np.ndarray, n_classes: int, batch_size: int = BATCH_SIZE) -> Iterator[tuple[np.ndarray, np.ndarray]]:
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
                    resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
                    images.append(np.asarray(resized))
                    targets.append(batch_target)

            # Prepare the inputs and targets for the convolutional net
            X_train = np.array(images).reshape(
                len(images), IMG_WIDTH, IMG_HEIGHT, 1).repeat(3, axis=-1).astype("float32") / 255
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
    dropout = layers.Dropout(0.4)(flatten)
    dense = layers.Dense(1000, activation="relu")(dropout)
    dropout = layers.Dropout(0.1)(dense)
    dense = layers.Dense(1000, activation="relu")(dropout)
    dense = layers.Dense(500, activation="relu")(dense)
    dense = layers.Dense(200, activation="relu")(dense)
    output = layers.Dense(n_writers, activation="softmax")(dense)
    model = Model(inputs=base_model.input, outputs=output)

    return model


if __name__ == "__main__":
    writer2words: dict[str, str]
    if len(sys.argv) > 1 and sys.argv[1].startswith("skip"):
        # Skip preprocessing and assume data has already been processed
        writer2words = get_segmented_data(WORDS)
    else:
        # Create output directory
        os.makedirs(OUT_DIR, exist_ok=True)

        # Clear generated data
        shutil.rmtree(SEGMENTS)

        # Create data and output directories if not present
        os.makedirs(PARAGRAPHS)
        os.makedirs(WORDS)

        filenames, writers = get_data(DATASET_PATH)
        writer2words = segment_data(filenames, writers, PARAGRAPHS, WORDS)

    # Retrieve and split the dataset
    le, train_files,  validation_files, test_files, train_targets, validation_targets, test_targets =\
        split_data(writer2words)

    n_writers = len(le.classes_)
    model = gen_model(n_writers)

    train_generator = gen_data(train_files, train_targets, n_writers)
    validation_generator = gen_data(
        validation_files, validation_targets, n_writers)
    test_generator = gen_data(test_files, test_targets, n_writers)

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy", top_3_accuracy, top_5_accuracy])
    print(model.summary())
    tf.keras.utils.plot_model(model, to_file=MODEL_PLOT_IMG, show_shapes=True)

    # Train the model
    history = model.fit(train_generator, validation_data=validation_generator,
                        epochs=35, steps_per_epoch=200, validation_steps=100)

    # Plot training history
    plot_history(history, path=ACC_GRAPH_IMG)
    plt.close()

    scores = model.evaluate(test_generator, steps=500)

    model.save(SAVED_MODEL)
