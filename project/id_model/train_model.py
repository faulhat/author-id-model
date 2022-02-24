"""
    Thomas: Program to create and train a model for handwriting identification using transfer learning.
    A lot of code repurposed from here: https://www.kaggle.com/tejasreddy/offline-handwriting-recognition-cnn/notebook
"""

import glob
import os
import numpy as np
from tensorflow import keras
from keras import Model, layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Run from repository root
DATA_DIR = "data"
FORMS_PATH = os.path.join(DATA_DIR, "forms_for_parsing.txt")

# Max dimensions for input images
MAX_X = 640
MAX_Y = 1136


# Retrieve data for training
def get_data(data_file: str) -> tuple:
    key2writer = {}
    with open(FORMS_PATH, "r") as f:
        for line in f:
            parts = line.split()
            key = parts[0]
            writer = parts[1]
            key2writer[key] = writer

    filenames = []  # Inputs by filename
    writers = []  # Corresponding targets
    path_to_files = os.path.join(DATA_DIR, "data_subset", "data_subset")
    for filename in sorted(glob.glob(path_to_files)):
        filenames.append(filename)
        image_name = filename.split("/")[-1]
        image_key, _ = os.path.splitext(image_name)
        parts = image_key.split("-")
        form = parts[0] + "-" + parts[1]
        writers.append(key2writer[form])

    img_files = np.asarray(filenames)
    img_targets = np.asarray(writers)
    return img_files, img_targets


def split_data(img_files: np.ndarray, img_targets: np.ndarray) -> tuple:
    # Split data for training, validation, and testing
    encoder = LabelEncoder()
    encoder.fit(img_targets)
    encoded_Y = encoder.transform(img_targets)

    train_files, other_files, train_targets, other_targets = train_test_split(
        img_files, encoded_Y, train_size=0.66, random_state=52, shuffle=True)
    validation_files, test_files, validation_targets, test_targets = train_test_split(
        other_files, other_targets, train_size=0.5, random_state=25, shuffle=True)

    return encoder, train_files,  validation_files, test_files, train_targets, validation_targets, test_targets


def gen_model(n_writers: int) -> Model:
    # Create the model to be trained
    # We're using transfer learning with the Xception image recognition model
    base_model = keras.applications.xception.Xception(
        weights="imagenet", include_top=False)
    dropout1 = layers.Dropout(0.5)(base_model.output)
    fingerprint = layers.Dense(64, activation="relu")(dropout1)
    dropout = layers.Dropout(0.5)(fingerprint)
    output = layers.Dense(n_writers)(dropout)
    model = Model(inputs=base_model.input, outputs=output)

    return model


def train_model(model: Model, in_files: np.ndarray, out_targets: np.ndarray) -> None:
    pass


if __name__ == "__main__":
    le, train_files,  validation_files, test_files, train_targets, validation_targets, test_targets = split_data(*get_data(
        os.path.join(DATA_DIR, "forms_for_parsing.txt")))

    n_writers = len(le.classes_)
    model = gen_model(n_writers)
