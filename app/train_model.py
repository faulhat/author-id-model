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
    
    Output files:
    + model.png - Diagram of the model
    + saved_model.h5 - Saved model
    + accuracy.png - Graph of training progress
    + training_set.pkl - List of paragraphs included in training set

    Note that the accuracy shown for the model is not its true accuracy. It is the accuracy on
    classifications for single words, not for classification on full pages. For full pages
    it is efficient to sum up the classification vectors for all of the words extracted from
    a given page and then use that for prediction. This metric can be tested with the
    test_model.py script.
"""

import os
import tensorflow as tf
import math
import sys
import numpy as np
import pickle
from typing import Iterator
from random import shuffle, randint
from keras import Model, layers
from keras.callbacks import ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy
from keras.applications.mobilenet import MobileNet
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from plot_keras_history import plot_history
from matplotlib import pyplot as plt

from .segment_data import *


# Save generated images to
OUT_DIR = "out/"
MODEL_PLOT_IMG = os.path.join(OUT_DIR, "model.png")
ACC_GRAPH_IMG = os.path.join(OUT_DIR, "accuracy.png")
SAVED_MODEL = os.path.join(OUT_DIR, "saved_model.h5")
DS_LABELS_PATH = os.path.join(OUT_DIR, "ds_labels.pkl")

BATCH_SIZE = 16
STEPS_PER_EPOCH = 200
VALIDATION_STEPS = 50

# The length of fingerprint arrays
N_FINGERPRINT = 200


def top_3_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_5_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


# A function to categorize words in the set by their original form.
# Also returns a dictionary mapping paragraphs to authors.
def categorize_all(
    paragraphs_dir: str, words_dir: str
) -> tuple[dict[str, str], dict[str, str]]:
    para2words = {}
    para2writer = {}
    for writer_dir in glob.glob(os.path.join(paragraphs_dir, "*")):
        _, writer_id = os.path.split(writer_dir)
        for paragraph in glob.glob(os.path.join(writer_dir, "*")):
            para2writer[paragraph] = writer_id
            para2words[paragraph] = []
            paragraph_id = os.path.split(paragraph)[1].split(".")[0]
            for word in glob.glob(os.path.join(words_dir, writer_id, "*")):
                _, word_filename = os.path.split(word)
                if word_filename.startswith(paragraph_id):
                    para2words[paragraph].append(word)

    return para2words, para2writer


# Split data for training, validation, and testing
def split_data(
    para2words: dict[str, str],
    para2writer: dict[str, str],
    encoder: LabelEncoder,
    store_ds_to: str = None,
) -> tuple[np.ndarray, ...]:
    train_files, validation_files, test_files = [], [], []
    train_targets, validation_targets, test_targets = [], [], []

    paragraphs = list(para2words.keys())
    shuffle(paragraphs)
    writer2paras = {}
    train_paras, validation_paras, test_paras = [], [], []
    for paragraph, writer in para2writer.items():
        if writer in writer2paras:
            writer2paras[writer].append(paragraph)
        else:
            writer2paras[writer] = [paragraph]

    for writer, paragraphs in writer2paras.items():
        n_train = math.ceil(len(paragraphs) / 2)
        for _ in range(n_train):
            paragraph = paragraphs.pop(randint(0, len(paragraphs) - 1))
            train_paras.append(paragraph)

            words = para2words[paragraph]
            train_files.extend(words)
            train_targets.extend([writer for _ in words])

        n_valid = math.ceil(len(paragraphs) / 2)
        for _ in range(n_valid):
            paragraph = paragraphs.pop(randint(0, len(paragraphs) - 1))
            validation_paras.append(paragraph)

            words = para2words[paragraph]
            validation_files.extend(words)
            validation_targets.extend([writer for _ in words])

        n_test = len(paragraphs)
        for _ in range(n_test):
            paragraph = paragraphs.pop(randint(0, len(paragraphs) - 1))
            test_paras.append(paragraph)

            words = para2words[paragraph]
            test_files.extend(words)
            test_targets.extend([writer for _ in words])

    if store_ds_to is not None:
        with open(store_ds_to, "wb") as pkl_fp:
            pickle.dump(
                (
                    train_paras,
                    (validation_paras, test_paras),
                    (train_targets, validation_targets, test_targets),
                ),
                pkl_fp,
            )

    train_files, validation_files, test_files = (
        np.asarray(train_files),
        np.asarray(validation_files),
        np.asarray(test_files),
    )

    train_targets, validation_targets, test_targets = (
        np.asarray(train_targets),
        np.asarray(validation_targets),
        np.asarray(test_targets),
    )
    train_targets, validation_targets, test_targets = (
        encoder.transform(train_targets),
        encoder.transform(validation_targets),
        encoder.transform(test_targets),
    )

    # Return the encoder in addition to the split dataset
    # so that it can be used by other parts of the program
    return (
        train_files,
        validation_files,
        test_files,
        train_targets,
        validation_targets,
        test_targets,
    )


# Generator function to get words from the dataset.
def gen_data(
    samples: np.ndarray,
    targets: np.ndarray,
    n_classes: int,
    batch_size: int = BATCH_SIZE,
    do_resize: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    n_samples = len(samples)
    samples_targets = list(zip(samples, targets))

    while True:
        shuffle(samples_targets)
        for offset in range(0, n_samples - batch_size, batch_size):
            batch = samples_targets[offset : offset + batch_size]

            images = []
            targets = []
            for i in range(batch_size):
                # Get the next image in the batch
                sample, target = batch[i]
                with Image.open(sample) as img:
                    if do_resize:
                        img = img.resize((IMG_WIDTH, IMG_HEIGHT))

                    images.append(np.asarray(img))
                    targets.append(target)

            # Prepare the inputs and targets for the convolutional net
            X_train = transform_images(images)
            y_train = tf.keras.utils.to_categorical(np.array(targets), n_classes)

            yield X_train, y_train


# Create the model to be trained
def gen_model(n_writers: int) -> Model:
    # The MobileNet image recognition model will be used as a base
    base_model = MobileNet(
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), weights="imagenet", include_top=False
    )
    base_model.trainable = False
    flatten = layers.Flatten()(base_model.output)

    # Dropout layer to prevent overfitting
    dropout = layers.Dropout(0.6)(flatten)
    dense = layers.Dense(400, activation="relu")(dropout)
    dense = layers.Dense(N_FINGERPRINT, activation="relu")(dense)

    output = layers.Dense(n_writers, activation="softmax")(dense)
    model = Model(inputs=base_model.input, outputs=output)

    return model


def get_model_generators(
    split_ds: tuple[np.ndarray, ...], encoder: LabelEncoder
) -> tuple[Model, tuple[Iterator[tuple[np.ndarray, np.ndarray]], ...]]:
    (
        train_files,
        validation_files,
        test_files,
        train_targets,
        validation_targets,
        test_targets,
    ) = split_ds

    n_writers = len(encoder.classes_)
    model = gen_model(n_writers)

    train_generator = gen_data(train_files, train_targets, n_writers)
    validation_generator = gen_data(validation_files, validation_targets, n_writers)
    test_generator = gen_data(test_files, test_targets, n_writers)

    return model, (train_generator, validation_generator, test_generator)


CHECKPOINT_CALLBACK = ModelCheckpoint(
    filepath=SAVED_MODEL,
    verbose=1,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

if __name__ == "__main__":
    n_epochs = 30
    if len(sys.argv) > 1:
        try:
            n_epochs = int(sys.argv[1])
        except ValueError:
            pass

    os.makedirs(OUT_DIR, exist_ok=True)

    # Assume data has already been processed
    writer2words, encoder = get_segmented_data(WORDS, LE_SAVE_PATH, do_gen_encoder=True)

    # Retrieve and split the dataset
    para2words, para2writer = categorize_all(PARAGRAPHS, WORDS)
    split_ds = split_data(para2words, para2writer, encoder, store_ds_to=DS_LABELS_PATH)

    model, (
        train_generator,
        validation_generator,
        test_generator,
    ) = get_model_generators(split_ds, encoder)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy", top_3_accuracy, top_5_accuracy],
    )
    print(model.summary())
    tf.keras.utils.plot_model(model, to_file=MODEL_PLOT_IMG, show_shapes=True)

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=n_epochs,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        callbacks=[CHECKPOINT_CALLBACK],
    )

    # Plot training history
    plot_history(history, path=ACC_GRAPH_IMG)
    plt.close()

    scores = model.evaluate(test_generator, steps=500)
