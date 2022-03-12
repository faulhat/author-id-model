"""
    Thomas: this program is meant to test the true accuracy of the model produced by
    train_model.py. It will do this by iterating over all of the forms in the dataset
    which met the requirements for inclusion in training, validation, and testing
    (meaning that only contributors who contributed 5 or more forms are included),
    and running the classifier on each word extracted from each. The classification vectors
    for all of the words in each form will be summed up to find a classifcation vector
    for the full form.
"""

from random import choices
import sys
import glob
import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import trange

from train_model import *


# Function to vote on one individual form
def vote(model: Model, words: list[str], y_label: int, do_resize: bool = False)\
        -> tuple[np.int32, np.int32, np.int32]:
    y_sparse = tf.keras.utils.to_categorical(np.asarray([y_label]))

    word_imgs = list(map(Image.open, words))
    if do_resize:
        for i, word_img in enumerate(word_imgs):
            word_imgs[i] = word_img.resize((IMG_WIDTH, IMG_HEIGHT))

    word_img_array = list(map(np.asarray, word_imgs))
    word_img_array = transform_images(word_img_array)
    pred = model.predict(word_img_array)

    pred_sum = np.sum(pred, axis=0)
    pred_max = np.argmax(pred_sum)
    pred_correct = np.equal(pred_max, y_label).astype("int32")

    pred_sum_array = np.asarray([pred_sum])
    pred_correct_top3 = top_k_categorical_accuracy(y_sparse, pred_sum_array, k=3)[0]
    pred_correct_top5 = top_k_categorical_accuracy(y_sparse, pred_sum_array, k=5)[0]

    for fp in word_imgs:
        fp.close()

    return pred_correct, pred_correct_top3, pred_correct_top5


def transform_by_para(paragraph2words: dict[str, str], paragraph2writer: dict[str, str], encoder: LabelEncoder)\
        -> list[tuple[list[str], int]]:
    words_label = []
    for paragraph, words in paragraph2words.items():
        writer = paragraph2writer[paragraph]
        y_label = encoder.transform([writer])[0]
        words_label.append((words, y_label))

    return words_label


if __name__ == "__main__":
    encoder = load_encoder(LE_SAVE_PATH)
    paragraph2words, paragraph2writer = categorize_all(PARAGRAPHS, WORDS)
    words_label = transform_by_para(paragraph2words, paragraph2writer, encoder)

    model = gen_model(len(encoder.classes_))
    model.load_weights(SAVED_MODEL)

    print("Evaluating true accuracy...")
    n_correct = 0
    n_correct_top3 = 0
    n_correct_top5 = 0
    for _, (words, y_label) in zip(trange(len(words_label)), words_label):
        correct, correct_top3, correct_top5 = vote(model, words, y_label)
        n_correct += correct
        n_correct_top3 += correct_top3
        n_correct_top5 += correct_top5

    print(f"Accuracy: {n_correct/len(words_label)}")
    print(f"Top 3 accuracy: {n_correct_top3/len(words_label)}")
    print(f"Top 5 accuracy: {n_correct_top5/len(words_label)}")
