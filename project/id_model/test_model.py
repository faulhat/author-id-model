"""
    Thomas: this program is meant to test the true accuracy of the model produced by
    train_model.py. It will do this by iterating over all of the forms in the dataset
    which met the requirements for inclusion in training, validation, and testing
    (meaning that only contributors who contributed 5 or more forms are included),
    and running the classifier on each word extracted from each. The classification
    for each word will be considered a "vote" on the author of the form, and whichever
    author earns a plurality of the votes will be the model's prediction for the form.
"""

import sys
import glob
import os
import numpy as np
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import trange

from train_model import *


# A function to categorize words in the set by their original form.
# Also returns a dictionary mapping paragraphs to authors.
def categorize_all(paragraphs_dir: str, words_dir: str)\
        -> tuple[dict[str, str], dict[str, str]]:
    paragraph2words = {}
    paragraph2writer = {}
    for writer_dir in glob.glob(os.path.join(paragraphs_dir, "*")):
        _, writer_id = os.path.split(writer_dir)
        for paragraph in glob.glob(os.path.join(writer_dir, "*")):
            paragraph2writer[paragraph] = writer_id
            paragraph2words[paragraph] = []
            _, para_filename = os.path.split(paragraph)
            for word in glob.glob(os.path.join(words_dir, writer_id, "*")):
                if word.startswith(para_filename):
                    paragraph2words[paragraph].append(word)

    return paragraph2words, paragraph2writer


# Function to vote on one individual form
def vote(model: Model, encoder: LabelEncoder, words: list[str], y_label: int, do_resize: bool = False) -> bool:
    label2votes = [0 for _ in encoder.classes_]
    word_imgs = list(map(Image.open, words))
    if do_resize:
        for i, word_img in enumerate(word_imgs):
            word_imgs[i] = word_img.resize((IMG_WIDTH, IMG_HEIGHT))

    word_img_array = transform_images(word_imgs)
    pred = model.predict(word_img_array)
    pred = np.argmax(pred, axis=1)
    for label in pred:
        label2votes[label] += 1

    pred_label = np.argmax(np.asarray(label2votes))
    pred_correct = pred_label == y_label

    for fp in word_imgs:
        fp.close()

    return pred_correct


def transform_by_para(paragraph2words: dict[str, str], paragraph2writer: dict[str, str], encoder: LabelEncoder)\
        -> list[tuple[list[str], int]]:
    words_label = []
    for paragraph, words in paragraph2words.items():
        writer = paragraph2writer[paragraph]
        y_label = encoder.transform([writer])[0]
        words_label.append((words, y_label))

    return words_label


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("preprocess"):
        clear_and_process_data()

    encoder = load_encoder(LE_SAVE_PATH)
    paragraph2words, paragraph2writer = categorize_all(PARAGRAPHS, WORDS)
    words_label = transform_by_para(paragraph2words, paragraph2writer, encoder)

    model = gen_model(len(encoder.classes_))
    model.load_weights(SAVED_MODEL)

    print("Evaluating true accuracy...")
    n_correct = 0
    for _, (words, y_label) in zip(trange(len(words_label)), words_label):
        correct, score = vote(model, encoder, words, y_label)
        if correct:
            n_correct += 1

    print(f"Accuracy: {n_correct/len(words_label)}")
