"""
    Thomas: this program is meant to test the true accuracy of the model produced by
    train_model.py. It will do this by iterating over all of the forms in the dataset
    included in the validation and testing sets
    (meaning that only contributors who contributed 5 or more forms are included and there
    will be no overlap with the training set),
    and running the classifier on each word extracted from each form. The classification vectors
    for all of the words in each form will be summed up to find a classifcation vector
    for the full form.
"""

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import trange

from .train_model import *
from .continue_training import retrieve_set_labels


# Function to classify one form
def getAvgOutputImgs(model: Model, word_imgs: list[Image.Image], do_resize: bool = False)\
        -> np.ndarray:
    if do_resize:
        for i, word_img in enumerate(word_imgs):
            word_imgs[i] = word_img.resize((IMG_WIDTH, IMG_HEIGHT))

    word_img_array = list(map(np.asarray, word_imgs))
    word_img_array = transform_images(word_img_array)
    pred = model.predict(word_img_array)

    pred_mean = np.mean(pred, axis=0)

    for fp in word_imgs:
        fp.close()
    
    return pred_mean

def getAvgOutput(model: Model, word_paths: list[str], do_resize: bool = False)\
        -> np.ndarray:
    word_imgs = list(map(Image.open, word_paths))

    return getAvgOutputImgs(model, word_imgs, do_resize=do_resize)

# Function to get the accuracy of a classification
def getAccuracy(pred_mean: np.ndarray, y_label: int)\
        -> tuple[np.int32, np.int32, np.int32]:
    y_sparse = tf.keras.utils.to_categorical(np.asarray([y_label]))
    pred_max = np.argmax(pred_mean)
    pred_correct = np.equal(pred_max, y_label).astype("int32")
    
    pred_mean_array = np.asarray([pred_mean])
    pred_correct_top3 = top_k_categorical_accuracy(y_sparse, pred_mean_array, k=3)[0]
    pred_correct_top5 = top_k_categorical_accuracy(y_sparse, pred_mean_array, k=5)[0]

    return pred_correct, pred_correct_top3, pred_correct_top5


def get_test_paras(para2words: dict[str, str], train_paras: list[str]) -> dict[str, str]:
    test_para2words = {}
    for para, words in para2words.items():
        if para not in train_paras:
            test_para2words[para] = words
    
    return test_para2words


def transform_by_para(para2words: dict[str, str], para2writer: dict[str, str], encoder: LabelEncoder)\
        -> list[tuple[list[str], int]]:
    words_label = []
    for paragraph, words in para2words.items():
        writer = para2writer[paragraph]
        y_label = encoder.transform([writer])[0]
        words_label.append((words, y_label))

    return words_label


if __name__ == "__main__":
    train_paras, _, _ = retrieve_set_labels(DS_LABELS_PATH)

    encoder = load_encoder(LE_SAVE_PATH)
    para2words, para2writer = categorize_all(PARAGRAPHS, WORDS)
    para2words = get_test_paras(para2words, train_paras)
    words_label = transform_by_para(para2words, para2writer, encoder)

    model = gen_model(len(encoder.classes_))
    model.load_weights(SAVED_MODEL)

    print("Evaluating true classification accuracy...")
    n_correct = 0
    n_correct_top3 = 0
    n_correct_top5 = 0
    for _, (words, y_label) in zip(trange(len(words_label)), words_label):
        correct, correct_top3, correct_top5 = getAccuracy(getAvgOutput(model, words), y_label)
        n_correct += correct
        n_correct_top3 += correct_top3
        n_correct_top5 += correct_top5

    print(f"Accuracy: {n_correct/len(words_label)}")
    print(f"Top 3 accuracy: {n_correct_top3/len(words_label)}")
    print(f"Top 5 accuracy: {n_correct_top5/len(words_label)}")
