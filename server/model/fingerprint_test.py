from random import shuffle
from keras.models import Model, load_model
from tqdm import trange
import numpy as np

from .train_model import top_3_accuracy, top_5_accuracy
from .test_model import *


def get_fingerprint_model(saved_model: str) -> Model:
    orig_model = load_model(saved_model, custom_objects={"top_3_accuracy": top_3_accuracy, "top_5_accuracy": top_5_accuracy})
    new_model = Model(inputs=orig_model.inputs, outputs=orig_model.layers[-2].output)

    return new_model


def getGroups(para2writer: dict[str, str]) -> dict[str, list[str]]:
    writer2para: dict[str, list[str]] = {}
    for paragraph, writer in para2writer.items():
        if writer in writer2para:
            writer2para[writer].append(paragraph)
        else:
            writer2para[writer] = [paragraph]
    
    out = {}
    for paragraphs in writer2para.values():
        shuffle(paragraphs)
        out[paragraphs[0]] = paragraphs[1:]

    return out


if __name__ == "__main__":
    train_paras, _, _ = retrieve_set_labels(DS_LABELS_PATH)

    encoder = load_encoder(LE_SAVE_PATH)
    para2words, para2writer = categorize_all(PARAGRAPHS, WORDS)
    para2words = get_test_paras(para2words, train_paras)
    paragraphs = list(para2writer.keys())
    for paragraph in paragraphs:
        if paragraph not in para2words:
            del para2writer[paragraph]

    grouped = getGroups(para2writer)

    model = get_fingerprint_model(SAVED_MODEL)

    print("Evaluating fingerprint accuracy...")

    print("Finding fingerprints for labelled paragraphs.")
    labelled = list(grouped.keys())
    avgOutputs = {paragraph: 0 for paragraph in labelled}
    for _, paragraph in zip(trange(len(labelled)), labelled):
        avgOutputs[paragraph] = getAvgOutput(model, para2words[paragraph])

    reverse = {}
    for key, value in grouped.items():
        for paragraph in value:
            reverse[paragraph] = key

    print("Finding distances...")
    n_correct = 0
    n_correct_top3 = 0
    n_correct_top5 = 0
    for _, (unlabelled, match) in zip(trange(len(reverse.keys())), reverse.items()):
        avgOutput = getAvgOutput(model, para2words[unlabelled])
        distances = [(compare_to, np.linalg.norm(avgOutputs[compare_to] - avgOutput)) for compare_to in labelled]

        ordered = sorted(distances, key=lambda pair: pair[1])
        ordered = [pair[0] for pair in ordered]
        if match == ordered[0]:
            n_correct += 1
        
        if match in ordered[:3]:
            n_correct_top3 += 1
        
        if match in ordered[:5]:
            n_correct_top5 += 1

    print(f"Accuracy: {n_correct/len(reverse)}")
    print(f"Top 3 accuracy: {n_correct_top3/len(reverse)}")
    print(f"Top 5 accuracy: {n_correct_top5/len(reverse)}")
