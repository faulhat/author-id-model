"""
    This program will use the models provided by the handwritten-text-recognition submodule included
    in this repo to extract individual handwritten words from pages with blocks of handwritten text
    on them. It will store these word segments as separate image files in a client-specified
    target directory.
"""

import glob
import os
import shutil
import numpy as np
import pickle
from tqdm import trange
from PIL import Image
from sklearn.preprocessing import LabelEncoder

from .word_segmentation import get_paragraph, get_words


# Run from repository root
DATA_DIR = "data/"
DATASET_PATH = os.path.join(DATA_DIR, "data/")
SEGMENTS = os.path.join(DATA_DIR, "segments/")
PARAGRAPHS = os.path.join(SEGMENTS, "paragraphs/")
WORDS = os.path.join(SEGMENTS, "words/")
# Path to encoder save file
LE_SAVE_PATH = os.path.join(SEGMENTS, "key.pkl")

# Dimensions of input images
# From default input dimensions for MobileNet
IMG_WIDTH = 224
IMG_HEIGHT = 224

# Min forms a contributor must have filled out to be included in set
MIN_FORMS = 5

# Max forms which can be included from one contributor
MAX_FORMS_PER_WRITER = 6


# Retrieve data for training
def get_data(dataset_path: str, min_forms: int = MIN_FORMS, max_forms_per_writer: int = MAX_FORMS_PER_WRITER)\
        -> tuple[list[str], list[str]]:
    writer_dirs = glob.glob(os.path.join(dataset_path, "*"))
    author2imgs = [(os.path.split(writer_dir)[1], glob.glob(os.path.join(writer_dir, "*")))
                   for writer_dir in writer_dirs]

    print("Getting data from dataset...")
    filenames = []
    writers = []
    for author, img_files in author2imgs:
        if len(img_files) >= min_forms:
            for img_file in img_files[:max_forms_per_writer]:
                filenames.append(img_file)
                writers.append(author)

    return filenames, writers


# Create an encoder that gives each writer a unique id starting from zero.
# Store the mapping to a json file.
def create_save_encoder(writers: list[str], out_file: str) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(np.asarray(writers))

    with open(out_file, "wb") as key_fp:
        pickle.dump(encoder, key_fp)

    return encoder


# Functions to preprocess images to feed to the CNN
def resize_transform(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w > h:
        ratio = IMG_WIDTH / w
        img.resize((IMG_WIDTH, int(h * ratio)))
    else:
        ratio = IMG_HEIGHT / h
        img.resize((int(w * ratio), IMG_HEIGHT))

    tmp = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT))
    tmp.paste(img, (0, 0))
    return tmp


def transform_images(img_arrays: list[np.ndarray]) -> np.ndarray:
    return np.array(img_arrays).reshape(len(img_arrays), IMG_WIDTH, IMG_HEIGHT, 3).astype("float32") / 255.0


# Extract paragraphs and words from image files
# paragraph_dir and word_dir must exist
def segment_data(filenames: list[str], writers: list[str], paragraph_dir: str, word_dir: str, le_save_path: str)\
        -> tuple[dict[str, str], LabelEncoder]:
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
        word_filenames = get_words(
            paragraph_img_path, writer_word_dir, prefix=paragraph_prefix, transform_fn=resize_transform)
        if writer in writer2words:
            writer2words[writer].extend(word_filenames)
        else:
            writer2words[writer] = word_filenames

        totalwords += len(word_filenames)

    print(f"Found a total of {totalwords} words.")
    encoder = create_save_encoder(list(writer2words.keys()), le_save_path)
    return writer2words, encoder


# Load encoder from key file
def load_encoder(le_save_path: str) -> LabelEncoder:
    with open(le_save_path, "rb") as key_fp:
        return pickle.load(key_fp)


# Get segmented data, assuming it's already been processed
def get_segmented_data(word_dir: str, le_save_path: str, do_gen_encoder: bool = False)\
        -> tuple[dict[str, str], LabelEncoder]:
    print("Getting segmented images...")
    writer2words = {}
    writer_dirs = glob.glob(os.path.join(word_dir, "*"))
    for writer_dir in writer_dirs:
        _, tail = os.path.split(writer_dir)
        writer2words[tail] = glob.glob(os.path.join(writer_dir, "*"))

    encoder: LabelEncoder
    if do_gen_encoder:
        encoder = create_save_encoder(list(writer2words.keys()), le_save_path)
    else:
        encoder = load_encoder(le_save_path)

    return writer2words, encoder


# Clear SEGMENTS directory and do preprocessing
def clear_and_process_data(segments: str, paragraphs: str, words: str, dataset_path: str, le_save_path: str):
    if os.path.isdir(segments):
        # Clear generated data
        shutil.rmtree(segments)

    # Create data and output directories if not present
    os.makedirs(paragraphs)
    os.makedirs(words)

    filenames, writers = get_data(dataset_path)
    return segment_data(filenames, writers, PARAGRAPHS, WORDS, le_save_path)


if __name__ == "__main__":
    clear_and_process_data(SEGMENTS, PARAGRAPHS, WORDS,
                           DATASET_PATH, LE_SAVE_PATH)
