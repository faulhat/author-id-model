"""
    Thomas: A program to find the words on the pages from the IAM dataset.
    Mostly copied from https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet/blob/master/0_handwriting_ocr.ipynb
"""

import sys
sys.path.append("handwritten-text-recognition-for-apache-mxnet/")

from ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from ocr.paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
from ocr.utils.expand_bounding_box import expand_bounding_box
from PIL import Image
import mxnet as mx
import numpy as np
import glob
import random
import os
from tqdm import tqdm


ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()

FORM_X = 1120
FORM_Y = 800

DATA_DIR = os.path.join("data/", "data/")

OUT_DIR = "test_out/"
PARAGRAPHS_DIR = os.path.join(OUT_DIR, "paragraphs/")
WORDS_DIR = os.path.join(OUT_DIR, "words/")


# Get bounding boxes for paragraphs
def get_paragraphs(images: list[Image.Image], output_imgs: bool = False, debug: bool = False) -> list[Image.Image]:
    paragraph_segmentation_net = SegmentationNetwork(ctx=ctx)
    paragraph_segmentation_net.cnn.load_parameters(
        "model_data/models/paragraph_segmentation2.params", ctx=ctx)
    paragraph_segmentation_net.hybridize()

    predicted_bbs = []

    # Ensure the presence of a progress bar if debug is true
    range_n = range(len(images))
    if debug:
        range_n = tqdm(range_n)
    
    for i, image in zip(range_n, images):
        img_array = np.asarray(image)

        resized_image = paragraph_segmentation_transform(
            img_array, (FORM_X, FORM_Y))
        bb_predicted = paragraph_segmentation_net(
            resized_image.as_in_context(ctx))
        bb_predicted = bb_predicted[0].asnumpy()
        bb_predicted = expand_bounding_box(bb_predicted, expand_bb_scale_x=0.03,
                                           expand_bb_scale_y=0.03)

        (x, y, w, h) = bb_predicted
        image_w, image_h = image.size
        (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)
        x, y = int(x), int(y)
        w, h = x + int(w), y + int(h)

        predicted_bbs.append((x, y, w, h))

    segmented_paragraph_size = (700, 700)
    paragraph_segmented_images = []
    for i, image in enumerate(images):
        bb = predicted_bbs[i]
        paragraph_image = image.crop(bb).resize(segmented_paragraph_size)
        paragraph_segmented_images.append(paragraph_image)

        if output_imgs:
            paragraph_image.save(os.path.join(PARAGRAPHS_DIR, f"{i}.png"))

    return paragraph_segmented_images


# Word segmentation function
def get_words(paragraph_segmented_image: Image.Image, topk: int = 10, out_dir: str = None, debug: bool = False) -> list[Image.Image]:
    word_segmentation_net = WordSegmentationNet(2, ctx=ctx)
    word_segmentation_net.load_parameters(
        "model_data/models/word_segmentation2.params")
    word_segmentation_net.hybridize()

    min_c = 0.1
    overlap_thres = 0.1

    predicted_words_bbs_array = []
    word_segmented_images = []
    predicted_bb = predict_bounding_boxes(
        word_segmentation_net, paragraph_segmented_image, min_c, overlap_thres, topk, ctx)

    predicted_words_bbs_array.append(predicted_bb)

    # Ensure progress bar if debug is true
    range_n = range(predicted_bb.shape[0])
    if debug:
        range_n = tqdm(range_n)
        print("Finding words...")

    for j in range_n:
        (x, y, w, h) = predicted_bb[j]
        image_w, image_h = paragraph_segmented_image.size
        (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)
        x, y = int(x), int(y)
        w, h = x + int(w), y + int(h)
        image = paragraph_segmented_image.crop((x, y, w, h))
        word_segmented_images.append(image)

        if out_dir is not None:
            image.save(os.path.join(out_dir, f"{j}.png"))

    return word_segmented_images


# Test
# Will output cropped images to OUT_DIR
if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs(PARAGRAPHS_DIR, exist_ok=True)
    os.makedirs(WORDS_DIR, exist_ok=True)

    # Get output for a small number of samples
    print("Selecting samples...")
    images = []
    img_dirs = glob.glob(os.path.join(DATA_DIR, "*"))
    for i in tqdm(range(10)):
        writer_id = random.randint(0, len(img_dirs) - 1)
        writer_dir = img_dirs.pop(writer_id)
        img_file = random.choice(glob.glob(os.path.join(writer_dir, "*")))
        img = Image.open(img_file)
        images.append(img)

    print("Finding paragraphs...")
    paragraphs = get_paragraphs(images, output_imgs=True, debug=True)
    for i, paragraph in enumerate(paragraphs):
        out_dir = os.path.join(WORDS_DIR, f"para_{i}")
        os.makedirs(out_dir, exist_ok=True)
        get_words(paragraph, out_dir=out_dir, debug=True)
