"""
    Thomas: A program to find the words on the pages from the IAM dataset.
    Largely repurposed from https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet/blob/master/0_handwriting_ocr.ipynb
"""

import sys
sys.path.append("handwritten-text-recognition-for-apache-mxnet/")

from ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from ocr.paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
from ocr.utils.expand_bounding_box import expand_bounding_box
from PIL import Image
from typing import Callable
import mxnet as mx
import numpy as np
import glob
import random
import os


ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()

FORM_X = 1120
FORM_Y = 800

DATA_DIR = os.path.join("data/", "data/")

OUT_DIR = "segments/"
PARAGRAPHS_DIR = os.path.join(OUT_DIR, "paragraphs/")
WORDS_DIR = os.path.join(OUT_DIR, "words/")


# Find bounding box for paragraph
def get_paragraph(img_path: str, out_file: str) -> None:
    paragraph_segmentation_net = SegmentationNetwork(ctx=ctx)
    paragraph_segmentation_net.cnn.load_parameters(
        "model_data/models/paragraph_segmentation2.params", ctx=ctx)
    paragraph_segmentation_net.hybridize()

    with Image.open(img_path) as image:
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
        box = (x, y, w, h)

        segmented_paragraph_size = (700, 700)

        paragraph_img = image.crop(box).resize(segmented_paragraph_size)
        paragraph_img.save(out_file)


# Word segmentation function
def get_words(paragraph_img_path: str, out_dir: str, topk: int = 100, debug: bool = False, prefix: str = "", transform_fn: Callable[[Image.Image], Image.Image] = None)\
        -> list[str]:
    word_segmentation_net = WordSegmentationNet(2, ctx=ctx)
    word_segmentation_net.load_parameters(
        "model_data/models/word_segmentation2.params")
    word_segmentation_net.hybridize()

    min_c = 0.1
    overlap_thres = 0.1

    word_segmented_images = []
    with Image.open(paragraph_img_path) as paragraph_img:
        predicted_bb = predict_bounding_boxes(
            word_segmentation_net, paragraph_img, min_c, overlap_thres, topk, ctx)

        if debug:
            print(f"Found {predicted_bb.shape[0]} words.")

        for j in range(predicted_bb.shape[0]):
            (x, y, w, h) = predicted_bb[j]
            image_w, image_h = paragraph_img.size
            (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)
            x, y = int(x), int(y)
            w, h = x + int(w), y + int(h)
            image = paragraph_img.crop((x, y, w, h))

            if transform_fn is not None:
                image = transform_fn(image)

            out_path = os.path.join(out_dir, f"{prefix}{j}.png")
            image.save(out_path)
            word_segmented_images.append(out_path)

    return word_segmented_images


# Test
# Will output cropped images to OUT_DIR
if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs(PARAGRAPHS_DIR, exist_ok=True)
    os.makedirs(WORDS_DIR, exist_ok=True)

    # Get output for a small number of samples
    print("Selecting samples...")
    img_paths = []
    img_dirs = glob.glob(os.path.join(DATA_DIR, "*"))
    for i in range(10):
        writer_id = random.randint(0, len(img_dirs) - 1)
        writer_dir = img_dirs.pop(writer_id)
        img_file = random.choice(glob.glob(os.path.join(writer_dir, "*")))
        img_paths.append(img_file)

    for i, img_path in enumerate(img_paths):
        print(f"Finding paragraph in image {i}")
        paragraph_img_path = os.path.join(PARAGRAPHS_DIR, f"{i}.png")
        get_paragraph(img_path, paragraph_img_path)

        print(f"Finding words in paragraph...")
        out_dir = os.path.join(WORDS_DIR, f"{i}")
        os.makedirs(out_dir, exist_ok=True)
        get_words(paragraph_img_path, out_dir, debug=True)
