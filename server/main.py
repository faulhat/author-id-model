"""
    Thomas: this program just serves as glue code between the server for the app (written
    in Go) and the ID model.
"""

import json, os
from flask import Flask, request
from PIL import Image

from .model.word_segmentation import *
from .model.test_model import getAvgOutputImgs
from .model.fingerprint_test import get_fingerprint_model
from .model.segment_data import LE_SAVE_PATH
from .model.train_model import SAVED_MODEL

app = Flask(__name__)
MODEL = get_fingerprint_model(SAVED_MODEL)


@app.route("/eval", methods=["POST"])
def evaluate():
    image = Image.open(request.files["rq_image"])
    paragraph = get_paragraph_img(image)
    word_imgs = get_word_imgs(paragraph)
    fingerprint = getAvgOutputImgs(MODEL, word_imgs).tolist()
    
    return json.dumps(fingerprint)


if __name__ == "__main__":
    app.run()
