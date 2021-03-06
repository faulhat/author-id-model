"""
    Thomas: this program just serves as glue code between the server for the app and the ID model.
"""

import json, os
from flask import Flask, request, Blueprint
from PIL import Image

from .word_segmentation import *
from .test_model import getAvgOutputImgs
from .fingerprint_test import get_fingerprint_model
from .train_model import SAVED_MODEL

CONFIG = "config.json"

views = Blueprint("allviews", __name__)
MODEL = get_fingerprint_model(SAVED_MODEL)


# Load program configuration. This should be auto-generated by setup.sh.
# Default settings will be provided if config.json is absent.
def get_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as config:
            return json.load(config)

    return {
        "port": 8080,
        "debug": True,
    }


# Return the fingerprint of an image in JSON format
@views.route("/", methods=["POST"])
def evaluate():
    image = Image.open(request.files["rq_image"])
    paragraph = get_paragraph_img(image)  # Will resize image automatically
    word_imgs = get_word_imgs(paragraph)
    fingerprint = getAvgOutputImgs(MODEL, list(word_imgs), do_resize=True).tolist()

    return json.dumps(fingerprint)


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(views)
    return app


if __name__ == "__main__":
    config = get_config(CONFIG)
    port = config.get("port")
    debug = config.get("debug")
    app = create_app()

    if port is None or not isinstance(port, int):
        app.run(port=8080, debug=debug)
    else:
        app.run(port=port, debug=debug)
