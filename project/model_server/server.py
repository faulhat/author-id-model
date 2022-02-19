from flask import Flask, Response, make_response

"""
    Thomas: Flask server for querying the handwriting id model.
    Meant to be run locally.
"""

app = Flask(__name__)
app.config["BASEDIR"] = ".."


@app.route("/")
def index() -> Response:
    return make_response("Hello there")


if __name__ == "__main__":
    app.run()
