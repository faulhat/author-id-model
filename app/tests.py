"""
    Thomas: Testing for this package. Returns a bunch of warnings from code I didn't write.
    It's not my fault.
"""

import pytest
import json

from .train_model import N_FINGERPRINT
from .main import create_app


@pytest.fixture
def app():
    app = create_app()
    app.config.update({"TESTING": True})
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


# Query the app with an image and check that we get back a fingerprint array
def test_get_fingerprint(client):
    with open("data/data/000/a01-000u.png", "rb") as test_img:
        res = client.post(
            "/",
            data={
                "rq_image": test_img,
            },
        )

        data = json.loads(res.data)
        for feature in data:
            assert isinstance(feature, float)

        assert len(data) == N_FINGERPRINT
