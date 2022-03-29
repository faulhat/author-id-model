import pytest

from .main import create_app


@pytest.fixture
def app():
    app = create_app()
    app.config.update({"TESTING": True})
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_get_fingerprint(client):
    with open("data/data/000/a01-000u.png", "rb") as test_img:
        res = client.post(
            "/",
            data={
                "rq_image": test_img,
            },
        )

        assert isinstance(res.json(), list[float])
