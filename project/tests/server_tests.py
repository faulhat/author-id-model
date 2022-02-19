import os
import unittest
import sys
sys.path.append("..")

from project.server.server import app, db


"""
    Thomas: Unit tests for server package
"""

TEST_DB = "test.db"


class FlaskTests(unittest.TestCase):
    # Run before each test
    def setUp(self) -> None:
        app.config["TESTING"] = True
        # Don't require CSRF verification
        app.config["WTF_CSRF_ENABLED"] = False
        app.config["DEBUG"] = False
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + \
            os.path.join(app.config["BASEDIR"], TEST_DB)
        self.app = app.test_client()
        db.drop_all()
        db.create_all()

        return super().setUp()

    # Run after each test
    def tearDown(self) -> None:
        return super().tearDown()

    ## Tests ##

    # Test to see if querying the server works.
    def test_root(self) -> None:
        response = self.app.get("/", follow_redirects=True)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
