#!/usr/bin/env bash

source .env/bin/activate

python -m pytest -v --pyargs app.tests
