# author-id-model
A handwriting recongition tool for teachers (specifically Mr. Rice)

This project is divided into two parts, a repository for the Tensorflow model to identify handwriting and one for the web app itself. This is the repository for the model. It contains a Flask server meant to be run on localhost so that the other system can query the model.

To install:
```
git clone https://github.com/tafaulhaber590/author-id-model/ --recurse-submodules &&
  cd author-id-model &&
  python3 -m venv .env &&
  source .env/bin/activate &&
  ./setup.sh &&
  ./params_get.sh &&
  wget https://github.com/tafaulhaber590/author-id-model/releases/download/checkpoint/model-out.zip &&
  unzip model-out.zip
```

To run the server:
```
# With venv active
python server/main.py
```

There will be some warnings about a missing Tensorboard installation, since it is uninstalled deliberately in setup.sh. This is necessary to prevent naming conflicts with mxboard, on which the handwritten-text-recognition submodule depends. It doesn't affect the functioning of the program.
