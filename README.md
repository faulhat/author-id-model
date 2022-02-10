# author-id
A handwriting recongition tool for teachers (specifically Mr. Rice)

General design:

Users should be able to upload labelled handwriting samples to be stored on the server side. They should also be able to upload unlabelled samples. The server will compare an unlabelled sample to the submitter's labelled samples and then return a response to the client containing a ranked list of some number of candidates for its author.

Current plan:

1. Use Keras with TensorFlow backend to generate a convolutional neural net which will generate fingerprints given handwriting samples ([I will use the dataset found here](https://www.kaggle.com/tejasreddy/iam-handwriting-top50)).

2. Create an HTTP server with Flask that runs on localhost and allows the main program to interface with the AI model. Given image data, it should generate a fingerprint for a submitted unlabelled sample and then compare that fingerprint to the submitter's labelled samples with a k-nearest-neighbor search.

3. Create a public HTTP server with node.js that interacts with the local server mentioned above but handles any non-AI requests directly. It will access a local MySQL server, but will use a Redis store for caching.

4. Write Vue.js templates to be rendered on the web frontend.

5. Write a REST API.

6. Write a mobile frontend with Xamarin using the REST API (optional endgame step).
