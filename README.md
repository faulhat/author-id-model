# author-id
A handwriting recongition tool for teachers (specifically Mr. Rice)

General design:

Users should be able to upload labelled handwriting samples to be stored on the server side. They should also be able to upload unlabelled samples. The server will compare an unlabelled sample to the submitter's labelled samples and then return a response to the client containing a ranked list of some number of candidates for its author.

Current plan:

1. Use Keras with TensorFlow backend to generate a convolutional neural net which will take an image padded to certain dimensions and produce a fingerprint. Fingerprints for images by the same author should be closer together in n-dimensional space than those from different authors ([I will use the dataset found here](https://www.kaggle.com/tejasreddy/iam-handwriting-top50)).

2. Create a very simple Flask server that allows a client to query the model with an image and get back a response as JSON.

3. Create another Flask server with a REST API to serve the application.

4. Add HTML templates to the application server for a Vue.js web application.

6. Write a mobile frontend with Xamarin using the REST API (optional endgame step).
