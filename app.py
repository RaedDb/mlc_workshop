import os
import logging

from flask import Flask, request, jsonify

from model import TranslatorClassifier

app = Flask(__name__)

# define model path
model_path = 'machine_translation_model.h5'

# create instance
model = TranslatorClassifier(model_path)
logging.basicConfig(level=logging.INFO)


@app.route("/")
def index():
    """Provide simple health check route."""
    return "Welcome To Our Machine Translation Model"


@app.route("/v1/predict", methods=["GET", "POST"])
def predict():
    """Provide main prediction API route. Responds to both GET and POST requests."""
    logging.info("Predict request received!")
    inputted_text = request.args.get("inputted_text")

    import pickle
    with open('eng_tokenizer.pickle', 'rb') as handle:
        eng_tokenizer = pickle.load(handle)
    with open('fr_tokenizer.pickle', 'rb') as handle:
        fr_tokenizer = pickle.load(handle)

    import json
    with open('sentences_len.json', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    eng_len = json_object['eng_len']
    fr_len = json_object['fr_len']


    prediction = model.predict(inputted_text,eng_tokenizer,fr_tokenizer,eng_len)

    logging.info("prediction from model= {}".format(prediction))
    return jsonify({"predicted_class": str(prediction)})


def main():
    """Run the Flask app."""
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()