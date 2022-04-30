import os

# hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model

from urllib.request import urlretrieve

import logging


class TranslatorClassifier:

    def __init__(self, model_path):
        logging.info("TranslatorClassifier class initialized")
        self.model = load_model(model_path)
        logging.info("Model is loaded!")


    def predict(self, sentence, eng_tokenizer,fr_tokenizer,max_eng):
        from keras.preprocessing.sequence import pad_sequences
        import numpy as np

        y_id_to_word = {value: key for key, value in fr_tokenizer.word_index.items()}
        y_id_to_word[0] = '<PAD>'

        sentence = eng_tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen=max_eng, padding='post')
        # predict the sentence which is in english
        predictions = self.model.predict(sentence)

        result = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])

        #returns the predicted sentece which is in french
        return result

def main():
    model = TranslatorClassifier('machine_translation_model.h5')

    import pickle
    with open('eng_tokenizer.pickle', 'rb') as handle:
        eng_tokenizer = pickle.load(handle)
    with open('fr_tokenizer.pickle', 'rb') as handle:
        fr_tokenizer = pickle.load(handle)

    predicted_class = model.predict("Hello there!",eng_tokenizer,fr_tokenizer,15)
    logging.info("The french translated sentence with respect to our model is:\n {}".format(predicted_class))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()