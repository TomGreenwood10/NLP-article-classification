import string

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from spacy.lang.en.stop_words import STOP_WORDS as stopwords

nlp = spacy.load('en_core_web_md')

# List of categories to classify
CATEGORIES = [
    "baseball", "cryptography", "electronics", "hardware", "medicine",
    "mideast", "motorcycles", "politics", "religion", "space"
]


class NewsgroupsModel:

    def __init__(self):

        self.model = MultinomialNB(alpha=0.1, fit_prior=False)
        self.label_enc = LabelEncoder()
        self.vectorizer = TfidfVectorizer()

    def preprocess_training_data(self, data):

        # Feature matrix
        X = [spacy_tokenizer(text) for (text, label) in data]
        self.vectorizer.fit(X)
        X = self.vectorizer.transform(X)

        # Categories
        y = [label for (text, label) in data]
        self.label_enc.fit(y)
        y = self.label_enc.transform(y)

        return X, y

    def fit(self, X, y):

        self.model.fit(X, y)

    def preprocess_unseen_data(self, data):

        data = self.vectorizer.transform(data)

        return data

    def predict(self, X):

        y_pred = self.model.predict(X)

        return self.label_enc.inverse_transform(y_pred)


def spacy_tokenizer(text):
    """
    Takes a string and returns string of lemmatized tokens without stopwords or punctuation.

    :param text: string -- Input string.
    :return: string -- tokenized string.
    """
    punctuations = string.punctuation
    tokens = nlp(text)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]

    return ' '.join(tokens)
