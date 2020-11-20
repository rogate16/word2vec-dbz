from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

story = pd.read_csv("data_input/dbz.csv").story

# function to clean the text
def preprocess(text):
    token = word_tokenize(text)
    token = [text.lower() for text in token]
    token = [text for text in token if text.isalpha()]
    token = [text for text in token if not text in stopwords.words("english")]
    lemma = WordNetLemmatizer()
    token = [lemma.lemmatize(text) for text in token]
    return token

text_clean = [preprocess(text) for text in story]

from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec

phrases = Phrases(text_clean)
phrases = Phraser(phrases)
sentences = phrases[text_clean]

model = Word2Vec(window=2, size=200, sample=6e-5, alpha=0.01, min_alpha=0.0001)
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=30)

model.save("data_input/model")