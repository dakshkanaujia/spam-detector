from flask import Flask, render_template, request
import joblib
import string
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

app = Flask(__name__)

MODEL_PATH = 'model/spam_model.pkl'
VECTORIZER_PATH = 'model/tfidf_vectorizer.pkl'

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return ' '.join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    proba = None
    if request.method == 'POST':
        message = request.form['message']
        processed = preprocess(message)
        vect = vectorizer.transform([processed])
        proba = model.predict_proba(vect)[0]
        pred = model.predict(vect)[0]
        result = 'Spam' if pred == 1 else 'Not Spam'
        confidence = np.max(proba)
    return render_template('index.html', result=result, confidence=confidence, proba=proba)

if __name__ == '__main__':
    app.run(debug=True)