from flask import Flask, render_template, request
import joblib
import string
import numpy as np
import nltk
import os

# Pre-download NLTK resources and check if they exist
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

app = Flask(__name__)

MODEL_PATH = 'model/spam_model.pkl'
VECTORIZER_PATH = 'model/tfidf_vectorizer.pkl'

# Print model file information for debugging
try:
    print(f"Model exists: {os.path.exists(MODEL_PATH)}")
    print(f"Vectorizer exists: {os.path.exists(VECTORIZER_PATH)}")
    if os.path.exists(MODEL_PATH):
        print(f"Model file size: {os.path.getsize(MODEL_PATH)} bytes")
    if os.path.exists(VECTORIZER_PATH):
        print(f"Vectorizer file size: {os.path.getsize(VECTORIZER_PATH)} bytes")
        
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

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
        confidence = proba[1] if pred == 1 else proba[0]
    return render_template('index.html', result=result, confidence=confidence, proba=proba)

if __name__ == '__main__':
    app.run(debug=True)