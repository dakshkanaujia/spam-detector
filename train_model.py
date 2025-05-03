import pandas as pd
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    # Remove punctuation except for currency symbols and numbers
    text = re.sub(r'[^\w\s£$€]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return ' '.join(tokens)

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv', sep='\t', header=None, names=['label', 'message'])
df['message'] = df['message'].apply(preprocess)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
joblib.dump(model, 'model/spam_model.pkl')
joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')

# Optional: print test performance
X_test_tfidf = tfidf.transform(X_test)
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))