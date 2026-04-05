import re
import sys
import nltk
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────
# DOWNLOAD NLTK DATA
# ─────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ─────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────
stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text   = text.lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# ─────────────────────────────────────────
# TRAIN MODEL AT STARTUP
# ─────────────────────────────────────────
app = Flask(__name__)

try:
    print("Training model...", flush=True)
    df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'text'])
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df = df.drop_duplicates().reset_index(drop=True)

    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            preprocessor=preprocess,
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000))
    ])

    model.fit(df['text'], df['label_num'])
    print("✅ Model trained successfully", flush=True)
except Exception as e:
    print(f"❌ Training failed: {e}", flush=True)
    sys.exit(1)

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'spam_detector_v1'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field'}), 400
    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Text cannot be empty'}), 400
    prob  = model.predict_proba([text])[0][1]
    label = 'spam' if prob > 0.5 else 'ham'
    return jsonify({
        'label':      label,
        'confidence': round(float(prob), 4),
        'is_spam':    label == 'spam',
        'text':       text[:100]
    }), 200


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'Missing "texts" field'}), 400
    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': '"texts" must be a non-empty list'}), 400
    probs   = model.predict_proba(texts)[:, 1]
    results = [{'text': t[:100], 'label': 'spam' if p > 0.5 else 'ham',
                'confidence': round(float(p), 4), 'is_spam': bool(p > 0.5)}
               for t, p in zip(texts, probs)]
    return jsonify({'count': len(results), 'results': results}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)
