import re
import os
import sys
import nltk
import joblib
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ─────────────────────────────────────────
# DOWNLOAD NLTK DATA
# ─────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─────────────────────────────────────────
# PREPROCESS (must be before model load)
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
# LOAD MODEL
# ─────────────────────────────────────────
app = Flask(__name__)

try:
    model = joblib.load('spam_model.pkl')
    print("✅ Model loaded successfully", flush=True)
except Exception as e:
    print(f"❌ Model load failed: {e}", flush=True)
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
    results = []
    for text, prob in zip(texts, probs):
        results.append({
            'text':       text[:100],
            'label':      'spam' if prob > 0.5 else 'ham',
            'confidence': round(float(prob), 4),
            'is_spam':    bool(prob > 0.5)
        })
    return jsonify({'count': len(results), 'results': results}), 200


if __name__ == '__main__':
    print("🚀 Starting Spam Detector API...")
    app.run(debug=True, port=5000)
