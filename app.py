import re
import joblib
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ─────────────────────────────────────────
# MUST define preprocess BEFORE loading model
# (joblib needs it to deserialize the pipeline)
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
# 1. LOAD MODEL (once at startup)
# ─────────────────────────────────────────
app   = Flask(__name__)
model = joblib.load('spam_model.pkl')
print("✅ Model loaded successfully")

# ─────────────────────────────────────────
# 2. ROUTES
# ─────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'spam_detector_v1'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Validate input
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Text cannot be empty'}), 400

    # Predict
    prob  = model.predict_proba([text])[0][1]   # spam probability
    label = 'spam' if prob > 0.5 else 'ham'

    return jsonify({
        'label':      label,
        'confidence': round(float(prob), 4),
        'is_spam':    label == 'spam',
        'text':       text[:100]                 # echo back (truncated)
    }), 200


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    data = request.get_json()

    if not data or 'texts' not in data:
        return jsonify({'error': 'Missing "texts" field (list) in request body'}), 400

    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': '"texts" must be a non-empty list'}), 400

    probs  = model.predict_proba(texts)[:, 1]
    results = []
    for text, prob in zip(texts, probs):
        results.append({
            'text':       text[:100],
            'label':      'spam' if prob > 0.5 else 'ham',
            'confidence': round(float(prob), 4),
            'is_spam':    bool(prob > 0.5)
        })

    return jsonify({
        'count':   len(results),
        'results': results
    }), 200


# ─────────────────────────────────────────
# 3. RUN
# ─────────────────────────────────────────
if __name__ == '__main__':
    print("🚀 Starting Spam Detector API...")
    print("   GET  http://localhost:5000/health")
    print("   POST http://localhost:5000/predict")
    print("   POST http://localhost:5000/predict-batch")
    app.run(debug=True, port=5000)
