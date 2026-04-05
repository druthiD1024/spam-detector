import re
import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────
print("=" * 50)
print("LOADING & TRAINING FINAL MODEL")
print("=" * 50)

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'text'])
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.drop_duplicates().reset_index(drop=True)

stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text   = text.lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

X = df['text']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────
# 2. TRAIN FINAL PIPELINE (full training data)
# ─────────────────────────────────────────
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        preprocessor=preprocess,
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ('clf', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000))
])

pipeline.fit(X_train, y_train)
print(f"Model trained on {len(X_train)} messages ✅")

# ─────────────────────────────────────────
# 3. SAVE MODEL
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("SAVING MODEL")
print("=" * 50)

model_path = 'spam_model.pkl'
joblib.dump(pipeline, model_path)
print(f"Model saved → {model_path}")

# Check file size
import os
size_kb = os.path.getsize(model_path) / 1024
print(f"File size   → {size_kb:.1f} KB")

# ─────────────────────────────────────────
# 4. VERIFY — LOAD BACK & PREDICT
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("VERIFYING — LOADING MODEL BACK")
print("=" * 50)

loaded_model = joblib.load(model_path)
print(f"Model loaded successfully ✅")
print(f"Model type: {type(loaded_model)}")

test_emails = [
    "Congratulations! You won a FREE $1000 prize! Click now!!!",
    "Hi, are we still on for lunch tomorrow at 1pm?",
    "URGENT: Your account is suspended. Verify immediately.",
    "Can you send me the report by end of day?",
    "FREE entry! Text WIN to 87121 now!"
]

print(f"\nPredictions from loaded model:")
print("-" * 50)
for email in test_emails:
    prob  = loaded_model.predict_proba([email])[0][1]
    label = '🚨 SPAM' if prob > 0.5 else '✅ HAM '
    print(f"{label}  ({prob*100:5.1f}%)  {email[:55]}")

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  Model file    : spam_model.pkl")
print(f"  Size          : {size_kb:.1f} KB")
print(f"  Algorithm     : Logistic Regression + TF-IDF")
print(f"  Ready to use  : joblib.load('spam_model.pkl')")

print("\n✅ Model saved — ready for Step 6: Flask API")
