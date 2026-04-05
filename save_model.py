import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from model_utils import preprocess

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("=" * 50)
print("LOADING & TRAINING FINAL MODEL")
print("=" * 50)

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'text'])
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.drop_duplicates().reset_index(drop=True)

X = df['text']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────
# 2. TRAIN
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
# 3. SAVE
# ─────────────────────────────────────────
import os
joblib.dump(pipeline, 'spam_model.pkl')
size_kb = os.path.getsize('spam_model.pkl') / 1024
print(f"Model saved → spam_model.pkl ({size_kb:.1f} KB)")

# ─────────────────────────────────────────
# 4. VERIFY
# ─────────────────────────────────────────
loaded = joblib.load('spam_model.pkl')
test   = ["FREE prize click now!!!", "See you at lunch tomorrow"]
for t in test:
    prob = loaded.predict_proba([t])[0][1]
    print(f"{'SPAM' if prob>0.5 else 'HAM'} ({prob*100:.1f}%) {t}")

print("\n✅ Model saved successfully!")
