import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ─────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────
print("=" * 50)
print("LOADING DATA")
print("=" * 50)

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'text'])
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.drop_duplicates().reset_index(drop=True)
print(f"Messages loaded: {len(df)}")

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

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ─────────────────────────────────────────
# 2. BUILD PIPELINES
# ─────────────────────────────────────────
tfidf = lambda: TfidfVectorizer(
    preprocessor=preprocess,
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

nb_pipeline = Pipeline([
    ('tfidf', tfidf()),
    ('clf',   MultinomialNB(alpha=1.0))
])

lr_pipeline = Pipeline([
    ('tfidf', tfidf()),
    ('clf',   LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000))
])

# ─────────────────────────────────────────
# 3. TRAIN & EVALUATE BOTH MODELS
# ─────────────────────────────────────────
models = [
    ('Naive Bayes',          nb_pipeline),
    ('Logistic Regression',  lr_pipeline),
]

results = {}

for name, pipeline in models:
    print("\n" + "=" * 50)
    print(f"MODEL: {name}")
    print("=" * 50)

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Classification report
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"              Predicted Ham  Predicted Spam")
    print(f"  Actual Ham       {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"  Actual Spam      {cm[1][0]:>5}          {cm[1][1]:>5}")

    # False positives / negatives
    fp = cm[0][1]   # ham classified as spam
    fn = cm[1][0]   # spam classified as ham
    print(f"\n  False Positives (ham → spam) : {fp}  ← legit emails wrongly blocked")
    print(f"  False Negatives (spam → ham) : {fn}  ← spam that slipped through")

    # 5-fold cross validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
    print(f"\n  5-Fold CV F1 Scores : {[round(s,3) for s in cv_scores]}")
    print(f"  Mean CV F1          : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    results[name] = {
        'f1':    f1_score(y_test, y_pred),
        'cv_f1': cv_scores.mean(),
        'pipeline': pipeline
    }

# ─────────────────────────────────────────
# 4. PICK WINNER
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print(f"{'Model':<25} {'Test F1':>10} {'CV F1':>10}")
print("-" * 47)
for name, r in results.items():
    print(f"{name:<25} {r['f1']:>10.4f} {r['cv_f1']:>10.4f}")

winner_name = max(results, key=lambda n: results[n]['cv_f1'])
best_pipeline = results[winner_name]['pipeline']
print(f"\n🏆 Winner: {winner_name}")

# ─────────────────────────────────────────
# 5. TEST ON CUSTOM EMAILS
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("CUSTOM PREDICTIONS")
print("=" * 50)

test_emails = [
    "Congratulations! You won a FREE $1000 prize! Click now to claim!!!",
    "Hi, are we still on for lunch tomorrow at 1pm?",
    "URGENT: Your bank account is suspended. Verify immediately.",
    "Can you send me the project report by end of day?",
    "FREE entry to win tickets! Text WIN to 87121 now!"
]

for email in test_emails:
    prob  = best_pipeline.predict_proba([email])[0][1]
    label = '🚨 SPAM' if prob > 0.5 else '✅ HAM '
    print(f"{label}  ({prob*100:5.1f}%)  {email[:60]}")

print(f"\n✅ Training complete — ready for Step 5: Save the Model")
