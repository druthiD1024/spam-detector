import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────
# 1. LOAD & CLEAN DATASET
# ─────────────────────────────────────────
print("=" * 50)
print("LOADING & CLEANING DATA")
print("=" * 50)

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'text'])
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Drop duplicates (we found 403 in EDA)
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
after = len(df)
print(f"Dropped {before - after} duplicates: {before} → {after} messages")

# ─────────────────────────────────────────
# 2. TEXT PREPROCESSING FUNCTION
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("TEXT PREPROCESSING")
print("=" * 50)

stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Step 1: Lowercase
    text = text.lower()
    # Step 2: Remove punctuation & numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Step 3: Tokenize
    tokens = text.split()
    # Step 4: Remove stop words & stem
    tokens = [stemmer.stem(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# Test on a sample
samples = [
    "Congratulations! You WON a FREE prize! Click NOW!!!",
    "Hey, are you coming to the meeting tomorrow at 10am?"
]

print("\nPreprocessing test:")
for s in samples:
    print(f"\n  BEFORE: {s}")
    print(f"  AFTER : {preprocess(s)}")

# ─────────────────────────────────────────
# 3. APPLY PREPROCESSING
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("APPLYING TO FULL DATASET")
print("=" * 50)

df['clean_text'] = df['text'].apply(preprocess)

print(f"Sample cleaned texts:")
print(df[['label', 'clean_text']].head(6).to_string())

# ─────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("TRAIN / TEST SPLIT")
print("=" * 50)

X = df['clean_text']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # keeps spam/ham ratio same in both splits
)

print(f"Training set : {len(X_train)} messages")
print(f"Test set     : {len(X_test)} messages")
print(f"\nTraining spam ratio : {y_train.mean()*100:.1f}%")
print(f"Test     spam ratio : {y_test.mean()*100:.1f}%")

# ─────────────────────────────────────────
# 5. TF-IDF VECTORIZATION
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("TF-IDF VECTORIZATION")
print("=" * 50)

vectorizer = TfidfVectorizer(
    max_features=10000,     # top 10k words
    ngram_range=(1, 2),     # unigrams + bigrams
    sublinear_tf=True       # log(tf) scaling
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"Vocabulary size : {len(vectorizer.vocabulary_)}")
print(f"Train matrix    : {X_train_tfidf.shape}")
print(f"Test  matrix    : {X_test_tfidf.shape}")

# Top features by TF-IDF score
feature_names = vectorizer.get_feature_names_out()
print(f"\nSample features (first 20): {list(feature_names[:20])}")

# ─────────────────────────────────────────
# 6. SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  Messages after dedup : {len(df)}")
print(f"  Train size           : {X_train_tfidf.shape[0]}")
print(f"  Test  size           : {X_test_tfidf.shape[0]}")
print(f"  Features (TF-IDF)    : {X_train_tfidf.shape[1]}")
print(f"  Stratified split     : ✅")
print(f"  Preprocessing        : lowercase → strip → stop words → stem")

print("\n✅ Preprocessing complete — ready for Step 4: Train Models")
