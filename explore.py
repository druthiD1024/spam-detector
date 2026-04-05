import pandas as pd

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'text'])

print("=" * 50)
print("DATASET LOADED")
print("=" * 50)
print(f"Shape: {df.shape}")          # (5574, 2)
print(f"\nFirst 5 rows:")
print(df.head())

# ─────────────────────────────────────────
# 2. CLASS BALANCE
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("CLASS DISTRIBUTION")
print("=" * 50)
print(df['label'].value_counts())
print()
print(df['label'].value_counts(normalize=True).round(2))

# ─────────────────────────────────────────
# 3. MESSAGE LENGTH ANALYSIS
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("MESSAGE LENGTH BY CLASS")
print("=" * 50)
df['length'] = df['text'].str.len()
print(df.groupby('label')['length'].describe().round(1))
print(f"\nAvg ham  length: {df[df['label'] == 'ham']['length'].mean():.1f} chars")
print(f"Avg spam length: {df[df['label'] == 'spam']['length'].mean():.1f} chars")

# ─────────────────────────────────────────
# 4. SAMPLE MESSAGES
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("SPAM SAMPLES")
print("=" * 50)
for i, row in df[df['label'] == 'spam'].head(3).iterrows():
    print(f"[{i}] {row['text']}\n")

print("=" * 50)
print("HAM SAMPLES")
print("=" * 50)
for i, row in df[df['label'] == 'ham'].head(3).iterrows():
    print(f"[{i}] {row['text']}\n")

# ─────────────────────────────────────────
# 5. ENCODE LABEL AS NUMBER
# ─────────────────────────────────────────
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print("=" * 50)
print("LABEL ENCODING")
print("=" * 50)
print(df[['label', 'label_num']].drop_duplicates())
print(f"\nTotal spam messages : {df['label_num'].sum()}")
print(f"Total ham  messages : {(df['label_num'] == 0).sum()}")

# ─────────────────────────────────────────
# 6. EXTRA SIGNAL CHECKS
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("EXTRA SIGNAL CHECKS")
print("=" * 50)

df['has_url']    = df['text'].str.contains(r'http|www|click', case=False, regex=True).astype(int)
df['has_phone']  = df['text'].str.contains(r'\d{5,}', regex=True).astype(int)
df['excl_count'] = df['text'].str.count(r'!')
df['upper_ratio']= df['text'].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))

for feature in ['has_url', 'has_phone', 'excl_count', 'upper_ratio']:
    spam_val = df[df['label'] == 'spam'][feature].mean()
    ham_val  = df[df['label'] == 'ham'][feature].mean()
    print(f"{feature:15s} → spam avg: {spam_val:.3f}  |  ham avg: {ham_val:.3f}")

# ─────────────────────────────────────────
# 7. FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  Total messages : {len(df)}")
print(f"  Spam           : {df['label_num'].sum()} ({df['label_num'].mean()*100:.1f}%)")
print(f"  Ham            : {(df['label_num']==0).sum()} ({(1-df['label_num'].mean())*100:.1f}%)")
print(f"  Null values    : {df[['label','text']].isnull().sum().sum()}")
print(f"  Duplicates     : {df.duplicated().sum()}")
print("\n✅ EDA complete — ready for Step 3: Preprocessing")
