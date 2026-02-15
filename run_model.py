"""
Quick test script to run the classifier and generate metrics
This creates a lightweight version to validate the approach works
"""
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# Don't use emoji/nltk for this quick test
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib

print("Loading data...")
df = pd.read_csv('twcs.csv')
print(f"Total records: {len(df)}")

# Filter inbound
df = df[df['inbound'] == True].copy()
print(f"Inbound messages: {len(df)}")

# Weak supervision keywords
LABELING_KEYWORDS = {
    'billing': ['bill', 'charge', 'payment', 'invoice', 'refund', 'subscription', 
                'pay', 'card', 'credit', 'price', 'cost', 'fee', 'receipt', 'paid',
                'transaction', 'billing', 'overcharge'],
    'technical': ['error', 'crash', 'bug', 'issue', 'problem', 'broken',
                  'slow', 'loading', 'update', 'ios', 'android', 'app', 'website',
                  'battery', 'freeze', 'lag', 'download', 'install', 'version',
                  'device', 'glitch', 'wifi', 'connection'],
    'account': ['account', 'password', 'login', 'username', 'profile',
                'reset', 'verify', 'access', 'locked', 'deactivate', 'email',
                'register', 'authentication', 'security', 'settings']
}

def assign_label(text):
    if pd.isna(text):
        return None
    text_lower = text.lower()
    billing_score = sum(1 for kw in LABELING_KEYWORDS['billing'] if kw in text_lower)
    technical_score = sum(1 for kw in LABELING_KEYWORDS['technical'] if kw in text_lower)
    account_score = sum(1 for kw in LABELING_KEYWORDS['account'] if kw in text_lower)
    scores = [billing_score, technical_score, account_score]
    max_score = max(scores)
    if max_score == 0 or scores.count(max_score) > 1:
        return None
    if billing_score == max_score:
        return 'Billing'
    elif technical_score == max_score:
        return 'Technical'
    else:
        return 'Account'

print("\nLabeling data...")
df['label'] = df['text'].apply(assign_label)
df = df[df['label'].notna()].copy()
print(f"Labeled records: {len(df)}")
print(f"\nLabel distribution:\n{df['label'].value_counts()}")

# Simple cleaning (no emoji/stemming for quick test)
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

print("\nCleaning text...")
df['text_clean'] = df['text'].apply(clean_text)
df = df[df['text_clean'].str.len() > 0].copy()
print(f"Records after cleaning: {len(df)}")

# Sample
SAMPLE_SIZE = min(20000, len(df))  # Smaller for faster execution
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)

X = df_sample['text_clean']
y = df_sample['label']

# Split
print("\nSplitting data...")
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

print(f"Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

# Flatten labeling keywords to exclude
all_labeling_keywords = []
for keywords in LABELING_KEYWORDS.values():
    all_labeling_keywords.extend(keywords)

print(f"\nExcluding {len(all_labeling_keywords)} labeling keywords from TF-IDF features")

# TF-IDF
print("\nVectorizing...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.7,
    stop_words=list(set(all_labeling_keywords))  # CRITICAL: exclude labeling keywords
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF shape: {X_train_tfidf.shape}")

# Train
print("\nTraining model...")
model = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced', random_state=42)
model.fit(X_train_tfidf, y_train)
print("Training complete!")

# Cross-validation
print("\n" + "="*60)
print("5-FOLD CROSS-VALIDATION")
print("="*60)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=cv, scoring='f1_macro')
print(f"CV F1-Scores: {cv_scores}")
print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Validation
print("\n" + "="*60)
print("VALIDATION SET PERFORMANCE")
print("="*60)
y_val_pred = model.predict(X_val_tfidf)
val_acc = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average='macro')
print(f"Accuracy: {val_acc:.4f}")
print(f"Macro F1: {val_f1:.4f}")

# Test set
print("\n" + "="*60)
print("TEST SET PERFORMANCE (FINAL METRICS)")
print("="*60)
y_test_pred = model.predict(X_test_tfidf)

label_order = ['Billing', 'Technical', 'Account']
accuracy = accuracy_score(y_test, y_test_pred)
macro_precision = precision_score(y_test, y_test_pred, average='macro', labels=label_order)
macro_recall = recall_score(y_test, y_test_pred, average='macro', labels=label_order)
macro_f1 = f1_score(y_test, y_test_pred, average='macro', labels=label_order)

print(f"\nAccuracy:           {accuracy:.4f}")
print(f"Macro Precision:    {macro_precision:.4f}")
print(f"Macro Recall:       {macro_recall:.4f}")
print(f"Macro F1-Score:     {macro_f1:.4f}")

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_test_pred, labels=label_order))

print("="*60)
print("CONFUSION MATRIX")
print("="*60)
print(confusion_matrix(y_test, y_test_pred, labels=label_order))
print(f"\nLabel order: {label_order}")

# Save
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("✓ Model saved as model.pkl")
print("✓ Vectorizer saved as tfidf_vectorizer.pkl")

print("\n" + "="*60)
print("EXECUTION COMPLETE!")
print("="*60)
