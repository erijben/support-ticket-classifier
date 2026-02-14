import os
import re
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# -----------------------------
# 1) Choose dataset automatically
# -----------------------------
DATA_PATH = "twcs.csv" if os.path.exists("twcs.csv") else "sample.csv"
MODEL_PATH = "model.pkl"


# -----------------------------
# 2) Text cleaning
# -----------------------------
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
MULTISPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("&amp;", "and")
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)  # keep letters/numbers/' only
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


# -----------------------------
# 3) Labeling (rule-based weak supervision)
# -----------------------------
BILLING_KWS = [
    "refund", "refunded", "charge", "charged", "charged twice", "double charged",
    "billing", "bill", "invoice", "payment", "paid", "subscription", "price",
    "cancel", "cancellation", "renewal", "renew", "fee", "fees", "overcharged",
    "money", "credit", "debit"
]
TECH_KWS = [
    "error", "crash", "crashing", "bug", "glitch", "issue", "not working",
    "doesn't work", "cant", "can't", "cannot", "freeze", "frozen", "slow",
    "broken", "fails", "failed", "problem", "down"
]
ACCOUNT_KWS = [
    "password", "login", "log in", "sign in", "signin", "account", "email",
    "username", "verify", "verification", "reset", "locked", "lockout",
    "cannot access", "can't access", "hacked"
]

def label_text(text: str):
    t = clean_text(text)

    # priority order helps when text contains mixed signals
    if any(k in t for k in BILLING_KWS):
        return "Billing"
    if any(k in t for k in ACCOUNT_KWS):
        return "Account"
    if any(k in t for k in TECH_KWS):
        return "Technical"
    return "Other"


# -----------------------------
# 4) Load and prepare data
# -----------------------------
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# keep only customer messages
df["inbound"] = df["inbound"].astype(str).str.lower().map({"true": True, "false": False})
df = df[df["inbound"] == True]

# keep only text
df = df[["text"]].dropna()
df["text"] = df["text"].astype(str)
df["label"] = df["text"].apply(label_text)

print("\nLabel distribution (before sampling):")
print(df["label"].value_counts())

# -----------------------------
# 5) Balance classes (optional but useful)
# -----------------------------
TARGET_PER_CLASS = 40000  # you can reduce if your PC is slow (ex: 15000)
balanced = []
for label, g in df.groupby("label"):
    if len(g) > TARGET_PER_CLASS:
        g = g.sample(TARGET_PER_CLASS, random_state=42)
    balanced.append(g)

df_bal = pd.concat(balanced).sample(frac=1, random_state=42)  # shuffle

print("\nLabel distribution (after sampling):")
print(df_bal["label"].value_counts())


# -----------------------------
# 6) Train model
# -----------------------------
print("\nTraining model...")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=150000,
        min_df=2
    )),
    ("clf", LogisticRegression(
        max_iter=300,
        class_weight="balanced"
    ))
])

pipeline.fit(df_bal["text"], df_bal["label"])

with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print(f"\n✅ Model saved as {MODEL_PATH}")
