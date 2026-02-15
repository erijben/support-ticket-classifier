# 🎫 Support Ticket Classifier

> **Challenge**: Multi-class text classification using traditional NLP (TF-IDF + Logistic Regression)

Classify customer support messages into **Billing**, **Technical**, or **Account** categories.

---



## 🔍 Class Imbalance & Improving "Account" Precision

During evaluation, the **Account** class initially showed low precision due to **class imbalance** (fewer Account samples compared to other categories).  
To address this, I applied **imbalance handling techniques** (e.g., class balancing / resampling) and adjusted the training pipeline to improve the model’s ability to correctly predict the Account category.

As a result, the **precision for Account increased significantly** compared to the initial baseline, while keeping strong overall performance (Macro F1 and Accuracy).
  
- Before: Account precision ≈ 0.35  
- After:  Account precision ≈ 0.55



## 📊 Challenge Overview

**Task**: Build a classifier that automatically categorizes customer support tickets

**Categories**:
- 💰 **Billing** – Payments, charges, refunds, invoices
- 🔧 **Technical** – Bugs, crashes, errors, app issues
- 👤 **Account** – Login, passwords, authentication, profile access

**Method**: Traditional NLP approach using TF-IDF vectorization + Logistic Regression

---

## 📁 Datasets

### twcs.csv (Training Data)
- **Source**: Twitter Customer Support dataset
- **Size**: ~2.8M total tweets, ~1.5M inbound customer messages
- **Labels**: None provided → generated via weak supervision
- **Note**: ⚠️ **NOT included in GitHub** (too large - 516MB). Download separately and place in project root.

### sample.csv (Demo Data)
- **Purpose**: Demo predictions only
- **Size**: 100 rows
- **Included**: ✅ In repository

---

## 🔧 Solution Pipeline

### 1. Data Loading & Filtering
- Load `twcs.csv` 
- Filter to inbound customer messages only

### 2. Weak Supervision Labeling
Since twcs.csv has no ground-truth labels, we generate them using keyword-based rules:

**Labeling Strategy**:
- Define keyword sets for Billing, Technical, Account
- Use **priority rules** to reduce overlap (e.g., strong Account signals like "login", "password" override weak matches)
- Drop ambiguous cases (ties)

**Result**: ~550k labeled messages

### 3. Text Preprocessing
- Remove URLs, mentions (@), hashtags (#)
- Lowercase & remove special characters
- Remove stopwords (custom list, no external dependencies)
- Keep only words > 2 characters

### 4. Label Leakage Mitigation 🔥
**Critical improvement**: Exclude all labeling keywords from TF-IDF features

**Why?** If we label with keywords AND let TF-IDF learn those keywords → circular reasoning!

**Solution**: Remove labeling keywords from feature space → model learns from context instead

### 5. TF-IDF Vectorization
- **Bigrams**: (1,2) for context
- **Max features**: 6000
- **Stopwords**: Custom list + labeling keywords
- **Parameters**: min_df=3, max_df=0.7

### 6. Model Training
- **Algorithm**: Logistic Regression
- **Class weights**: `class_weight='balanced'` to handle imbalanced data
- **Hyperparameter tuning**: GridSearchCV on C ∈ [0.5, 1.0, 2.0, 4.0]
- **Validation**: 5-fold cross-validation + separate validation set

### 7. Evaluation
Print comprehensive metrics:
- ✅ **Accuracy**
- ✅ **Macro Precision**
- ✅ **Macro Recall**
- ✅ **Macro F1-Score**
- ✅ **Confusion Matrix**
- ✅ **Classification Report** (per-class metrics)

### 8. Export
- `model.pkl` – Complete pipeline (TF-IDF + classifier)
- `tfidf_vectorizer.pkl` – Vectorizer alone

### 9. Demo
- Predict on first 10 rows from `sample.csv`
- Display text + predicted category

---

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- Only **core libraries** (no external NLP dependencies)

### Setup

```bash
# 1. Clone repository
git clone https://github.com/erijben/support-ticket-classifier.git
cd support-ticket-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download twcs.csv
# ⚠️ NOT included in repo (516MB)
# Download from: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
# Place twcs.csv in project root directory

# 4. Run the notebook
# Open solution.ipynb in Jupyter/VS Code/Colab
# Click "Run All"
```

### Expected Output
- Cross-validation scores
- Validation set performance
- **Test set metrics** (final results)
- Confusion matrix
- Demo predictions
- Saved files: `model.pkl`, `tfidf_vectorizer.pkl`

---

## 📈 Metrics Displayed

The notebook prints the following metrics on the **test set**:

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of correct predictions |
| **Macro Precision** | Average precision across all classes |
| **Macro Recall** | Average recall across all classes |
| **Macro F1-Score** | Harmonic mean of precision & recall |
| **Confusion Matrix** | Class-by-class prediction breakdown |
| **Classification Report** | Per-class precision, recall, F1 |

**Special focus**: Account class precision (target: >55%)

---

## 💡 Key Improvements

### Problem: Low Account Precision (~35%)

**Root Causes**:
1. Class imbalance (Account 12%, Technical 61%, Billing 27%)
2. Account/Technical keyword overlap
3. Weak labeling rules

**Solutions Implemented**:
1. ✅ **Priority-based labeling** – Strong Account signals override weak matches
2. ✅ **Balanced class weights** – `class_weight='balanced'` in Logistic Regression
3. ✅ **Refined keywords** – Focus on authentication issues (login, password, locked)
4. ✅ **Label leakage mitigation** – Exclude labeling keywords from features

---

## 🛠️ Technical Stack

**Dependencies** (minimal):
- `pandas` – Data manipulation
- `numpy` – Numerical operations  
- `scikit-learn` – TF-IDF, Logistic Regression, metrics
- `joblib` – Model persistence
- `re` (built-in) – Text cleaning

**No external NLP libraries** (nltk, spacy, emoji, imbalanced-learn) – works everywhere!

---

## 📂 Repository Structure

```
support-ticket-classifier/
├── solution.ipynb          # 👈 Main notebook (run this!)
├── sample.csv             # Demo data (100 rows)
├── model.pkl              # Trained model (generated)
├── tfidf_vectorizer.pkl   # Text vectorizer (generated)
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── twcs.csv              # ⚠️ NOT in repo - download separately
```

---

## 🎓 Submission Info

- **Deliverable**: Single Jupyter notebook (`solution.ipynb`)
- **Method**: TF-IDF + Logistic Regression (as required)
- **Dataset**: twcs.csv (training) + sample.csv (demo)
- **Validation**: Cross-validation + separate test set
- **Metrics**: Comprehensive (4+ metrics + reports)
- **Code**: Clean, reproducible, no complex dependencies

---

## 📝 Notes

### Why Weak Supervision?
`twcs.csv` has no ground-truth labels. We generate labels automatically using keyword matching. This is a practical approach for unlabeled data but has limitations (label noise).

### Label Leakage Prevention
Critical issue in weak supervision: if we label with keywords AND train on those keywords, the model just memorizes our rules.

**Our fix**: Remove all labeling keywords from TF-IDF vocabulary → model learns from context and patterns instead.

### Class Imbalance
Account class is underrepresented (12% vs 61% Technical). We use `class_weight='balanced'` to automatically adjust loss weights during training.

---

## ✨ Quick Start

```bash
pip install -r requirements.txt
# Download twcs.csv to project root
# Open solution.ipynb and click "Run All"
```

That's it! 🚀