# Support Ticket Topic Classification

Multi-class text classification system for categorizing customer support messages into **Billing**, **Technical**, and **Account** categories using traditional NLP methods.

## 🎯 Challenge

Develop a classifier using **TF-IDF + Logistic Regression** to categorize short customer support messages into three classes:
- **Billing** - Payment, charges, invoices, refunds
- **Technical** - Bugs, crashes, errors, performance issues  
- **Account** - Login, password, profile, security

## 📊 Datasets

- **twcs.csv**: Main dataset (Twitter Customer Support) - used for training. No ground-truth labels provided; labels are generated via weak supervision.
- **sample.csv**: Sample data for demo predictions.

## 🔧 Approach

The notebook (`solution.ipynb`) performs the following steps:

1. **Load Data**: Read `twcs.csv` and filter inbound customer messages
2. **Text Cleaning**: Remove URLs, mentions, hashtags, and special characters
3. **Weak Supervision Labeling**: Generate labels using keyword-based rules
   - Billing keywords: bill, charge, payment, invoice, refund, etc.
   - Technical keywords: error, crash, bug, slow, battery, etc.
   - Account keywords: password, login, account, reset, etc.
4. **Data Preparation**: Filter labeled rows and create stratified train/test split
5. **Feature Engineering**: TF-IDF vectorization with unigrams + bigrams (1,2)
6. **Model Training**: Logistic Regression classifier with balanced class weights
7. **Evaluation**: Compute and display comprehensive metrics
8. **Export**: Save trained model and vectorizer as `.pkl` files
9. **Demo**: Run predictions on first 10 rows from `sample.csv`

## 📈 Metrics

The notebook outputs the following evaluation metrics:

- **Accuracy**
- **Macro Precision**
- **Macro Recall**
- **Macro F1-Score**
- **Confusion Matrix**
- **Classification Report** (per-class precision, recall, F1)

## 🚀 How to Run

### Requirements
- Python 3.8+
- Dependencies: pandas, numpy, scikit-learn, joblib

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install pandas numpy scikit-learn joblib
```

### Execution

1. Open `solution.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab
2. Click **Run All** to execute the entire pipeline
3. Expected output:
   - Printed metrics (accuracy, precision, recall, F1, confusion matrix)
   - Demo predictions on sample data
   - Saved files: `model.pkl` and `tfidf_vectorizer.pkl`


## 📝 Notes

**Weak Supervision**: Since `twcs.csv` does not contain ground-truth labels, we use a reproducible keyword-based weak supervision approach. Each message is scored against three keyword sets (Billing, Technical, Account). The category with the highest score is assigned as the label. Messages with no matches or tied scores are excluded to maintain label quality.

This approach provides a fast, reproducible baseline for classification without manual annotation.

## 🎓 Submission

- **Deliverable**: Single Jupyter notebook (`solution.ipynb`) that runs end-to-end
- **Model**: TF-IDF + Logistic Regression
- **Dataset**: twcs.csv (training) + sample.csv (demo)
- **Output**: Comprehensive metrics + saved model artifacts