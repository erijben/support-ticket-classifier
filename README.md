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

The notebook (`solution.ipynb`) implements a robust classification pipeline with the following improvements:

### Data Processing
1. **Load Data**: Read `twcs.csv` and filter inbound customer messages
2. **Weak Supervision Labeling**: Generate labels using keyword-based rules for Billing, Technical, Account
3. **Enhanced Text Cleaning**:
   - Convert emojis to text descriptions
   - Remove URLs, mentions, hashtags
   - Filter non-English characters
   - Remove stopwords (NLTK)
   - Apply stemming (Porter Stemmer)

### Label Leakage Mitigation ⚠️
**Critical improvement**: To avoid circular reasoning where the model simply learns the keywords used for labeling:
- All weak supervision keywords are **excluded from TF-IDF feature space**
- This forces the model to learn from contextual patterns rather than direct keyword matching
- Custom stop words = English stopwords + labeling keywords

### Model Training & Validation
4. **Three-way Split**: Train (70%), Validation (15%), Test (15%) - stratified
5. **TF-IDF Vectorization**: Unigrams + bigrams (1,2) with labeling keywords excluded
6. **Logistic Regression**: Balanced class weights, L-BFGS solver
7. **5-Fold Cross-Validation**: Robust performance estimation on training set
8. **Validation Monitoring**: Track overfitting with held-out validation set

### Evaluation
9. **Comprehensive Metrics**:
   - Accuracy
   - Macro Precision
   - Macro Recall  
   - Macro F1-Score
   - Confusion Matrix (fixed label order)
   - Classification Report (per-class metrics)
10. **Model Export**: Save pipeline as `model.pkl`
11. **Demo**: Predictions on first 10 rows from `sample.csv`

## 📈 Key Improvements

✅ **Label leakage mitigation** - Exclude labeling keywords from features  
✅ **Cross-validation** - 5-fold stratified CV for robust evaluation  
✅ **Validation set** - Monitor overfitting during training  
✅ **Larger sample** - 50,000 records (up from 20,000)  
✅ **Stopword removal** - NLTK English stopwords  
✅ **Stemming** - Porter Stemmer normalization  
✅ **Emoji handling** - Convert to text descriptions  
✅ **Non-English filtering** - English characters only

## 🚀 How to Run

### Requirements
- Python 3.8+
- Dependencies: pandas, numpy, scikit-learn, joblib, nltk, emoji

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### Execution

1. Open `solution.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab
2. Click **Run All** to execute the entire pipeline
3. Expected output:
   - Cross-validation scores
   - Validation set performance
   - Test set metrics (accuracy, precision, recall, F1, confusion matrix)
   - Demo predictions on sample data
   - Saved files: `model.pkl` and `tfidf_vectorizer.pkl`

**Note**: First run will download NLTK stopwords data automatically.

## 📝 Methodology Notes

### Weak Supervision
Since `twcs.csv` lacks ground-truth labels, we use keyword-based weak supervision. Each message is scored against three keyword sets (Billing, Technical, Account). The category with the highest unique score becomes the label. Messages with ties or no matches are excluded.

### Label Leakage Prevention
A critical challenge in weak supervision is **label leakage**: if we label using keywords and then TF-IDF learns those same keywords, the model is just pattern-matching our labeling rules (circular reasoning).

**Our solution**: Exclude all labeling keywords from the TF-IDF feature vocabulary. This forces the classifier to learn from:
- Contextual patterns
- Adjacent words and phrases
- Bigrams that don't directly contain labeling keywords
- Semantic relationships

This creates a more generalizable model that doesn't simply memorize the labeling heuristics.

## 🎓 Submission

- **Deliverable**: Single Jupyter notebook (`solution.ipynb`)
- **Model**: TF-IDF + Logistic Regression with label leakage mitigation
- **Dataset**: twcs.csv (50k sample) + sample.csv (demo)
- **Validation**: 5-fold CV + separate validation/test sets
- **Output**: Comprehensive metrics + saved model artifacts