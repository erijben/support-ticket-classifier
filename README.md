## Dataset
This project expects a dataset file named **`twcs.csv`** in the project root.  
The dataset is **not included in this repository** because it is too large.

Place `twcs.csv` next to `train_model.py`.

**Expected columns:**
- `text` (message content)
- `inbound` (True/False) — only inbound customer messages are used

---

## Setup (Installation)
### Requirements
- Python **3.9+**
- pip

Install dependencies:
```bash
pip install -r requirements.txt


## Train the model
Run: python train_model.py
Output: A trained model will be saved as: model.pkl

## Test / Predict (CLI)
Run: python predict.py
Then type a message and press Enter.
Type exit to quit.


## Example inputs (copy/paste)

Billing :
I was charged twice for my subscription, please refund me.
My payment failed but I still got charged.
Can you send me an invoice for last month?

Account
I can't log in, it says my password is incorrect.
I forgot my password, how do I reset it?
My account got locked after too many attempts.

Technical
The app keeps crashing after the latest update.
Notifications are not working on Android.
The page is stuck loading and never opens.

Other
Do you offer student discounts?
What is your support phone number?
How can I contact your support team?



##Model Output Explanation
The script prints:
Predicted Category
Top probabilities (confidence)
If the confidence is low (below a threshold), it may return:
Uncertain/Other
This prevents over-confident wrong predictions on ambiguous messages.


##Notes:
The dataset file twcs.csv is intentionally excluded from GitHub via .gitignore.

The project is fully reproducible: training + testing can be done locally using the commands above.

Approach follows the task requirement: TF-IDF + Logistic Regression (traditional NLP baseline).