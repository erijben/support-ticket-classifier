import pickle
import re
import html

MODEL_PATH = "model.pkl"
CONF_THRESHOLD = 0.60   # change 0.55 / 0.65 selon ce que tu veux
TOP_K = 4               # top probas à afficher

def clean_text(text: str) -> str:
    # decode HTML entities (&amp; -> &)
    text = html.unescape(text)

    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # remove @mentions (Twitter style)
    text = re.sub(r"@\w+", " ", text)

    # keep basic punctuation, remove weird extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("✅ Support Ticket Topic Classifier")
print("Type a message and press Enter.")
print("Commands: 'exit' to quit, 'help' for tips.\n")

labels = list(model.classes_)

while True:
    text = input("Enter message: ").strip()

    if text.lower() == "exit":
        print("Bye 👋")
        break

    if text.lower() == "help":
        print("- Example: 'I can't log in to my account'")
        print("- Example: 'I was charged twice, please refund'")
        print("- Example: 'App keeps crashing after update'\n")
        continue

    cleaned = clean_text(text)

    pred = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]

    ranked = sorted(zip(labels, proba), key=lambda x: x[1], reverse=True)
    top1 = ranked[0]

    final_label = pred
    if top1[1] < CONF_THRESHOLD:
        final_label = "Uncertain/Other"

    print(f"Input (cleaned): {cleaned}")
    print(f"Predicted Category: {final_label} (raw: {pred})")

    # show top-k probabilities
    print("Top probabilities:")
    for lab, p in ranked[:TOP_K]:
        print(f"  - {lab}: {p:.2f}")

    print()
