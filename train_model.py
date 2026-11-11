from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# This script lives in the same folder as app.py and smartspend_training_data.csv.
folder = Path(__file__).resolve().parent
csv_path = folder / "smartspend_training_data.csv"

if not csv_path.exists():
    raise FileNotFoundError(f"Training file not found: {csv_path}")

df = pd.read_csv(csv_path)
required = {"Description", "Category"}
if not required.issubset(df.columns):
    raise ValueError("CSV must have Description and Category columns")

df = df.dropna(subset=["Description", "Category"])
X = df["Description"].astype(str)
y = df["Category"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    MultinomialNB()
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")
print(f"Accuracy on held-out examples: {accuracy_score(y_test, predictions):.1%}")
print("\nResults by category:\n")
print(classification_report(y_test, predictions, zero_division=0))

output = folder / "classifier.joblib"
dump(model, output)
print(f"\nSaved your model to: {output}")
print("Example prediction:", model.predict(["coffee from cafe"])[0])
