
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.metrics import accuracy_score, classification_report

# Find the project folder
folder = Path(__file__).resolve().parent

# Load your trained AI model
model = load(folder / "classifier.joblib")

# Test transactions
test_data = [
    ("Iced cappuccino", "Food & Drinks"),
    ("Pizza restaurant", "Food & Drinks"),
    ("Bubble tea shop", "Food & Drinks"),
    ("Burger lunch", "Food & Drinks"),

    ("Taxi ride", "Transportation"),
    ("Bus ticket", "Transportation"),
    ("Train fare", "Transportation"),
    ("Parking fee", "Transportation"),

    ("New handbag", "Shopping"),
    ("Winter jacket", "Shopping"),
    ("Running shoes", "Shopping"),
    ("Phone case", "Shopping"),

    ("Movie ticket", "Entertainment"),
    ("Karaoke session", "Entertainment"),
    ("Gaming subscription", "Entertainment"),
    ("Concert ticket", "Entertainment"),

    ("Fresh vegetables", "Groceries"),
    ("Rice and cooking oil", "Groceries"),
    ("Milk and eggs", "Groceries"),
    ("Supermarket shopping", "Groceries"),
    ("Apartment rent payment", "Housing"),
    ("Electricity utility bill", "Utilities"),
    ("Doctor clinic consultation", "Healthcare"),
    ("University course tuition", "Education"),
    ("Car insurance premium", "Insurance"),
    ("Cloud software subscription", "Subscriptions"),
    ("Hotel accommodation booking", "Travel"),
    ("Charity donation payment", "Other")
]

df = pd.DataFrame(
    test_data,
    columns=["Description", "Actual Category"]
)

# Predict categories
df["Predicted Category"] = model.predict(
    df["Description"]
)

# Calculate accuracy
df["Correct"] = (
    df["Actual Category"] == df["Predicted Category"]
)

accuracy = accuracy_score(
    df["Actual Category"],
    df["Predicted Category"]
)

print("\nSMARTSPEND AI TEST RESULTS")
print("--------------------------------")

print(df.to_string(index=False))

print(f"\nAccuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        df["Actual Category"],
        df["Predicted Category"],
        zero_division=0
    )
)

# Save results
df.to_csv(
    folder / "smartspend_test_results.csv",
    index=False
)

print("\nTest results saved successfully!")
