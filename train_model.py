import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load data
df = pd.read_csv("gesture_data.csv")
X = df.iloc[:, 1:]  # landmark coords
y = df.iloc[:, 0]   # labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as gesture_model.pkl")
print("🎯 Accuracy on test set:", model.score(X_test, y_test))
