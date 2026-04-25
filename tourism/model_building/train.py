import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load processed data
Xtrain = pd.read_csv("tourism/data/Xtrain.csv")
ytrain = pd.read_csv("tourism/data/ytrain.csv")

# Train a simple model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(Xtrain, ytrain.values.ravel())

# Save model locally
output_dir = "tourism/model"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(model, f"{output_dir}/best_random_forest_model.joblib")

print("✅ Model trained and saved locally.")
