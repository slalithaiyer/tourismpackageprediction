import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from huggingface_hub import HfApi, hf_hub_download

# Get Hugging Face token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

repo_id = "Lalithas/Tourism-Prediction" # Dataset repo
model_repo_id = "Lalithas/Tourism-Prediction-Model" # Model repo

# Load processed data from Hugging Face Hub
Xtrain = pd.read_csv(hf_hub_download(repo_id=repo_id, filename="Xtrain.csv", repo_type="dataset", token=HF_TOKEN))
ytrain = pd.read_csv(hf_hub_download(repo_id=repo_id, filename="ytrain.csv", repo_type="dataset", token=HF_TOKEN))

# Train a simple model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(Xtrain, ytrain.values.ravel())

# Save model locally
output_dir = "tourism/model"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "best_random_forest_model.joblib")
joblib.dump(model, model_path)

print("✅ Model trained and saved locally.")

# Upload the trained model to Hugging Face Model Hub
try:
    api.repo_info(repo_id=model_repo_id, repo_type="model")
    print(f"✅ Model repository '{model_repo_id}' already exists. Using it.")
except Exception:
    api.create_repo(repo_id=model_repo_id, repo_type="model", private=False, token=HF_TOKEN)
    print(f"✅ Model repository '{model_repo_id}' created successfully.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="best_random_forest_model.joblib",
    repo_id=model_repo_id,
    repo_type="model",
    token=HF_TOKEN
)
print(f"✅ Model uploaded to Hugging Face Model Hub: {model_repo_id}")