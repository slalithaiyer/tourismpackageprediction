from huggingface_hub import HfApi, create_repo
import os

# Set your Hugging Face token
HF_TOKEN = os.getenv("Tourism_token") # Re-using the previously set token
api = HfApi(token=HF_TOKEN)

# Define your Space details
space_id = "slalithaiyer/Tourism-Prediction-App1" # You can customize this
space_sdk = "streamlit" # Changed from "streamlit" to "docker"
space_hardware = "cpu-basic"

# 1. Create the Hugging Face Space if it doesn't exist
try:
    api.repo_info(repo_id=space_id, repo_type="space")
    print(f"✅ Space '{space_id}' already exists. Using it.")
except Exception:
    print(f"🚀 Creating new Space: {space_id}...")
    create_repo(repo_id=space_id, repo_type="space", private=False, space_sdk=space_sdk, space_hardware=space_hardware, token=HF_TOKEN)
    print(f"✅ Space '{space_id}' created successfully.")

# 2. Define the local paths to the deployment files
deployment_dir = "/content/tourism_project/deployment"
files_to_upload = [
    os.path.join(deployment_dir, "Dockerfile"),
    os.path.join(deployment_dir, "requirements.txt"),
    os.path.join(deployment_dir, "app.py"),
]

# 3. Upload each file to the Hugging Face Space
print("📤 Uploading deployment files to Hugging Face Space...")
for file_path in files_to_upload:
    if os.path.exists(file_path):
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=space_id,
            repo_type="space",
            commit_message=f"Add {os.path.basename(file_path)}"
        )
        print(f"  - Uploaded {os.path.basename(file_path)}")
    else:
        print(f"  ❌ Warning: File not found: {file_path}")

print(f"🎉 Deployment files uploaded to Space '{space_id}'.")
print(f"Your Streamlit app should be available at: https://huggingface.co/spaces/{space_id}")
