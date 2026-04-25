from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Get your Hugging Face token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

# Define repository details for tourism prediction dataset
repo_id = "Lalithas/Tourism-Prediction"
repo_type = "dataset"

# Step 1: Check if the repository exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    # Step 2: Create the repository if it doesn't exist
    try:
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=HF_TOKEN)
        print(f"✅ Repository '{repo_id}' created successfully.")
    except HfHubHTTPError as e:
        print(f"❌ Failed to create repository: {e}")

# Step 3: Explicitly upload the tourism.csv file
local_file_path = "/content/tourism_project/data/tourism.csv"
if os.path.exists(local_file_path):
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo="tourism.csv",
        repo_id=repo_id,
        repo_type=repo_type,
        token=HF_TOKEN # Ensure token is used for upload as well
    )
    print(f"📂 File '{os.path.basename(local_file_path)}' uploaded to '{repo_id}'.")
else:
    print(f"❌ Error: Local file '{local_file_path}' not found. Cannot upload to Hugging Face.")
