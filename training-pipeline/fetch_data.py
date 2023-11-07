from huggingface_hub import hf_hub_download
import zipfile
import os

def fetch_data(save_location = "./train_data"):
    REPO_ID = "RHRolun/bird_data"
    ZIP_FILE = "bird_data.zip"

    zip_location = hf_hub_download(repo_id=REPO_ID, filename=ZIP_FILE, repo_type="dataset", local_dir=".", force_download=True)
    
    if not os.path.exists(save_location): 
        os.makedirs(save_location) 
    
    with zipfile.ZipFile(zip_location, 'r') as zip_ref:
        zip_ref.extractall(f"{save_location}/bird_data")
    print(os.listdir(save_location))
        
if __name__ == '__main__':
    fetch_data("/train_data")