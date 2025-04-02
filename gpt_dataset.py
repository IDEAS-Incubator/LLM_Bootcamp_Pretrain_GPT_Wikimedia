import os
from pathlib import Path
from loguru import logger
from datasets import load_dataset
from tqdm import tqdm

def get_data(file_path="wikipedia_data.txt", chunk_size=1000, max_articles=None, stream_mode=True, max_lines=5000):
    file_path = Path(file_path)
    
    # If file exists and stream_mode is enabled, return a generator function
    if file_path.exists() and stream_mode:
        logger.info(f"Loading data from {file_path} in streaming mode (limited to {max_lines} lines)")
        
        def text_generator():
            with open(file_path, 'r', encoding='utf-8') as file:
                for i, line in enumerate(file):
                    if i >= max_lines:
                        break
                    yield line.strip()
        
        return text_generator()
    
    # If file exists and stream_mode is disabled, load data directly (original behavior)
    elif file_path.exists():
        logger.info(f"Loading data from {file_path}")
        if max_lines:
            with open(file_path, 'r', encoding='utf-8') as file:
                return ''.join([next(file) for _ in range(max_lines) if file])
        else:
            return file_path.read_text(encoding="utf-8")

    # If file doesn't exist, download and process the data
    logger.info("Downloading Wikipedia data...")
    dataset = load_dataset("wikipedia", "20220301.en", streaming=True, trust_remote_code=True)
    
    # Process data and save in chunks
    with file_path.open("w", encoding="utf-8") as file:
        chunk = []
        iterator = iter(dataset['train'])
        
        # Use either max_articles or a small default if max_articles is None
        article_limit = max_articles if max_articles else 100
        
        for _ in tqdm(range(article_limit), desc="Processing Wikipedia articles"):
            try:
                item = next(iterator)
                chunk.append(item['text'])
                
                if len(chunk) >= chunk_size:
                    file.writelines(" ".join(chunk) + " ")
                    chunk = []
            except StopIteration:
                break

        # Write any remaining data
        if chunk:
            file.writelines(" ".join(chunk))

    logger.info(f"Data saved to {file_path}")
    
    # Return as generator if stream_mode is enabled
    if stream_mode:
        def text_generator():
            with open(file_path, 'r', encoding='utf-8') as file:
                for i, line in enumerate(file):
                    if i >= max_lines:
                        break
                    yield line.strip()
        
        return text_generator()
    else:
        if max_lines:
            with open(file_path, 'r', encoding='utf-8') as file:
                return ''.join([next(file) for _ in range(max_lines) if file])
        else:
            return file_path.read_text(encoding="utf-8")



def download_data():
    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download

    # Load the Hugging Face token from .env
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Set your username and dataset name (same as in uploading.py)
    USERNAME = "faizack"
    DATASET_NAME = "wikipedia-data"
    FILE_NAME = "wikipedia_data.txt"

    # Set the local directory to save the downloaded file
    DOWNLOAD_DIR = os.getcwd()

    # Create the download directory if it doesn't exist
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    try:
        # Download the file from Hugging Face
        downloaded_file_path = hf_hub_download(
            repo_id=f"{USERNAME}/{DATASET_NAME}",
            filename=FILE_NAME,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=DOWNLOAD_DIR,
            local_dir_use_symlinks=False
        )
        
        print(f"File downloaded successfully to '{downloaded_file_path}'")
        
        # Optionally, verify the file exists
        if os.path.exists(downloaded_file_path):
            file_size = os.path.getsize(downloaded_file_path) / (1024 * 1024)  # Convert to MB
            print(f"File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"Error downloading file: {e}") 

if __name__ == "__main__":
    # Download the data from Hugging Face as long txt file
    download_data()