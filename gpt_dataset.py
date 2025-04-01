import os
from loguru import logger


def get_data(file_path="wikipedia_data.txt"):
    # Check if the data file already exists locally
    if os.path.exists(file_path):
        logger.info(f"Loading data from {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    else:
        logger.info("Downloading Wikipedia data...")
        # Load the Wikipedia dataset
        from datasets import load_dataset
        dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
        
        # Extract the 'text' field from the dataset (for example, all rows)
        text_data = dataset['train']['text']
        
        # Join all the text data into a single large string
        text_data = " ".join(text_data)
        
        # Save the text data to a local file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
        logger.info(f"Data saved to {file_path}")
    
    return text_data
    
