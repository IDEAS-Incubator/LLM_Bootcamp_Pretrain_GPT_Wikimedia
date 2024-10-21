# Training GPT Model on Wikipedia Dataset

This README provides instructions for training a GPT model using the `gpt_train.py` script on a Wikipedia dataset.

## GPT Architecture Model 

![GPT Architecture](assest/GPT-2-architecture.ppm)


## Overview

The `gpt_train.py` script implements a training routine for a Generative Pre-trained Transformer (GPT) model using PyTorch. The model is trained on a dataset derived from Wikipedia, allowing it to generate coherent text based on the patterns learned during training.

## Prerequisites

- Python 3.7 or higher
- PyTorch
- Matplotlib
- Loguru
- Tiktoken
- Access to a Wikipedia dataset (e.g., from [Wikimedia Dumps](https://huggingface.co/datasets/legacy-datasets/wikipedia))

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/bayesianinstitute/gpt_train.git
   cd your-repo
   ```

2. **Install Dependencies:**

   Ensure you have the required libraries installed. You can create a `requirements.txt` file with the following content:

   ```
   torch
   matplotlib
   loguru
   tiktoken
   ```

   Then run:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset:**

    View Code in gpt_dataset which download

## Usage

1. **Configure the Model and Settings:**

   The model configuration and training settings are defined in the `main` function of `gpt_train.py`. You can adjust the parameters in the `GPT_CONFIG_124M` and `OTHER_SETTINGS` dictionaries.

2. **Run the Training Script:**

   Execute the training script:

   ```bash
   python gpt_train.py
   ```

   This will start the training process, logging the training and validation losses, and generating sample outputs at specified intervals.

## Training Process

- The script initializes the model, sets up data loaders for training and validation, and runs the training loop.
- Training losses and validation losses are tracked and can be plotted after training.
- The trained model is saved as `wiki_model.pth`.

## Inference

1. **Run the Inference Script:**

   After training, you can use the `infer.py` script to generate text based on an input prompt. The model must be saved as `model.pth` after training.

   Execute the inference script with an input prompt:

   ```bash
   python LLM/Train/infer.py --input "Your input text here"
   ```

   Replace `"Your input text here"` with the text you want the model to generate from.


## Results

After training, the script generates a plot of training and validation losses, saved as `loss.pdf`.

## Conclusion

This README provides a basic framework for training a GPT model on a Wikipedia dataset using the `gpt_train.py` script. Adjust the parameters and methods as needed for your specific use case.

## References

- [Tiktoken](https://pypi.org/project/tiktoken/)
- [Wikimedia Dumps](https://huggingface.co/datasets/legacy-datasets/wikipedia)