# Training GPT Model on Wikipedia Dataset

This project provides a hands-on learning experience for understanding and implementing a GPT (Generative Pre-trained Transformer) model from scratch. By training on Wikipedia data, you'll learn the fundamentals of transformer architecture, language modeling, and deep learning best practices.

## Learning Objectives

By completing this project, you will:
- Understand the core concepts of transformer architecture
- Learn how to implement attention mechanisms
- Master the process of training large language models
- Gain practical experience with PyTorch
- Learn about tokenization and text preprocessing
- Understand model evaluation and inference

## GPT Architecture Overview

The GPT model architecture consists of several key components:

1. **Input Processing**
   - Tokenization using tiktoken
   - Positional encoding
   - Word embeddings

2. **Transformer Blocks**
   - Multi-head self-attention
   - Layer normalization
   - Feed-forward networks
   - Residual connections

3. **Output Layer**
   - Language modeling head
   - Softmax activation

![GPT Architecture](assest/GPT-2-architecture.ppm)

## Prerequisites

Before starting, ensure you have:
- Python 3.7 or higher
- Basic understanding of:
  - Deep learning concepts
  - PyTorch fundamentals
  - Natural Language Processing basics
  - Git version control

## Required Libraries

```bash
conda create -n gpts python=3.12
conda activate gpts
pip install -r requirements.txt
```

Key dependencies:
- PyTorch: Deep learning framework
- Matplotlib: Visualization
- Loguru: Logging
- Tiktoken: Tokenization
- Datasets: Data handling

## Project Structure

```
LLM_Bootcamp_Pretrain_GPT_Wikimedia/
├── gpt_train.py      # Main training script
├── gpt_dataset.py    # Dataset handling
├── transformer.py    # Transformer model implementation
├── infer.py          # Inference script
├── requirements.txt  # Dependencies
└── assets/          # Project resources
```

## Getting Started

1. **Dataset Preparation**
   ```bash
   python gpt_dataset.py
   ```
   This script downloads and preprocesses the Wikipedia dataset.

2. **Model Configuration**
   The model can be configured through parameters in `gpt_train.py`:
   - Model size (number of parameters)
   - Number of layers
   - Attention heads
   - Learning rate
   - Batch size

3. **Training Process**
   ```bash
   python gpt_train.py
   ```
   The training process includes:
   - Data loading and preprocessing
   - Model initialization
   - Training loop with validation
   - Loss tracking and visualization
   - Model checkpointing

4. **Inference**
   ```bash
   python infer.py --input "Your input text here"
   ```
   Test your trained model by generating text from prompts.

## Training Details

### Hyperparameters
- Learning rate: Adjustable based on model size
- Batch size: Optimized for available GPU memory
- Number of epochs: Based on dataset size
- Warmup steps: For stable training
- Gradient clipping: To prevent exploding gradients

### Monitoring
- Training loss
- Validation loss
- Learning rate schedule
- Memory usage
- Training speed (tokens/second)


## Resources for Further Learning

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
