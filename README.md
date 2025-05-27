# 20 Newsgroups Classification with DistilBERT

This project demonstrates how to fine-tune the **DistilBERT** model on the **20 Newsgroups dataset** for text classification. The model is trained to predict the category of a newsgroup post from 20 different classes, utilizing the pre-trained **distilbert-base-uncased** model from Hugging Face.

## Project Overview
- **Dataset**: 20 Newsgroups dataset from `sklearn.datasets.fetch_20newsgroups`.
- **Model**: **DistilBERT** (`distilbert-base-uncased`), a smaller version of BERT for efficient text classification.
- **Training Setup**: The model is fine-tuned using **PyTorch**, **AdamW optimizer**, and a **learning rate scheduler**. The model was trained for **5 epochs**.
- **Performance**: Achieved an **Evaluation Accuracy** of **85.17%** and an **Evaluation Loss** of **0.55** on the test set.

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- Hugging Face Transformers
- scikit-learn
- Accelerate (for training on GPUs or TPUs)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/20-newsgroups-classification.git
   cd 20-newsgroups-classification
