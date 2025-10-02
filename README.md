📑 AG News Topic Classifier (BERT)
📌 Overview

This project implements a news headline classifier using the AG News dataset and a BERT transformer model. The model classifies headlines into one of four categories:

🌍 World

🏟️ Sports

💼 Business

🔬 Sci/Tech

The model was fine-tuned using Hugging Face Transformers in Google Colab and deployed with Gradio for live interaction.

🚀 Features

Fine-tuned bert-base-uncased on AG News dataset

Achieves competitive performance with Accuracy and F1-score

Provides an interactive Gradio demo for real-time classification

Includes training notebook and saved model for reproducibility

📂 Repository Structure
├── Task1_AGNews_BERT.ipynb     # Jupyter notebook (Google Colab)
├── bert-ag-news-model/         # Fine-tuned model (saved weights & config)
├── README.md                   # Project description (this file)

🛠️ Tech Stack

Python 3

PyTorch

Hugging Face Transformers

Datasets (AG News)

Gradio (for demo)

⚡ How to Run
1️⃣ Clone the repo
git clone https://github.com/yourusername/agnews-bert-classifier.git
cd agnews-bert-classifier

2️⃣ Install dependencies
pip install -U transformers datasets accelerate evaluate gradio torch

3️⃣ Run training (optional)

Open the notebook Task1_AGNews_BERT.ipynb in Google Colab, enable GPU, and run all cells.

4️⃣ Launch demo

After training:

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "bert-ag-news-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

label_mapping = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

def classify_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=-1).item()
    return f"Predicted Topic: {label_mapping[pred]}"

demo = gr.Interface(fn=classify_news, inputs="text", outputs="text", title="AG News Classifier (BERT)")
demo.launch()

📊 Results
Metric	Score
Accuracy	~XX%
F1 (macro)	~XX%

(Replace XX with your actual evaluation results from trainer.evaluate())

✨ Example Predictions

Input: "NASA announces discovery of new exoplanet" → Sci/Tech

Input: "Wall Street rises after strong earnings report" → Business

Input: "UN discusses global peace mission" → World

Input: "Lionel Messi scores twice in Copa America" → Sports

📌 Author

Developed as part of AI/ML Engineering Advanced Internship by DeveloperHub_Corporation
