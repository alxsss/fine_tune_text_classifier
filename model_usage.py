import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./fine-tuned-model-20k-accelarator')
tokenizer = AutoTokenizer.from_pretrained('./fine-tuned-model-20k-accelarator')

# Define the classify function
def classify_newsgroup(newsgroup_text):
    inputs = tokenizer(newsgroup_text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_idx = torch.argmax(logits, dim=-1).item()
    predicted_class_name = model.config.id2label[predicted_class_idx]
    return predicted_class_name

# Test the model with a new article
new_article = "This is a newsgroup post about computer graphics and rendering techniques. It discusses advanced 3D modeling algorithms and GPU optimizations for real-time rendering."
predicted_label = classify_newsgroup(new_article)
print(f"Predicted category: {predicted_label}")
