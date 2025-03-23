import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
from io import BytesIO
import json

# Download the image from Unsplash
image_url = "https://unsplash.com/photos/russian-blue-cat-wearing-yellow-sunglasses-yMSecCHsIBc"
response = requests.get(image_url)
image_content = response.content

# Check if we got the actual image or the webpage
if b"<!DOCTYPE html>" in image_content:
    # If we got HTML, we need to extract the actual image URL
    print("Got HTML instead of image, extracting actual image URL...")
    # Try to get the actual image URL from the download link
    response = requests.get("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1686&q=80")
    image_content = response.content

# Load the image
image = Image.open(BytesIO(image_content))

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

# Process the image
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
predicted_class = model.config.id2label[predicted_class_idx]

# Get the probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)
top_5_probs, top_5_indices = torch.topk(probs, 5)
top_5_labels = [model.config.id2label[idx.item()] for idx in top_5_indices[0]]
top_5_probs = top_5_probs[0].tolist()

# Create result dictionary
result = {
    "top_prediction": {
        "class": predicted_class,
        "confidence": probs[0][predicted_class_idx].item()
    },
    "top_5_predictions": [
        {"class": label, "confidence": prob} 
        for label, prob in zip(top_5_labels, top_5_probs)
    ]
}

# Save to JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print(f"Prediction complete. Top prediction: {predicted_class}")
print("Results saved to result.json")