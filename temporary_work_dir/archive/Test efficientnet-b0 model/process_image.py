import json
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

# Direct image URL from Unsplash
image_url = "https://images.unsplash.com/photo-1533738363-b7f9aef128ce?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1935&q=80"
print(f"Using image URL: {image_url}")

# Download the image
image_response = requests.get(image_url)
image = Image.open(BytesIO(image_response.content))

# Load the model and processor
processor = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b0")
model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0")

# Process the image
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class and probabilities
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# Get the class label
predicted_label = model.config.id2label[predicted_class_idx]
confidence = probabilities[0][predicted_class_idx].item()

# Get top 5 predictions
top5_indices = probabilities[0].topk(5).indices.tolist()
top5_probs = probabilities[0].topk(5).values.tolist()
top5_labels = [model.config.id2label[idx] for idx in top5_indices]
top5_predictions = [{"label": label, "probability": prob} for label, prob in zip(top5_labels, top5_probs)]

# Create result dictionary
result = {
    "model": "google/efficientnet-b0",
    "image_url": "https://unsplash.com/photos/russian-blue-cat-wearing-yellow-sunglasses-yMSecCHsIBc",
    "top_prediction": {
        "label": predicted_label,
        "probability": confidence
    },
    "top5_predictions": top5_predictions
}

# Save to JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print(f"Results saved to result.json")
print(f"Top prediction: {predicted_label} with confidence {confidence:.4f}")