import json
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

# Download the image - using the direct image URL from the Unsplash page
image_url = "https://images.unsplash.com/photo-1533738363-b7f9aef128ce?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1935&q=80"
response = requests.get(image_url)
image_content = response.content

# Load the image
image = Image.open(BytesIO(image_content))

# Load the model and processor
model_name = "google/efficientnet-b0"
processor = EfficientNetImageProcessor.from_pretrained(model_name)
model = EfficientNetForImageClassification.from_pretrained(model_name)

# Process the image
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class and probabilities
logits = outputs.logits
probabilities = torch.nn.functional.softmax(logits, dim=-1)
predicted_class_idx = probabilities.argmax(-1).item()
predicted_class = model.config.id2label[predicted_class_idx]
confidence = probabilities[0][predicted_class_idx].item()

# Create result dictionary
result = {
    "predicted_class": predicted_class,
    "confidence": confidence,
    "class_id": predicted_class_idx,
    "top_5_predictions": []
}

# Get top 5 predictions
topk_values, topk_indices = torch.topk(probabilities, 5)
for i in range(5):
    idx = topk_indices[0][i].item()
    result["top_5_predictions"].append({
        "class": model.config.id2label[idx],
        "confidence": topk_values[0][i].item(),
        "class_id": idx
    })

# Save to JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print("Processing complete. Results saved to result.json")