import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import requests
from PIL import Image
import json
import io

# Download the image from Unsplash
image_url = "https://unsplash.com/photos/russian-blue-cat-wearing-yellow-sunglasses-yMSecCHsIBc"
response = requests.get(image_url)
webpage_content = response.text

# Extract the actual image URL from the webpage
import re
image_pattern = r'https://images\.unsplash\.com/photo-[^"]*'
image_matches = re.findall(image_pattern, webpage_content)
if image_matches:
    actual_image_url = image_matches[0]
    print(f"Found image URL: {actual_image_url}")
    # Download the actual image
    image_response = requests.get(actual_image_url)
    image = Image.open(io.BytesIO(image_response.content))
else:
    print("Could not find image URL in the webpage")
    exit(1)

# Load the model and processor
model_name = "microsoft/resnet-18"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

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
    "model": model_name,
    "image_url": image_url,
    "predicted_class": predicted_class,
    "predicted_class_id": predicted_class_idx,
    "confidence": confidence,
    "top_5_predictions": []
}

# Get top 5 predictions
topk_values, topk_indices = torch.topk(probabilities, 5, dim=-1)
for i in range(5):
    class_id = topk_indices[0][i].item()
    class_name = model.config.id2label[class_id]
    prob = topk_values[0][i].item()
    result["top_5_predictions"].append({
        "class_id": class_id,
        "class_name": class_name,
        "probability": prob
    })

# Save the result to a JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print("Results saved to result.json")