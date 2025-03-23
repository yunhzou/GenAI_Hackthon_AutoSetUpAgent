import json
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

# Download the image
image_url = "https://unsplash.com/photos/russian-blue-cat-wearing-yellow-sunglasses-yMSecCHsIBc"
response = requests.get(image_url)
webpage_content = response.text

# Extract the actual image URL from the webpage
import re
img_pattern = r'https://images\.unsplash\.com/photo-[^"\']*'
img_matches = re.findall(img_pattern, webpage_content)
if img_matches:
    actual_image_url = img_matches[0]
    print(f"Found image URL: {actual_image_url}")
    # Download the actual image
    img_response = requests.get(actual_image_url)
    image = Image.open(BytesIO(img_response.content))
else:
    raise Exception("Could not find image URL in the webpage")

# Load model and processor
model_name = "google/efficientnet-b0"
processor = EfficientNetImageProcessor.from_pretrained(model_name)
model = EfficientNetForImageClassification.from_pretrained(model_name)

# Process image
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get prediction
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]

# Get top 5 predictions
topk_values, topk_indices = torch.topk(logits, 5)
topk_values = topk_values.squeeze().tolist()
topk_indices = topk_indices.squeeze().tolist()

# Create results
results = {
    "top_prediction": {
        "label": predicted_label,
        "score": float(logits.softmax(dim=-1)[0, predicted_class_idx].item())
    },
    "top_5_predictions": [
        {
            "label": model.config.id2label[idx],
            "score": float(score)
        }
        for idx, score in zip(topk_indices, topk_values)
    ]
}

# Save to JSON file
with open("result.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to result.json")