import requests
import json
from PIL import Image
from io import BytesIO
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch

# Download the image from Unsplash
image_url = "https://unsplash.com/photos/russian-blue-cat-wearing-yellow-sunglasses-yMSecCHsIBc"
response = requests.get(image_url)
webpage_content = response.text

# Extract the actual image URL from the webpage
import re
image_pattern = r'https://images\.unsplash\.com/[^"\'&]*'
image_matches = re.findall(image_pattern, webpage_content)
if image_matches:
    actual_image_url = image_matches[0]
    print(f"Found image URL: {actual_image_url}")
    img_response = requests.get(actual_image_url)
    image = Image.open(BytesIO(img_response.content))
else:
    raise Exception("Could not find image URL in the webpage")

# Load the feature extractor and model
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

# Process the image
inputs = feature_extractor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get the logits and convert to probabilities
logits = outputs.logits
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# Get the top 5 predictions
values, indices = torch.topk(probabilities, 5)

# Get the class labels
id2label = model.config.id2label

# Create the result dictionary
result = {
    "predictions": [
        {
            "label": id2label[idx.item()],
            "score": val.item()
        }
        for val, idx in zip(values[0], indices[0])
    ]
}

# Save the result to a JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print("Results saved to result.json")