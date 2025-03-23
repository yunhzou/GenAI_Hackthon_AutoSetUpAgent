import json
import requests
from PIL import Image
from io import BytesIO
from transformers import DeiTFeatureExtractor, DeiTForImageClassification

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
    raise Exception("Could not find the image URL in the webpage")

# Load the feature extractor and model
feature_extractor = DeiTFeatureExtractor.from_pretrained("facebook/deit-tiny-distilled-patch16-224")
model = DeiTForImageClassification.from_pretrained("facebook/deit-tiny-distilled-patch16-224")

# Process the image
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# Get the predicted class
predicted_class_idx = logits.argmax(-1).item()
predicted_class = model.config.id2label[predicted_class_idx]

# Get the top 5 predictions
import torch
probs = torch.nn.functional.softmax(logits, dim=-1)
top5_prob, top5_indices = torch.topk(probs, 5)

# Create a result dictionary
result = {
    "top_prediction": {
        "class": predicted_class,
        "score": float(probs[0][predicted_class_idx].item())
    },
    "top_5_predictions": [
        {
            "class": model.config.id2label[idx.item()],
            "score": float(prob.item())
        }
        for idx, prob in zip(top5_indices[0], top5_prob[0])
    ]
}

# Save the result to a JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print("Results saved to result.json")