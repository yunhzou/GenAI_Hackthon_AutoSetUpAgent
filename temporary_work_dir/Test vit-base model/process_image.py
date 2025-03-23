import requests
import json
from PIL import Image
from io import BytesIO
from transformers import ViTImageProcessor, ViTForImageClassification

# Download the image from Unsplash
image_url = "https://unsplash.com/photos/russian-blue-cat-wearing-yellow-sunglasses-yMSecCHsIBc"
response = requests.get(image_url)
webpage_content = response.text

# Extract the actual image URL from the webpage
import re
image_pattern = r'https://images\.unsplash\.com/photo-[^"\']*'
image_matches = re.findall(image_pattern, webpage_content)
if image_matches:
    actual_image_url = image_matches[0]
    print(f"Found image URL: {actual_image_url}")
    
    # Download the actual image
    image_response = requests.get(actual_image_url)
    image = Image.open(BytesIO(image_response.content))
else:
    print("Could not find image URL in the webpage")
    exit(1)

# Load the ViT model and processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# Process the image
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# Get the predicted class
predicted_class_idx = logits.argmax(-1).item()
predicted_class = model.config.id2label[predicted_class_idx]

# Get the top 5 predictions
import torch
probs = torch.nn.functional.softmax(logits, dim=-1)
top5_prob, top5_indices = torch.topk(probs, 5, dim=-1)

# Create a result dictionary
result = {
    "top_prediction": {
        "class": predicted_class,
        "score": float(probs[0][predicted_class_idx].item())
    },
    "top_5_predictions": []
}

for i, (prob, idx) in enumerate(zip(top5_prob[0], top5_indices[0])):
    result["top_5_predictions"].append({
        "rank": i + 1,
        "class": model.config.id2label[idx.item()],
        "score": float(prob.item())
    })

# Save the result to a JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print("Results saved to result.json")