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
    # We got the webpage, need to extract the actual image URL
    print("Got webpage instead of direct image, extracting image URL...")
    # Try to get the actual image URL from the page
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(image_content, 'html.parser')
    # Look for the image in the meta tags or other elements
    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        image_url = og_image.get("content")
        print(f"Found image URL: {image_url}")
        response = requests.get(image_url)
        image_content = response.content
    else:
        print("Could not find image URL in the webpage")
        exit(1)

# Load the image
image = Image.open(BytesIO(image_content))

# Load the model and processor
model_name = "microsoft/resnet-50"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Process the image
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
predicted_class = model.config.id2label[predicted_class_idx]

# Get the top 5 predictions
top5_indices = logits[0].topk(5).indices.tolist()
top5_scores = logits[0].topk(5).values.tolist()
top5_predictions = [
    {"label": model.config.id2label[idx], "score": float(score)}
    for idx, score in zip(top5_indices, top5_scores)
]

# Create the result dictionary
result = {
    "model": model_name,
    "image_url": image_url,
    "predicted_class": predicted_class,
    "top5_predictions": top5_predictions
}

# Save to result.json
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print(f"Prediction complete. Top prediction: {predicted_class}")
print("Results saved to result.json")