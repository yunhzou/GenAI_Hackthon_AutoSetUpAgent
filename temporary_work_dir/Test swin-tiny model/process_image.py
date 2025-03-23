import requests
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import json
import io

# Download the image from Unsplash
url = "https://unsplash.com/photos/russian-blue-cat-wearing-yellow-sunglasses-yMSecCHsIBc"
response = requests.get(url)
html_content = response.text

# Extract the actual image URL from the HTML
import re
image_url_match = re.search(r'https://images\.unsplash\.com/[^"]+', html_content)
if image_url_match:
    image_url = image_url_match.group(0)
    print(f"Found image URL: {image_url}")
    
    # Download the actual image
    image_response = requests.get(image_url)
    image = Image.open(io.BytesIO(image_response.content))
    
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
    labels = model.config.id2label
    
    # Create the result dictionary
    result = {
        "predictions": [
            {
                "label": labels[idx.item()],
                "score": val.item()
            }
            for val, idx in zip(values[0], indices[0])
        ]
    }
    
    # Save the result to a JSON file
    with open("result.json", "w") as f:
        json.dump(result, f, indent=4)
    
    print("Results saved to result.json")
else:
    print("Could not find image URL in the Unsplash page")