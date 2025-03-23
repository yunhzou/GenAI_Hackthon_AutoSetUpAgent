import json
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

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
    
    # Download the actual image
    image_response = requests.get(actual_image_url)
    image = Image.open(BytesIO(image_response.content))
    
    # Load the model and feature extractor
    model_name = "microsoft/resnet-50"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    
    # Process the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    # Get the predicted class and probabilities
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)
    
    # Get the top 5 predictions
    values, indices = probabilities[0].topk(5)
    
    # Convert to Python types for JSON serialization
    predictions = [
        {
            "label": model.config.id2label[idx.item()],
            "score": val.item()
        }
        for val, idx in zip(values, indices)
    ]
    
    # Save the results to a JSON file
    with open("result.json", "w") as f:
        json.dump({"predictions": predictions}, f, indent=4)
    
    print("Results saved to result.json")
else:
    print("Could not find image URL in the webpage")