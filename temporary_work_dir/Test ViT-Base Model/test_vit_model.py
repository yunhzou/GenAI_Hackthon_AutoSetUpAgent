import requests
import json
from PIL import Image
from io import BytesIO
from transformers import ViTImageProcessor, ViTForImageClassification

# Download the image
image_url = "https://unsplash.com/photos/russian-blue-cat-wearing-yellow-sunglasses-yMSecCHsIBc"
response = requests.get(image_url)
print(f"Image URL response status: {response.status_code}")

# If we get a redirect or HTML page instead of an image, we need to extract the actual image URL
if "image" not in response.headers.get("Content-Type", ""):
    print("Direct URL didn't return an image. Trying to find image in HTML...")
    # For Unsplash, we need to get the actual image URL from the page
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    # Look for the image in the meta tags or other elements
    meta_img = soup.find('meta', property='og:image')
    if meta_img and meta_img.get('content'):
        image_url = meta_img.get('content')
        print(f"Found image URL: {image_url}")
        response = requests.get(image_url)
    else:
        # Try to find image in other ways
        img_tags = soup.find_all('img')
        for img in img_tags:
            if img.get('src') and ('cat' in img.get('src').lower() or 'sunglasses' in img.get('src').lower()):
                image_url = img.get('src')
                if not image_url.startswith('http'):
                    image_url = 'https:' + image_url if image_url.startswith('//') else 'https://unsplash.com' + image_url
                print(f"Found image URL: {image_url}")
                response = requests.get(image_url)
                break

# Load the image
try:
    image = Image.open(BytesIO(response.content))
    print(f"Successfully loaded image of size {image.size}")
except Exception as e:
    print(f"Error loading image: {e}")
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

# Create result dictionary
result = {
    "model": "google/vit-base-patch16-224",
    "image_url": image_url,
    "predicted_class_idx": predicted_class_idx,
    "predicted_class": predicted_class,
    "confidence": float(logits.softmax(dim=-1)[0, predicted_class_idx].item())
}

# Save to result.json
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print(f"Results saved to result.json")
print(f"Prediction: {predicted_class} (Confidence: {result['confidence']:.4f})")