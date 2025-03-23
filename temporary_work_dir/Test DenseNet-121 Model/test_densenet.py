import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import requests
from PIL import Image
import json

# Download the image
image_url = "https://unsplash.com/photos/russian-blue-cat-wearing-yellow-sunglasses-yMSecCHsIBc"
response = requests.get(image_url)
image_content = response.content

# Save the image locally
with open("cat_image.jpg", "wb") as f:
    f.write(image_content)

# Load the image
try:
    image = Image.open("cat_image.jpg")
except:
    # If direct download doesn't work, try to get the actual image URL from the webpage
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')
    for img in img_tags:
        if img.get('alt') and 'cat' in img.get('alt').lower():
            actual_img_url = img.get('src')
            if actual_img_url:
                print(f"Found image URL: {actual_img_url}")
                img_response = requests.get(actual_img_url)
                with open("cat_image.jpg", "wb") as f:
                    f.write(img_response.content)
                image = Image.open("cat_image.jpg")
                break

# Load the model and processor
model_name = "timm/densenet121.tv_in1k"
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

# Get top 5 predictions
probs = torch.nn.functional.softmax(logits, dim=-1)[0]
top5_probs, top5_indices = torch.topk(probs, 5)


top5_results = []
for i, (prob, idx) in enumerate(zip(top5_probs.tolist(), top5_indices.tolist())):
    label = model.config.id2label[idx]
    top5_results.append({
        "rank": i+1,
        "label": label,
        "probability": prob
    })

# Create result dictionary
result = {
    "model": model_name,
    "top_prediction": predicted_class,
    "top_prediction_confidence": probs[predicted_class_idx].item(),
    "top5_predictions": top5_results
}

# Save to JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print("Results saved to result.json")