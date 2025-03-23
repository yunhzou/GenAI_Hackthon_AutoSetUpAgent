import json
import requests
from PIL import Image
from io import BytesIO
from transformers import MobileNetV2ImageProcessor, MobileNetV2ForImageClassification

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
    img_response = requests.get(actual_image_url)
    image = Image.open(BytesIO(img_response.content))
else:
    raise Exception("Could not find image URL in the webpage")

# Load MobileNetV2 model and processor
processor = MobileNetV2ImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
model = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")

# Process the image
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# Get the predicted class
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]

# Create result dictionary
result = {
    "predicted_class_index": predicted_class_idx,
    "predicted_label": predicted_label,
    "confidence_score": float(logits.softmax(dim=-1)[0, predicted_class_idx].item())
}

# Save to JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print(f"Results saved to result.json")
print(f"Prediction: {predicted_label} with confidence {result['confidence_score']:.4f}")