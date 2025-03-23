import requests
import json
from io import BytesIO
from PIL import Image
import torch
from transformers import ConvNextImageProcessor, ConvNextForImageClassification

# Download the image
image_url = "https://images.unsplash.com/photo-1533738363-b7f9aef128ce"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Load the model and processor
processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224")
model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")

# Process the image
inputs = processor(image, return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted class
predicted_class_idx = logits.argmax(-1).item()
predicted_class_label = model.config.id2label[predicted_class_idx]
confidence = torch.softmax(logits, dim=1)[0, predicted_class_idx].item()

# Create result dictionary
result = {
    "model": "facebook/convnext-tiny-224",
    "image": "Russian blue cat wearing yellow sunglasses",
    "prediction": {
        "class_id": predicted_class_idx,
        "class_name": predicted_class_label,
        "confidence": confidence
    },
    "top_5_predictions": []
}

# Get top 5 predictions
topk_values, topk_indices = torch.topk(torch.softmax(logits, dim=1)[0], 5)
for i, (score, idx) in enumerate(zip(topk_values.tolist(), topk_indices.tolist())):
    result["top_5_predictions"].append({
        "rank": i + 1,
        "class_id": idx,
        "class_name": model.config.id2label[idx],
        "confidence": score
    })

# Save to JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print("Results saved to result.json")