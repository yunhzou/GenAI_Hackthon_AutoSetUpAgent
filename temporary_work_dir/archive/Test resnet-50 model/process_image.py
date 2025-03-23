import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import json

# Load the image
image = Image.open("cat_image.jpg")

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

# Process the image
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
predicted_class = model.config.id2label[predicted_class_idx]

# Get the probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)
top_5_probs, top_5_indices = torch.topk(probs, 5)

# Create a result dictionary
result = {
    "top_prediction": {
        "class": predicted_class,
        "confidence": float(probs[0][predicted_class_idx].item())
    },
    "top_5_predictions": [
        {
            "class": model.config.id2label[idx.item()],
            "confidence": float(prob.item())
        }
        for idx, prob in zip(top_5_indices[0], top_5_probs[0])
    ]
}

# Save to JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print("Processing complete. Results saved to result.json")