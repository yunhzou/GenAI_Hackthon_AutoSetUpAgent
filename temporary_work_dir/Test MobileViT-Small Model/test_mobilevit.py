import json
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Load the image
image_path = "cat_with_sunglasses.jpg"
image = Image.open(image_path)

# Load the MobileViT-Small model and processor
processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
model = AutoModelForImageClassification.from_pretrained("apple/mobilevit-small")

# Process the image
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# Get the predicted class and probabilities
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]

# Get top 5 predictions
import torch
import torch.nn.functional as F
probs = F.softmax(logits, dim=-1)
top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)

# Convert to list
top5_probs = top5_probs.squeeze().tolist()
top5_indices = top5_indices.squeeze().tolist()

# Create result dictionary
results = {
    "top_prediction": {
        "label": predicted_label,
        "score": float(probs[0, predicted_class_idx].item())
    },
    "top5_predictions": [
        {
            "label": model.config.id2label[idx],
            "score": float(prob)
        }
        for idx, prob in zip(top5_indices, top5_probs)
    ]
}

# Save to JSON file
with open("result.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to result.json")
print(f"Top prediction: {predicted_label} with confidence {results['top_prediction']['score']:.4f}")