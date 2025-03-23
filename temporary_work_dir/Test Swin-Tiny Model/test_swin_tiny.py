import json
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image

# Load the image
image_path = "cat_with_sunglasses.jpg"
image = Image.open(image_path)

# Load the model and feature extractor
model_name = "microsoft/swin-tiny-patch4-window7-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Process the image
inputs = feature_extractor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
predicted_class = model.config.id2label[predicted_class_idx]

# Get the confidence scores
probs = torch.nn.functional.softmax(logits, dim=-1)
confidence = probs[0][predicted_class_idx].item()

# Get top 5 predictions
topk_values, topk_indices = torch.topk(probs, 5)
top_predictions = []
for i, (score, idx) in enumerate(zip(topk_values[0].tolist(), topk_indices[0].tolist())):
    label = model.config.id2label[idx]
    top_predictions.append({
        "rank": i + 1,
        "label": label,
        "score": score
    })

# Create result dictionary
result = {
    "image": image_path,
    "model": model_name,
    "top_prediction": {
        "label": predicted_class,
        "confidence": confidence
    },
    "top_5_predictions": top_predictions
}

# Save to JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print(f"Prediction complete. Results saved to result.json")
print(f"Top prediction: {predicted_class} with confidence {confidence:.4f}")