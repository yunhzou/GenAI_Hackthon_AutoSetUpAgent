import json
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image

# Load the model and feature extractor
model_name = "microsoft/resnet-18"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Load the image
image_path = "cat_image.jpg"
image = Image.open(image_path)

# Process the image
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# Get the predicted class
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]

# Get the top 5 predictions
import torch
import torch.nn.functional as F
probs = F.softmax(logits, dim=-1)
top5_prob, top5_indices = torch.topk(probs, 5)

# Create result dictionary
result = {
    "top_prediction": {
        "label": predicted_label,
        "score": float(probs[0][predicted_class_idx].item())
    },
    "top_5_predictions": []
}

# Add top 5 predictions to result
for i, (prob, idx) in enumerate(zip(top5_prob[0], top5_indices[0])):
    label = model.config.id2label[idx.item()]
    score = float(prob.item())
    result["top_5_predictions"].append({
        "rank": i+1,
        "label": label,
        "score": score
    })

# Save the result to a JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print("Classification complete. Results saved to result.json")