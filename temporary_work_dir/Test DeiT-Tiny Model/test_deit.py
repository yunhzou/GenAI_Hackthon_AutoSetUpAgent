import json
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image

# Load the image
image_path = "cat_with_sunglasses.jpg"
image = Image.open(image_path)

# Load the model and feature extractor
model_name = "facebook/deit-tiny-distilled-patch16-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Process the image
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# Get the predicted class
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]
score = logits.softmax(dim=1)[0, predicted_class_idx].item()

# Create result dictionary
result = {
    "model": model_name,
    "image": image_path,
    "prediction": {
        "class_id": predicted_class_idx,
        "class_name": predicted_label,
        "confidence": score
    }
}

# Save to JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print(f"Prediction saved to result.json")
print(f"Predicted class: {predicted_label}")
print(f"Confidence: {score:.4f}")