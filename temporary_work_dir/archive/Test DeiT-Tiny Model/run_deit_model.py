import json
from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher
from PIL import Image

# Load the image
image_path = "cat_image.jpg"
image = Image.open(image_path)

# Load the feature extractor and model
model_name = "facebook/deit-tiny-distilled-patch16-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = DeiTForImageClassificationWithTeacher.from_pretrained(model_name)

# Process the image
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# Get the predicted class
predicted_class_idx = logits.argmax(-1).item()
predicted_class = model.config.id2label[predicted_class_idx]

# Get the top 5 predictions
probs = outputs.logits.softmax(dim=-1)[0]
top5_prob, top5_indices = probs.topk(5)

# Create a result dictionary
result = {
    "model": model_name,
    "image": image_path,
    "top_prediction": {
        "class_id": predicted_class_idx,
        "class_name": predicted_class,
        "confidence": float(probs[predicted_class_idx].item())
    },
    "top_5_predictions": [
        {
            "class_id": idx.item(),
            "class_name": model.config.id2label[idx.item()],
            "confidence": float(prob.item())
        }
        for idx, prob in zip(top5_indices, top5_prob)
    ]
}

# Save the result to a JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print(f"Results saved to result.json")
print(f"Top prediction: {predicted_class} with confidence {probs[predicted_class_idx].item():.4f}")