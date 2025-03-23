import json
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

# Load the model and processor
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Load the image
image_path = "cat_image.jpg"
image = Image.open(image_path)

# Process the image
inputs = processor(images=image, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted class
predicted_class_idx = logits.argmax(-1).item()
predicted_class = model.config.id2label[predicted_class_idx]

# Get the top 5 predictions
probs = torch.nn.functional.softmax(logits, dim=-1)[0]
top5_prob, top5_indices = torch.topk(probs, 5)

# Create result dictionary
result = {
    "model": model_name,
    "image": image_path,
    "top_prediction": {
        "class": predicted_class,
        "confidence": float(probs[predicted_class_idx].item())
    },
    "top_5_predictions": [
        {
            "class": model.config.id2label[idx.item()],
            "confidence": float(prob.item())
        }
        for idx, prob in zip(top5_indices, top5_prob)
    ]
}

# Save the result to a JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print(f"Top prediction: {predicted_class} with confidence {probs[predicted_class_idx].item():.4f}")
print("Results saved to result.json")