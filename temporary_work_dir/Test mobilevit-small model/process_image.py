import json
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Load the image
image = Image.open("cat_image.jpg")

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
model = AutoModelForImageClassification.from_pretrained("apple/mobilevit-small")

# Process the image
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class and probabilities
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
predicted_class = model.config.id2label[predicted_class_idx]

# Get the probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)
top_5_probs, top_5_indices = torch.topk(probs, 5)

# Create a dictionary with the top 5 predictions
results = {
    "top_prediction": {
        "class": predicted_class,
        "score": float(probs[0][predicted_class_idx].item())
    },
    "top_5_predictions": [
        {
            "class": model.config.id2label[idx.item()],
            "score": float(prob.item())
        }
        for idx, prob in zip(top_5_indices[0], top_5_probs[0])
    ]
}

# Save the results to a JSON file
with open("result.json", "w") as f:
    json.dump(results, f, indent=4)

print("Processing complete. Results saved to result.json")