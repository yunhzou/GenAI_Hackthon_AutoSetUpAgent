import json
import torch
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
from PIL import Image

# Load the image
image_path = "cat_with_sunglasses.jpg"
image = Image.open(image_path)

# Load the model and processor
model_name = "facebook/convnext-tiny-224"
processor = ConvNextImageProcessor.from_pretrained(model_name)
model = ConvNextForImageClassification.from_pretrained(model_name)

# Process the image
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class and probabilities
logits = outputs.logits
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# Get the top 5 predictions
top5_prob, top5_indices = torch.topk(probabilities, 5)

# Convert to Python lists
top5_prob = top5_prob.squeeze().tolist()
top5_indices = top5_indices.squeeze().tolist()

# Get the class labels
predicted_labels = [model.config.id2label[idx] for idx in top5_indices]

# Create a dictionary with the results
results = {
    "model": model_name,
    "image": image_path,
    "predictions": [
        {"label": label, "probability": prob} 
        for label, prob in zip(predicted_labels, top5_prob)
    ]
}

# Save to JSON file
with open("result.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to result.json")