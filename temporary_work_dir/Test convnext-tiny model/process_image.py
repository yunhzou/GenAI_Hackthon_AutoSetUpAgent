import torch
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
from PIL import Image
import json

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

# Convert to list for JSON serialization
predictions = []
for i, (prob, idx) in enumerate(zip(top5_prob[0], top5_indices[0])):
    label = model.config.id2label[idx.item()]
    score = prob.item()
    predictions.append({
        "rank": i + 1,
        "label": label,
        "score": score
    })

# Create the result dictionary
result = {
    "model": model_name,
    "image": image_path,
    "predictions": predictions
}

# Save to JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("Processing complete. Results saved to result.json")