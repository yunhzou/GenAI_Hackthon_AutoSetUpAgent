import torch
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
from PIL import Image
import json

# Load the image
image = Image.open("cat_with_sunglasses.jpg")

# Load the model and processor
processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224")
model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")

# Process the image
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get the logits and convert to probabilities
logits = outputs.logits
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# Get the top 5 predictions
top5_prob, top5_indices = torch.topk(probabilities, 5)

# Get the class labels
predicted_classes = [
    {
        "label": model.config.id2label[idx.item()],
        "score": prob.item()
    }
    for idx, prob in zip(top5_indices[0], top5_prob[0])
]

# Save the results to a JSON file
with open("result.json", "w") as f:
    json.dump({"predictions": predicted_classes}, f, indent=4)

print("Processing complete. Results saved to result.json")