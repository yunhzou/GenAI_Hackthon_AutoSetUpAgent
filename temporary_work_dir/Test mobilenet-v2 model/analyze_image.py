import json
from transformers import MobileNetV2ImageProcessor, MobileNetV2ForImageClassification
from PIL import Image

# Load the image
image_path = "cat_with_sunglasses.jpg"
image = Image.open(image_path)

# Load the model and processor
processor = MobileNetV2ImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
model = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")

# Process the image
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# Get the predicted class
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]

# Get the top 5 predictions
import torch
probs = torch.nn.functional.softmax(logits, dim=-1)
top5_probs, top5_indices = torch.topk(probs, 5)

# Create a result dictionary
result = {
    "top_prediction": {
        "label": predicted_label,
        "score": float(probs[0][predicted_class_idx].item())
    },
    "top_5_predictions": []
}

# Add top 5 predictions to the result
for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
    result["top_5_predictions"].append({
        "rank": i + 1,
        "label": model.config.id2label[idx.item()],
        "score": float(prob.item())
    })

# Save the result to a JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("Analysis complete. Results saved to result.json")