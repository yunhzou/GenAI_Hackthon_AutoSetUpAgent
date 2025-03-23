import json
from transformers import MobileNetV2ImageProcessor, MobileNetV2ForImageClassification
from PIL import Image

# Load the image
image_path = "cat_with_sunglasses.jpg"
image = Image.open(image_path)

# Load the MobileNet-V2 model and processor
processor = MobileNetV2ImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
model = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")

# Process the image
inputs = processor(images=image, return_tensors="pt")

# Get the model predictions
outputs = model(**inputs)
logits = outputs.logits

# Get the predicted class
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]

# Get the top 5 predictions
import torch
probs = torch.nn.functional.softmax(logits, dim=-1)
top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)

# Convert to list for JSON serialization
top5_results = []
for i, (prob, idx) in enumerate(zip(top5_probs[0].tolist(), top5_indices[0].tolist())):
    label = model.config.id2label[idx]
    top5_results.append({
        "rank": i + 1,
        "label": label,
        "probability": prob
    })

# Create the result dictionary
result = {
    "image_path": image_path,
    "top_prediction": {
        "label": predicted_label,
        "probability": probs[0][predicted_class_idx].item()
    },
    "top5_predictions": top5_results
}

# Save to JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=4)

print(f"Analysis complete. Results saved to result.json")
print(f"Top prediction: {predicted_label} with probability {probs[0][predicted_class_idx].item():.4f}")