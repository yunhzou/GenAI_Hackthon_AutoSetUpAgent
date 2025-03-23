import json
import torch
import timm
from PIL import Image
import requests
from io import BytesIO
import urllib.request

# URL of the image
image_url = "https://images.unsplash.com/photo-1533738363-b7f9aef128ce"

# Download the image
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))

# Load the model
print("Loading DenseNet-121 model...")
model = timm.create_model('densenet121.tv_in1k', pretrained=True)
model = model.eval()

# Get model specific transforms
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# Transform and process the image
print("Processing image...")
input_tensor = transforms(img).unsqueeze(0)  # Add batch dimension

# Get predictions
with torch.no_grad():
    output = model(input_tensor)

# Get top 5 predictions
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_indices = torch.topk(probabilities, 5)

# Load ImageNet class labels
with urllib.request.urlopen("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt") as f:
    categories = [line.decode("utf-8").strip() for line in f.readlines()]

# Create results dictionary
results = {
    "model": "densenet121.tv_in1k",
    "image_url": image_url,
    "predictions": []
}

for i in range(5):
    results["predictions"].append({
        "class": categories[top5_indices[i].item()],
        "probability": float(top5_prob[i].item())
    })

# Save results to JSON file
with open("result.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to result.json")