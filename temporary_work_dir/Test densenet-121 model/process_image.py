import torch
from PIL import Image
import requests
from io import BytesIO
import json
from torchvision import transforms
from timm import create_model

# Direct image URL from Unsplash
image_url = "https://images.unsplash.com/photo-1533738363-b7f9aef128ce?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80"
print(f"Downloading image from: {image_url}")

# Download the image
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
print("Image downloaded successfully")

# Load the model
print("Loading model...")
model = create_model('densenet121', pretrained=True)
model.eval()

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)

# Run inference
print("Running inference...")
with torch.no_grad():
    output = model(input_batch)

# Get the predicted class
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

# Convert to list for JSON serialization
result = {
    "top5_categories": top5_catid.tolist(),
    "top5_probabilities": top5_prob.tolist()
}

# Save the result to a JSON file
with open('result.json', 'w') as f:
    json.dump(result, f, indent=4)

print("Results saved to result.json")