import json
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoImageProcessor, AutoModelForImageClassification

def main():
    # Download the image from Unsplash
    # Direct image URL from the Unsplash page
    image_url = "https://images.unsplash.com/photo-1533738363-b7f9aef128ce"
    response = requests.get(image_url, stream=True)
    image = Image.open(BytesIO(response.content))
    
    print("Image downloaded successfully")
    
    # Load the model and processor
    model_name = "microsoft/swin-tiny-patch4-window7-224"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    
    print("Model loaded successfully")
    
    # Process the image
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Get the predicted class
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    
    # Get top 5 predictions
    import torch
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top5_prob, top5_indices = torch.topk(probs, 5)
    
    # Convert to list for JSON serialization
    top5_prob = top5_prob.squeeze().tolist()
    top5_indices = top5_indices.squeeze().tolist()
    
    # Create results dictionary
    results = {
        "top_prediction": {
            "class": predicted_class,
            "class_id": predicted_class_idx,
            "confidence": float(probs[0][predicted_class_idx])
        },
        "top5_predictions": [
            {
                "class": model.config.id2label[idx],
                "class_id": idx,
                "confidence": float(prob)
            }
            for idx, prob in zip(top5_indices, top5_prob)
        ]
    }
    
    # Save results to JSON file
    with open("result.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to result.json")
    print(f"Top prediction: {predicted_class} with confidence {results['top_prediction']['confidence']:.4f}")

if __name__ == "__main__":
    main()