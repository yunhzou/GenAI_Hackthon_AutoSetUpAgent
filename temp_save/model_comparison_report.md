# Computer Vision Model Comparison Report

This report compares the performance of various computer vision models on an image classification task. The models were tested on an image of a cat wearing sunglasses.

## Model Performance Comparison

| Model | Top Prediction | Confidence | 2nd Prediction | Confidence | 3rd Prediction | Confidence |
|-------|---------------|------------|---------------|------------|---------------|------------|
| ConvNeXT-Tiny | sunglasses, dark glasses, shades | 47.32% | sunglass | 29.90% | Egyptian cat | 5.95% |
| DeiT-Tiny | wig | 0.74% | French bulldog | 0.59% | Gila monster | 0.49% |
| DenseNet-121 | sunglass | 40.96% | sunglasses | 27.54% | tabby | 3.12% |
| EfficientNet-B0 | sunglass | 8.64% | Persian cat | 8.15% | sunglasses, dark glasses, shades | 6.96% |
| MobileNet-V2 | Egyptian cat | 44.27% | N/A | N/A | N/A | N/A |
| MobileViT-Small | sunglasses, dark glasses, shades | 63.35% | sunglass | 20.32% | Egyptian cat | 3.85% |
| ResNet-18 | sunglasses, dark glasses, shades | 63.84% | sunglass | 32.48% | Egyptian cat | 0.41% |
| ResNet-50 | sunglasses, dark glasses, shades | 73.31% | sunglass | 23.04% | Egyptian cat | 0.82% |
| Swin-Tiny | sunglasses, dark glasses, shades | 49.65% | sunglass | 20.28% | Egyptian cat | 8.34% |
| ViT-Base | sunglasses, dark glasses, shades | 65.38% | sunglass | 17.90% | tiger cat | 5.07% |

## Analysis

1. **Best Performing Models**:
   - ResNet-50 had the highest confidence (73.31%) for the correct prediction of "sunglasses"
   - ViT-Base and ResNet-18 also performed very well with confidence scores above 63%
   - MobileViT-Small showed strong performance with 63.35% confidence

2. **Poorest Performing Models**:
   - DeiT-Tiny performed extremely poorly with only 0.74% confidence for an incorrect prediction ("wig")
   - EfficientNet-B0 had relatively low confidence (8.64%) for its top prediction
   - MobileNet-V2 identified the cat but missed the sunglasses entirely

3. **Consistency**:
   - Most models correctly identified either "sunglasses" or "sunglass" as the primary object
   - Cat-related classes ("Egyptian cat", "tabby cat", "tiger cat") consistently appeared in the top predictions
   - The models that performed well showed a clear distinction between the confidence of the top prediction and secondary predictions

4. **Observations**:
   - Traditional CNN architectures like ResNet-50 outperformed some newer transformer-based models
   - The Vision Transformer (ViT-Base) performed very well, showing the potential of transformer architectures for vision tasks
   - Smaller/lightweight models generally had lower confidence scores, with DeiT-Tiny being particularly poor

This comparison demonstrates the varying capabilities of different computer vision model architectures when classifying objects in images with multiple prominent features.