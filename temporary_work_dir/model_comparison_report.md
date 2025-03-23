# Computer Vision Model Comparison Report

This report compares the performance of various computer vision models on an image classification task. The models were tested on an image of a cat wearing sunglasses.

## Model Performance Comparison

| Model | Top Prediction | Confidence | 2nd Prediction | Confidence | 3rd Prediction | Confidence |
|-------|---------------|------------|---------------|------------|---------------|------------|
| ConvNeXT-Tiny | sunglasses, dark glasses, shades | 47.32% | sunglass | 29.90% | Egyptian cat | 5.95% |
| DeiT-Tiny | Egyptian cat | 14.96% | N/A | N/A | N/A | N/A |
| DenseNet-121 | sunglass | 58.23% | sunglasses, dark glasses, shades | 37.34% | tabby, tabby cat | 1.25% |
| EfficientNet-B0 | sunglass | 9.91% | Persian cat | N/A | sunglasses, dark glasses, shades | N/A |
| MobileNet-V2 | Egyptian cat | 44.27% | tiger cat | 27.49% | tabby, tabby cat | 7.71% |
| MobileViT-Small | sunglasses, dark glasses, shades | 63.35% | sunglass | 20.32% | Egyptian cat | 3.85% |
| ResNet-18 | sunglasses, dark glasses, shades | 63.84% | sunglass | 32.48% | Egyptian cat | 0.41% |
| ResNet-50 | tabby, tabby cat | N/A | sunglasses, dark glasses, shades | N/A | Egyptian cat | N/A |
| Swin-Tiny | sunglasses, dark glasses, shades | 47.48% | sunglass | 19.50% | Egyptian cat | 9.77% |
| ViT-Base | Persian cat | 63.61% | N/A | N/A | N/A | N/A |

## Analysis

1. **Best Performing Models for Sunglasses Detection**:
   - ResNet-18 had the highest confidence (63.84%) for correctly identifying "sunglasses, dark glasses, shades"
   - MobileViT-Small also performed very well with 63.35% confidence for sunglasses
   - DenseNet-121 showed strong performance with 58.23% confidence for "sunglass"

2. **Best Performing Models for Cat Detection**:
   - ViT-Base had the highest confidence (63.61%) for identifying "Persian cat"
   - MobileNet-V2 had good confidence (44.27%) for "Egyptian cat"
   - DeiT-Tiny identified "Egyptian cat" with 14.96% confidence

3. **Consistency**:
   - Most models correctly identified either "sunglasses" or "sunglass" as a primary object
   - Cat-related classes ("Egyptian cat", "tabby cat", "tiger cat", "Persian cat") consistently appeared in the predictions
   - The models showed varying abilities to detect both the cat and the sunglasses in the same image

4. **Observations**:
   - Traditional CNN architectures like ResNet-18 performed very well for sunglasses detection
   - Transformer-based models showed mixed results, with MobileViT-Small performing excellently for sunglasses detection while ViT-Base focused more on the cat
   - Different model architectures appear to prioritize different aspects of the image (sunglasses vs. cat)
   - The ResNet-50 results appear to use a different scoring system (negative values) and identified "tabby, tabby cat" as the top prediction

This comparison demonstrates the varying capabilities of different computer vision model architectures when classifying objects in images with multiple prominent features. Some models excel at identifying accessories (sunglasses), while others focus more on the main subject (cat).