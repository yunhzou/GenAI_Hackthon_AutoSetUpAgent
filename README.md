# Mistral-1L-Tiny Model Setup

This repository contains the setup for the `nilq/mistral-1L-tiny` model, a tiny single-layer 35.1M parameter Mistral model with a hidden size of 512 and an MLP intermediate size of 1024. The model was trained on the roneneldan/TinyStories dataset.

## Model Information

- **Model Name**: nilq/mistral-1L-tiny
- **Parameters**: 35.1M
- **Architecture**: Single-layer Mistral model
- **Training Dataset**: roneneldan/TinyStories
- **Purpose**: Analysis of feature dynamics and emergence in real-world language models

## Setup Instructions

1. Create a conda environment:
   ```bash
   conda create -n mistral-tiny python=3.10 -y
   conda activate mistral-tiny
   ```

2. Install required packages:
   ```bash
   pip install torch transformers datasets safetensors
   ```

3. Use the model in your Python code:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   # Load the model and tokenizer
   model_name = "nilq/mistral-1L-tiny"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)

   # Test the model with a simple prompt
   prompt = "Once upon a time, there was a little"
   inputs = tokenizer(prompt, return_tensors="pt")

   # Generate text
   outputs = model.generate(
       inputs["input_ids"],
       max_length=100,
       do_sample=True,
       temperature=0.7,
       top_p=0.9,
   )

   # Decode and print the generated text
   generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
   print(f"Prompt: {prompt}")
   print(f"Generated text: {generated_text}")
   ```

## Model Capabilities

This model can generate consistent English text and is particularly useful for:
- Studying feature dynamics in language models
- Experimenting with small-scale language models
- Educational purposes to understand transformer architecture

## Requirements

- Python 3.10 or later
- PyTorch
- Transformers library
- Datasets library
- Safetensors library

## Notes

The model will be downloaded from the Hugging Face Hub the first time you use it. The download size is approximately 141MB.