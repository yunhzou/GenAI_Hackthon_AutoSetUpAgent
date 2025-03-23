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