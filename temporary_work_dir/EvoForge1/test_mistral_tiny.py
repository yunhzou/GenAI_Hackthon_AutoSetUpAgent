from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "nilq/mistral-1L-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define a prompt
prompt = "Once upon a time, there was a little girl who"

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(
    inputs["input_ids"],
    max_length=100,
    temperature=0.7,
    do_sample=True,
)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the result
print("Generated text:")
print(generated_text)