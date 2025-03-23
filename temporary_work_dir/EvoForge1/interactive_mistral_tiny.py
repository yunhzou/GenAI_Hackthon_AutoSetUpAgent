from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "nilq/mistral-1L-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set pad_token_id if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\nMistral-1L-tiny is ready for text generation!")
    print("Enter a prompt to generate text (type 'exit' to quit):")
    
    while True:
        # Get user input
        user_input = input("\nPrompt: ")
        
        if user_input.lower() == 'exit':
            break
        
        # Tokenize the prompt
        inputs = tokenizer(user_input, return_tensors="pt")
        
        # Generate text
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            temperature=0.7,
            do_sample=True,
        )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Print the result
        print("\nGenerated text:")
        print(generated_text)

if __name__ == "__main__":
    main()