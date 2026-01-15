from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Using Phi-3 Mini - small, efficient, no auth required
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

print(f"\nUsing: {MODEL_NAME}")

try:
    print("Loading model (first time may take a few minutes to download)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Use memory-efficient loading with float16 (works on CPU too, saves 50% memory)
    print("Using memory-efficient loading (float16, low_cpu_mem_usage)...")
    
    # Try without trust_remote_code first (preferred)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,  # Use float16 even on CPU - saves memory
            device_map="auto",
            low_cpu_mem_usage=True,  # Critical for memory efficiency
            attn_implementation="eager",  # Use standard attention (works everywhere, no warnings)
            trust_remote_code=False,
        )
    except Exception as e:
        # Fallback to trust_remote_code if needed
        print(f"Note: Loading with trust_remote_code=True due to: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,  # Use float16 even on CPU - saves memory
            device_map="auto",
            low_cpu_mem_usage=True,  # Critical for memory efficiency
            attn_implementation="eager",  # Use standard attention (works everywhere, no warnings)
            trust_remote_code=True,
        )

    print("✅ Model loaded successfully!\n")

    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set model to eval mode
    model.eval()

    # Simple test - Phi-3 is instruction-tuned, use chat format
    prompt = "Explain what a language model is in one sentence."
    print(f"Prompt: {prompt}\n")

    # Use chat template for Phi-3 (instruction-tuned model)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print("Generating response...")
    print("(Using greedy decoding for speed - this should complete in 10-30 seconds)\n")
    
    # Try with cache first, fallback to no cache if there's an error
    try:
        print("Attempting generation with cache enabled...")
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=50,
                do_sample=False,  # Greedy decoding - faster and more reliable
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        print("✅ Generation completed with cache!")
    except (AttributeError, RuntimeError) as e:
        if "seen_tokens" in str(e) or "DynamicCache" in str(e):
            print(f"Cache error detected ({e}), retrying without cache...")
            print("(This will be slower but should work)")
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,  # Fallback: slower but works
                )
            print("✅ Generation completed without cache!")
        else:
            raise

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response (remove the prompt)
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    print(f"Response: {response}\n")

    print("=" * 60)
    print("✅ Success! environment is set up correctly.")
    print("=" * 60)
    print("\nNext steps:")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you installed dependencies: pip install -r requirements.txt")
    print("2. Check your internet connection (model needs to download)")
    print("3. See SETUP.md for more help")
