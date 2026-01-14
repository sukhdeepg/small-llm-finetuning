from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Using Phi-3 Mini - small, efficient, no auth required
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

print(f"\nUsing: {MODEL_NAME}")

try:
    print("Loading model (first time may take a few minutes to download)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,  # Required for Phi-3
    )

    print("✅ Model loaded successfully!\n")

    # Simple test
    prompt = "Explain what a language model is in one sentence."
    print(f"Prompt: {prompt}\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
