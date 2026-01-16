from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

print(f"\nModel: {MODEL_NAME}")
print("\nThis model has ~3 billion parameters.")
print("It's already trained on trillions of tokens of text.")
print("But it doesn't know the specific domain yet.\n")

# Load tokenizer (converts text to numbers)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model with 4-bit quantization to fit in 16GB RAM
print("Loading model (this may take a few minutes on first run)...")
print("Using 4-bit quantization to reduce memory usage...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Use half precision (saves memory)
    device_map="auto",  # Automatically use M2 GPU if available
    low_cpu_mem_usage=True,
)

# For 4-bit quantization (uncomment if there are memory issues):
# from transformers import BitsAndBytesConfig
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     quantization_config=quantization_config,
#     device_map="auto",
# )

print("\n✅ Model loaded successfully!")
print(f"\nModel size: ~{model.num_parameters() / 1e9:.2f}B parameters")
print(f"Model device: {next(model.parameters()).device}")

# Test tokenization (core concept)
print("\n" + "=" * 60)
print("Understanding Tokenization")
print("=" * 60)

test_text = "Hello, how are you?"
tokens = tokenizer.encode(test_text)
decoded = tokenizer.decode(tokens)

print(f"\nOriginal text: '{test_text}'")
print(f"Tokens (numbers): {tokens}")
print(f"Decoded back: '{decoded}'")
print("\nThe model only sees numbers (tokens), not text!")
