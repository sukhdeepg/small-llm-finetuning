This document explains the fundamental concepts behind small language models.

## 1. What is a Language Model?

Think of a language model as a **very advanced autocomplete**. 

- We give it some text: "The weather today is"
- It predicts what comes next: "sunny and warm"

But instead of just the next word, it understands:
- Context (what we're talking about)
- Grammar and syntax
- Meaning and relationships
- Style and tone

## 2. How Models Work (The Big Picture)

### Step 1: Pre-training (Already Done)
The model reads **trillions** of words from books, websites, code, etc. It learns:
- Language patterns
- Facts about the world
- How to structure sentences
- Relationships between concepts

**We don't do this** - companies like Meta, Mistral, etc. do this with massive compute.

### Step 2: Fine-tuning (What We Do)
We take the pre-trained model and show it **our specific examples**:
- Our company's terminology
- Our domain (medical, legal, etc.)
- Our style and format

The model adjusts slightly to be better at OUR task.

### Step 3: Inference (Using the Model)
We ask questions, and the model generates answers based on everything it learned.

## 3. Tokenization: Text → Numbers

Models don't understand text directly. They work with **numbers**.

```
Text: "Hello world"
     ↓ (tokenizer)
Tokens: [9906, 1917]  (numbers)
     ↓ (model processes)
     ↓ (decoder)
Text: "Hello world, how are you?"
```

**Key insight**: One word ≠ one token. "Hello" might be 1 token, but "unhappiness" might be 3 tokens: "un", "happy", "ness".

## 4. Embeddings: Meaning as Vectors

Each token becomes a **vector** (list of numbers) that captures meaning:

```
"king" → [0.2, -0.1, 0.8, ..., 0.3]  (768 numbers for example)
"queen" → [0.2, -0.1, 0.7, ..., 0.3]  (similar numbers!)
"car" → [-0.5, 0.3, -0.2, ..., 0.1]  (different numbers)
```

Similar words have similar vectors. The model learns these relationships.

## 5. Attention: What Matters?

When we say "The cat sat on the mat", the model learns:
- "sat" relates to "cat" (what sat?)
- "on" relates to "mat" (where?)
- "The" relates to "cat" (which cat?)

**Attention** is the mechanism that figures out which words are important for understanding each other.

### Flash Attention: Optimized Attention Computation

**What it is**: An optimized version of attention that's faster and uses less memory.

**The Problem with Standard Attention**:
- For a sequence of 8,192 tokens, standard attention creates an 8,192 × 8,192 matrix
- That's ~67 million numbers stored in memory!
- Memory usage grows **quadratically** with sequence length

**How Flash Attention Solves It**:
1. **Block-wise computation**: Instead of processing everything at once, it breaks the sequence into smaller blocks (e.g., 512 tokens each)
2. **Fused operations**: Combines multiple steps to avoid storing intermediate results
3. **Uses fast cache memory**: Keeps frequently used data in fast on-chip memory (SRAM) instead of slow main memory

**Example**:
```
Standard Attention:
Sequence: [token1, token2, ..., token8192]
→ Creates 8192×8192 matrix (67M numbers) in slow memory
→ Slow and memory-intensive

Flash Attention:
Sequence: [token1, token2, ..., token8192]
→ Breaks into blocks: [block1(512), block2(512), ...]
→ Processes each block using fast cache memory
→ Much faster, uses less memory!
```

## 6. Parameters: The Model's "Memory"

A model with **3 billion parameters** means:
- 3 billion numbers (weights) that store what it learned
- Each parameter adjusts during training
- More parameters = more capacity to learn (but also more memory needed)

**Small models** (1B-7B parameters) are a sweet spot:
- Powerful enough for most tasks
- Small enough to run on consumer hardware
- Fast enough for real-time use

## 7. Fine-tuning vs. Full Training

### Full Training (Pre-training)
- Starts from random weights
- Needs trillions of examples
- Takes weeks/months on supercomputers
- Cost: Millions of dollars

### Fine-tuning
- Starts from pre-trained weights
- Needs hundreds/thousands of examples
- Takes minutes/hours on our laptop
- Cost: Free (or cloud credits)

**Analogy**: 
- Full training = learning a language from scratch
- Fine-tuning = learning a dialect or specialized vocabulary

## 8. LoRA: Efficient Fine-tuning

**Problem**: Fine-tuning all 3 billion parameters uses lots of memory.

**Solution**: LoRA (Low-Rank Adaptation)
- Only train a small subset of parameters
- Add "adapter" layers that are much smaller
- Works almost as well as full fine-tuning
- Uses 10-100x less memory!

**Example**:
- Full fine-tuning: Train 3B parameters
- LoRA: Train only 10M parameters (0.3% of the model!)

## 9. Generation: How Text is Created

1. **Input**: "What is AI?"
2. **Tokenize**: Convert to numbers
3. **Process**: Model predicts probability of each possible next token
4. **Sample**: Pick a token based on probabilities (with some randomness)
5. **Repeat**: Use the generated token as input, predict next one
6. **Decode**: Convert numbers back to text

**Temperature** controls randomness:
- Low (0.1): Very deterministic, same input → same output
- High (1.0): Very creative, more varied outputs

## 10. Code Concepts: Understanding the Technical Terms

### `torch_dtype`: Precision vs Memory Tradeoff

**What it is**: Controls how many decimal places numbers use.

**Options**:
- `torch.float32`: 32-bit precision (more accurate, uses 2x memory)
- `torch.float16`: 16-bit precision (less accurate, uses half memory)

**Example**:
```python
# float32: 3.141592653589793 (very precise)
# float16: 3.1416 (good enough, saves memory!)
```

**Why it matters**: Models are huge! Using `float16` can cut memory usage in half with minimal quality loss.

**Code**: `torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32`
- If GPU available → use float16 (GPU handles it well)
- If CPU only → use float32 (more stable on CPU)

### `device_map`: Where to Run the Model

**What it is**: Controls which device (CPU, GPU) the model runs on.

**Options**:

1. **`"auto"`** (recommended): Automatically detects and uses the best available device
   - Checks for CUDA GPU → uses it if available
   - Checks for Apple Silicon GPU (MPS) → uses it on M1/M2/M3 Macs
   - Falls back to CPU if no GPU available
   - **Best choice**: Works everywhere, optimizes automatically

2. **`"cpu"`**: Force CPU usage
   - Always uses our computer's processor
   - Slower but guaranteed to work
   - Use when: We want to avoid GPU, or debugging

3. **`"cuda"`**: Use NVIDIA GPU (if available)
   - Much faster than CPU
   - Requires NVIDIA GPU with CUDA installed
   - Use when: We have NVIDIA GPU and want maximum speed

4. **`"cuda:0"`, `"cuda:1"`**: Specific GPU device
   - `"cuda:0"` = First GPU
   - `"cuda:1"` = Second GPU
   - Use when: We have multiple GPUs and want to choose

5. **`"mps"`**: Apple Silicon GPU (M1/M2/M3 Macs)
   - Uses Metal Performance Shaders
   - Faster than CPU on Apple Silicon
   - Use when: On Mac with M1/M2/M3 chip

6. **Dictionary**: Split model across multiple devices
   ```python
   device_map={"layer1": "cuda:0", "layer2": "cuda:1"}
   ```
   - Use when: Model is too large for one GPU

**Examples**:
```python
# Auto (recommended - smart choice)
device_map="auto"

# Force CPU
device_map="cpu"

# Force GPU (will error if no GPU)
device_map="cuda"

# Specific GPU
device_map="cuda:0"

# Apple Silicon
device_map="mps"
```

### `low_cpu_mem_usage=True`: Memory-Efficient Loading

**What it is**: Loads the model in a memory-efficient way, especially important for large models.

**Options**:
- `True`: Use memory-efficient loading (recommended)
- `False`: Standard loading (uses more memory)

**What it does**:
- Loads model weights **incrementally** instead of all at once
- Reduces peak memory usage during loading
- Prevents "out of memory" errors on systems with limited RAM

**Why it matters**:
- Models can be 4-14GB in size
- Loading everything at once might exceed available RAM
- Incremental loading uses less peak memory

**When to use**:
- **Always use `True`** for large models (3B+ parameters)
- Essential on systems with limited RAM (16GB or less)
- Helps prevent crashes during model loading

### `attn_implementation`: How Attention is Computed

**What it is**: Controls which algorithm is used to compute attention (the core mechanism that lets models understand relationships between words).

**Options**:

1. **`"eager"`** (standard attention - recommended for compatibility)
   - Standard, reliable attention implementation
   - Works on **all hardware** (CPU, Apple Silicon, NVIDIA GPU)
   - Slightly slower but guaranteed to work
   - **Best choice**: When we want reliability and compatibility

2. **`"flash_attention_2"`** (optimized - CUDA GPUs only)
   - Highly optimized attention algorithm
   - **Much faster** and uses less memory
   - **Only works on NVIDIA GPUs** with CUDA
   - Requires `flash-attn` package installed
   - **Best choice**: When we have NVIDIA GPU and want maximum speed

3. **`"sdpa"`** (Scaled Dot Product Attention - PyTorch native)
   - PyTorch's built-in optimized attention
   - Faster than eager, works on more hardware than flash
   - Available on newer PyTorch versions
   - **Best choice**: When we want a middle ground (faster than eager, works on more devices than flash)

4. **`None`** (auto-detect)
   - Library tries to pick the best available option
   - May fall back to eager if others aren't available
   - May show warnings if flash-attention isn't available

**Performance comparison** (for 100 tokens):
- `"eager"`: ~30 seconds (CPU), ~5 seconds (GPU)
- `"sdpa"`: ~25 seconds (CPU), ~4 seconds (GPU)
- `"flash_attention_2"`: N/A (CPU), ~2 seconds (CUDA GPU only)

### `return_tensors="pt"`: PyTorch Format

**What it is**: Converts text into PyTorch tensors (special arrays for neural networks).

**Options**:
- `"pt"`: PyTorch tensors (what we use)
- `"tf"`: TensorFlow tensors (different framework)
- `"np"`: NumPy arrays (basic Python arrays)

**Example**:
```python
# Text: "Hello"
# return_tensors="pt" → tensor([[9906]])  (PyTorch tensor)
# Without it → [9906]  (regular Python list)
```

**Why it matters**: Models need tensors, not regular Python lists. It's like needing a specific container shape.

### `.to(model.device)`: Moving Data to the Right Place

**What it is**: Ensures our data is on the same device as our model.

**Problem**: Model might be on GPU, but our input data is on CPU → they can't talk!

**Example**:
```python
model = model.to("cuda")  # Model on GPU
inputs = tokenizer("Hello")  # Inputs on CPU ❌
inputs = inputs.to(model.device)  # Move inputs to GPU ✅
```

### `torch.no_grad()`: Inference Mode

**What it is**: Tells PyTorch "don't track gradients" (we're not training, just using the model).

**Why it matters**:
- **Training**: Need gradients to update weights (slower, uses more memory)
- **Inference**: Don't need gradients (faster, uses less memory)

**Example**:
```python
# With gradients (training):
with torch.enable_grad():
    output = model(input)  # Tracks how to update weights

# Without gradients (inference):
with torch.no_grad():
    output = model(input)  # Just get the answer, faster!
```

**Mental model**: 
- Training = Taking notes while learning (slower)
- Inference = Just answering questions (faster)

### `do_sample=True`: Sampling Strategy

**What it is**: Whether to randomly sample from probabilities or just pick the most likely token.

**Options**:
- `do_sample=True`: Randomly pick based on probabilities (more creative)
- `do_sample=False`: Always pick the most likely token (deterministic)

**Example**:
```python
# Model thinks next token probabilities:
# "blue" (60%), "clear" (30%), "vast" (10%)

do_sample=False → Always picks "blue"
do_sample=True → Picks "blue" 60% of time, "clear" 30%, "vast" 10%
```

**Mental model**: 
- `False` = Always choosing the safest answer
- `True` = Sometimes taking creative risks

**Note**: Usually used with `temperature` - `temperature` controls how much randomness, `do_sample` enables/disables it.

### `skip_special_tokens=True`: Clean Output

**What it is**: Removes special tokens (like `<pad>`, `<eos>`) from the final output.

**Example**:
```python
# Without skip_special_tokens:
"Hello world<eos><pad>"

# With skip_special_tokens=True:
"Hello world"
```

**Why it matters**: Special tokens are for the model's internal use, not for humans to read!

## 11. Why Small Models Matter

### Privacy
- Data never leaves our infrastructure
- No API calls to external services
- Complete control

### Cost
- No per-API-call fees
- Run on existing hardware
- Predictable costs

### Customization
- Fine-tune for our exact needs
- No waiting for vendor features
- Full control over behavior

### Independence
- Not dependent on external services
- Works offline
- No rate limits

## Mental Model Summary

```
Pre-trained Model (knows language)
    ↓
+ Our Examples (domain-specific)
    ↓
Fine-tuning (adjusts weights slightly)
    ↓
Fine-tuned Model (knows OUR domain)
    ↓
Inference (generates text for our use case)
```
