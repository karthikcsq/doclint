# LLM-Based Conflict Verification

DocLint supports local LLM inference for verifying potential conflicts between document chunks. This feature uses [llama.cpp](https://github.com/ggerganov/llama.cpp) via the Python bindings to run models entirely locally—no API keys required.

## Overview

The conflict detector has two stages:
1. **Heuristic Detection**: Fast pattern matching for negations, antonyms, and numerical mismatches
2. **LLM Verification** (optional): Deeper semantic analysis for uncertain cases

LLM verification improves accuracy by catching subtle contradictions that heuristics miss, while keeping false positives low.

## Quick Start

### 1. Install with LLM Support

```bash
# Install DocLint with LLM extras
pip install doclint[llm]

# Or if using Poetry
poetry install --with llm
```

### 2. Configure DocLint (Auto-Download)

Create `.doclint.toml` in your project. The model downloads automatically on first use (~2.5GB):

```toml
[conflict]
enabled = true
similarity_threshold = 0.85

[conflict.llm_verifier]
enabled = true
type = "llama_cpp_hf"  # Auto-download from Hugging Face
# Uses Phi-4 Mini Reasoning by default (best for conflict detection)
# repo_id = "unsloth/Phi-4-mini-reasoning-GGUF"
# filename = "Phi-4-mini-reasoning-Q4_K_M.gguf"
n_gpu_layers = -1  # Use GPU if available, 0 for CPU-only
```

### 3. Run DocLint

```bash
doclint scan ./docs
```

The model downloads automatically on first run. Subsequent runs use the cached model.

## Recommended Models

| Model | Repo ID | Size | Speed | Use Case |
|-------|---------|------|-------|----------|
| **Phi-4 Mini Reasoning** | `unsloth/Phi-4-mini-reasoning-GGUF` | ~2.5GB | Fast | **Default** - Best for conflict detection |
| Phi-4 Mini Instruct | `microsoft/Phi-4-mini-instruct-gguf` | ~2.5GB | Fast | General tasks |
| Llama 3.2 3B | `bartowski/Llama-3.2-3B-Instruct-GGUF` | ~2GB | Fast | Alternative option |
| Mistral 7B | `TheBloke/Mistral-7B-v0.1-GGUF` | ~4.5GB | Slower | Higher quality |

## GPU Acceleration

### CUDA (NVIDIA)

```bash
# Install with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Set GPU layers in config
[conflict.llm_verifier]
n_gpu_layers = -1  # Offload all layers to GPU
```

### Metal (Apple Silicon)

```bash
# Install with Metal support (usually automatic on macOS)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Config
[conflict.llm_verifier]
n_gpu_layers = -1
```

### CPU Only

```bash
# Standard installation (CPU only)
pip install llama-cpp-python

# Config
[conflict.llm_verifier]
n_gpu_layers = 0
```

## Programmatic Usage

### Auto-Download from Hugging Face (Recommended)

```python
from doclint.detectors import ConflictDetector, LlamaCppVerifier

# Create verifier - auto-downloads Phi-4 Mini on first use
verifier = LlamaCppVerifier.from_pretrained(
    # Uses defaults: unsloth/Phi-4-mini-reasoning-GGUF
    n_gpu_layers=-1,  # Use GPU
)

# Or specify a different model
verifier = LlamaCppVerifier.from_pretrained(
    repo_id="unsloth/Phi-4-mini-reasoning-GGUF",
    filename="Phi-4-mini-reasoning-Q4_K_M.gguf",
)

# Create detector with verifier
detector = ConflictDetector(
    similarity_threshold=0.85,
    verifier=verifier,
)

# Run detection
issues = await detector.detect(documents)
```

### Using a Local Model File

```python
from doclint.detectors import LlamaCppVerifier

verifier = LlamaCppVerifier(
    model_path="path/to/phi-4-mini.gguf",
    n_gpu_layers=-1,
)
```

### Using the Factory Function

```python
from doclint.detectors import create_verifier_from_config

# Auto-download from Hugging Face
config = {
    "type": "llama_cpp_hf",
    "repo_id": "unsloth/Phi-4-mini-reasoning-GGUF",
    "n_gpu_layers": -1,
}

# Or use local file
config = {
    "type": "llama_cpp",
    "model_path": "/path/to/model.gguf",
}

verifier = create_verifier_from_config(config)
```

## Testing Without a Model

For testing or CI/CD, use the mock verifier:

```python
from doclint.detectors import MockLlamaCppVerifier

verifier = MockLlamaCppVerifier(
    default_is_contradiction=True,
    default_confidence=0.9,
)
```

Or in config:

```toml
[conflict.llm_verifier]
enabled = true
type = "mock"
```

## Performance Tips

1. **Use GPU acceleration** when available—significantly faster
2. **Q4_K_M quantization** offers the best speed/quality tradeoff
3. **Batch similar checks** to keep the model warm
4. **Increase context size** (`n_ctx`) for longer chunks

## Troubleshooting

### "Model file not found"

Ensure the path is correct and the file exists:

```bash
ls -la ~/.doclint/models/
```

### "llama-cpp-python is required"

Install the LLM extras:

```bash
pip install doclint[llm]
```

### Slow inference

- Enable GPU acceleration
- Use a smaller/faster quantization (Q4_K_M vs Q8_0)
- Reduce `max_tokens` in config

### Out of memory

- Reduce `n_ctx` (context window size)
- Use a smaller model
- Set `n_gpu_layers` to a lower value (partial GPU offload)

## Model Licensing

- **Phi-4 Mini**: MIT License (Microsoft)
- **Llama 3**: Meta Community License
- **Mistral**: Apache 2.0

Always check the specific model's license for your use case.
