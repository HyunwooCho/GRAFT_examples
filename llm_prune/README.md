# LLM Pruning and Quantization Example

This example provides a comprehensive solution for pruning and quantizing large language models (LLMs) from Hugging Face's Transformers library. It implements various pruning strategies, quantization techniques, and performance evaluation methods to optimize LLMs for deployment.

## Features

- **Multiple Pruning Methods**:
  - Unstructured pruning (magnitude-based)
  - Structured pruning (neuron/filter level)
  - Global pruning across layers
  - Iterative/progressive pruning
  - Targeted pruning for specific model components

- **Quantization Options**:
  - Dynamic quantization
  - Static quantization
  - INT8 precision support

- **Evaluation and Benchmarking**:
  - Perplexity evaluation on standard datasets
  - Performance comparison before/after optimization
  - Generation speed benchmarking
  - Side-by-side output quality comparison

- **Advanced Text Generation**:
  - Temperature control
  - Top-k and Top-p (nucleus) sampling
  - Multiple sequence generation

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.13+
- Transformers 4.26+
- Datasets 2.0+
- CUDA-capable GPU (recommended for larger models)

### Setup

1. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python3 -m venv llm_pruning_env

   # Activate virtual environment
   # On Windows:
   llm_pruning_env\Scripts\activate
   # On macOS/Linux:
   source llm_pruning_env/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python pruned_n_rebuilt_llm.py --model_name "facebook/opt-1.3b" --device cuda
```

This will:
1. Load the OPT-1.3B model
2. Apply default unstructured pruning (30%)
3. Generate sample text
4. Save the optimized model

### Advanced Usage

```bash
python pruned_n_rebuilt_llm.py \
  --model_name "facebook/opt-1.3b" \
  --pruning_method iterative \
  --iterative_amount 0.4 \
  --iterative_steps 3 \
  --target_modules "mlp" "attention" \
  --quantize \
  --backup_original \
  --evaluate \
  --test_prompt "In the distant future, humanity" \
  --temperature 0.7 \
  --max_length 150 \
  --save_path "./pruned_model_output"op
```

### Key Arguments

| Argument | Description |
|-----------|-------------|
| `--model_name` | Name of Hugging Face model to optimize |
| `--device` | Computing device (`cuda` or `cpu`) |
| `--pruning_method` | Method to use (`unstructured`, `structured`, `global`, `iterative`, `none`) |
| `--unstructured_amount` | Amount to prune using unstructured method (0-1) |
| `--structured_amount` | Amount to prune using structured method (0-1) |
| `--global_amount` | Amount to prune using global method (0-1) |
| `--iterative_amount` | Final amount for iterative pruning (0-1) |
| `--iterative_steps` | Number of steps for iterative pruning |
| `--target_modules` | Which module patterns to target (e.g., `attention`, `mlp`) |
| `--quantize` | Whether to apply quantization |
| `--quantization_type` | Type of quantization (`dynamic` or `static`) |
| `--evaluate` | Whether to evaluate model after optimization |
| `--backup_original` | Keep original model for comparison |
| `--test_prompt` | Text prompt for generation testing |
| `--temperature` | Temperature for generation (higher = more random) |
| `--top_p` | Top-p for nucleus sampling |
| `--max_length` | Maximum length for generation |

## Examples

### Memory-Constrained Environments

For systems with limited GPU memory:

```bash
python pruned_n_rebuilt_llm.py \
  --model_name "facebook/opt-1.3b" \
  --device cuda \
  --quantize \
  --pruning_method unstructured \
  --unstructured_amount 0.5
```

### Maximum Compression

To achieve maximum size reduction:

```bash
python pruned_n_rebuilt_llm.py \
  --model_name "facebook/opt-1.3b" \
  --pruning_method iterative \
  --iterative_amount 0.7 \
  --iterative_steps 5 \
  --quantize \
  --quantization_type dynamic
```

### Evaluation Focus

To focus on evaluating the impact of pruning:

```bash
python pruned_n_rebuilt_llm.py \
  --model_name "facebook/opt-1.3b" \
  --backup_original \
  --pruning_method global \
  --global_amount 0.3 \
  --evaluate \
  --test_prompt "Explain the theory of relativity"
```

## Dependencies

- torch
- transformers
- datasets
- numpy
- tqdm

## License

This project is open-source and available under the MIT License.[MIT License](LICENSE)