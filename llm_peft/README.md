# Parameter-Efficient Fine-Tuning for LLMs

This module provides implementations of various Parameter-Efficient Fine-Tuning (PEFT) methods for large language models. PEFT techniques allow for adapting large pre-trained models to specific tasks with minimal parameter updates, dramatically reducing computational requirements.

## Features

- **Multiple PEFT Methods**:
  - LoRA (Low-Rank Adaptation)
  - Prefix Tuning
  - Prompt Tuning
  - P-Tuning
  - IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

- **Comprehensive Training Pipeline**:
  - Automatic dataset handling and preprocessing
  - Split creation when needed
  - Text tokenization with proper padding
  - Perplexity evaluation
  - Sample generation for qualitative assessment

- **Flexible Configuration**:
  - JSON-based configuration
  - Support for local and HuggingFace datasets
  - Customizable training parameters

## Setup

1. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python3 -m venv llm_peft_env

   # Activate virtual environment
   # On Windows:
   llm_peft_env\Scripts\activate
   # On macOS/Linux:
   source llm_peft_env/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
    ```

## Usage

### Basic Usage

```bash
python llm_peft.py --config args.json
```

### Configuration Examples

The module uses JSON configuration files to define the fine-tuning process. Here are some examples:

#### LoRA Fine-Tuning

```json
{
  "model_name": "meta-llama/Llama-2-7b-hf",
  "dataset_config": {
    "name": "databricks/databricks-dolly-15k",
    "text_column": "context",
    "sample_prompt": "Explain quantum computing in simple terms"
  },
  "output_dir": "./output/llama2-lora",
  "epochs": 3,
  "batch_size": 4,
  "peft_method": "lora",
  "peft_config": {
    "r": 16,
    "alpha": 32,
    "dropout": 0.1,
    "bias": "none",
    "target_modules": ["q_proj", "v_proj"]
  },
  "eval_strategy": "steps",
  "save_strategy": "steps",
  "logging_steps": 100,
  "save_total_limit": 2,
  "fp16": true,
  "max_length": 512
}
```

#### P-Tuning Configuration

```json
{
  "model_name": "gpt2-large",
  "dataset_config": {
    "name": "./data/my_dataset.json",
    "text_column": "text",
    "sample_prompt": "The future of technology will"
  },
  "output_dir": "./output/gpt2-ptuning",
  "epochs": 5,
  "batch_size": 8,
  "peft_method": "p_tuning",
  "peft_config": {
    "num_virtual_tokens": 20,
    "encoder_hidden_size": 128,
    "encoder_num_layers": 2
  },
  "eval_strategy": "epoch",
  "save_strategy": "epoch",
  "logging_steps": 50,
  "save_total_limit": 3,
  "fp16": true,
  "max_length": 1024
}
```

### Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `model_name` | Pre-trained model name or path (from HuggingFace) |
| `dataset_config` | Dataset configuration including name, subset, text column |
| `output_dir` | Directory to save the fine-tuned model |
| `epochs` | Number of training epochs |
| `batch_size` | Batch size for training |
| `peft_method` | PEFT method to use (lora, prefix_tuning, prompt_tuning, p_tuning, ia3) |
| `peft_config` | Configuration specific to the chosen PEFT method |
| `eval_strategy` | Evaluation strategy (steps or epoch) |
| `save_strategy` | Save strategy (steps or epoch) |
| `logging_steps` | Logging frequency in steps |
| `save_total_limit` | Limit on total saved checkpoints |
| `fp16` | Whether to use mixed precision training |
| `max_length` | Maximum sequence length for tokenization |

## PEFT Methods Overview

### LoRA (Low-Rank Adaptation)
Adds low-rank matrices to transform layers, updating only these matrices during training.

Key parameters:
- `r`: Rank of the low-rank matrices
- `alpha`: Scaling factor
- `dropout`: Dropout probability
- `target_modules`: Which modules to apply LoRA to

### Prefix Tuning
Adds trainable prefix vectors to each transformer layer.

Key parameters:
- `num_virtual_tokens`: Number of virtual tokens to add
- `prefix_projection`: Whether to use an MLP to generate prefixes

### Prompt Tuning
Adds trainable embeddings to the input layer.

Key parameters:
- `num_virtual_tokens`: Number of virtual token embeddings to add
- `prompt_tuning_init`: How to initialize the prompt tuning embeddings

### P-Tuning
Adds learnable continuous prompts with an encoder network.

Key parameters:
- `num_virtual_tokens`: Number of virtual tokens
- `encoder_hidden_size`: Hidden size of the prompt encoder
- `encoder_num_layers`: Number of layers in the prompt encoder

### IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
Modulates activations with learned vectors.

Key parameters:
- `target_modules`: Modules to apply IA³ to
- `feedforward_modules`: Feedforward modules to apply IA³ to

## Evaluation

The module evaluates the fine-tuned models using:

1. **Perplexity (PPL)**: Measures how well the model predicts text
2. **Sample Generation**: Generates text samples for qualitative comparison

Results are saved to `output_dir/test_results.json`.

## Requirements

- torch
- transformers
- peft
- datasets
- tqdm

## License

This project is open-source and available under the MIT License.