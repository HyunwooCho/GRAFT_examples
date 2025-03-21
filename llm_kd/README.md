# LLM Knowledge Distillation Toolkit

This toolkit provides a comprehensive solution for performing knowledge distillation on large language models (LLMs) from Hugging Face's Transformers library. It supports distilling knowledge from larger teacher models to smaller student models to create more efficient models while maintaining performance.

## Features

- **Flexible Model Support**:
  - Causal language models (GPT, OPT, Llama, etc.)
  - Sequence-to-sequence models (T5, BART, etc.)
  - Automatic handling of model-specific requirements

- **Advanced Distillation Techniques**:
  - Temperature-scaled knowledge distillation
  - Memory-efficient chunked KL divergence computation
  - Gradient accumulation for effective larger batch sizes
  - Support for mixed precision training

- **Dataset Handling**:
  - Compatible with Hugging Face datasets
  - Automatic format detection for instruction-tuning, chat, and text datasets
  - Customizable sequence length and preprocessing

- **Training Optimizations**:
  - Multiple optimizer options (AdamW, SGD, Adafactor)
  - Learning rate scheduling with warmup
  - Gradient clipping
  - Checkpointing

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
   python3 -m venv llm_distillation_env

   # Activate virtual environment
   # On Windows:
   llm_distillation_env\Scripts\activate
   # On macOS/Linux:
   source llm_distillation_env/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python llm_kd.py \
  --teacher_model "facebook/opt-6.7b" \
  --student_model "facebook/opt-1.3b" \
  --task_type "causal" \
  --dataset "wikitext" \
  --output_dir "./distilled_model"
```

This will:
1. Load the OPT-6.7B model as the teacher
2. Load the OPT-1.3B model as the student
3. Train the student to mimic the teacher on the wikitext dataset
4. Save the distilled model to "./distilled_model"

### Advanced Usage

```bash
python llm_kd.py \
  --teacher_model "google/flan-t5-xl" \
  --student_model "google/flan-t5-base" \
  --task_type "seq2seq" \
  --dataset "databricks/databricks-dolly-15k" \
  --max_length 768 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --optimizer adafactor \
  --temperature 4.0 \
  --epochs 3 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.1 \
  --fp16 \
  --clip_grad_norm 1.0 \
  --save_every 1 \
  --output_dir "./distilled_flan_t5"
```

### Key Arguments

| Argument | Description |
|-----------|-------------|
| `--teacher_model` | Path or name of the teacher model on Hugging Face |
| `--student_model` | Path or name of the student model on Hugging Face |
| `--task_type` | Model architecture type (`causal` or `seq2seq`) |
| `--dataset` | Dataset name on Hugging Face or path to local dataset |
| `--max_length` | Maximum sequence length for tokenization |
| `--batch_size` | Batch size for training |
| `--gradient_accumulation_steps` | Number of steps to accumulate gradients |
| `--optimizer` | Optimizer type (`adamw`, `sgd`, or `adafactor`) |
| `--temperature` | Temperature for softening probability distributions |
| `--epochs` | Number of training epochs |
| `--learning_rate` | Learning rate for optimizer |
| `--warmup_ratio` | Ratio of steps for learning rate warmup |
| `--fp16` | Enable mixed precision training |
| `--clip_grad_norm` | Maximum norm for gradient clipping |
| `--save_every` | Save checkpoint every N epochs |
| `--output_dir` | Directory to save the distilled model |

## Examples

### Distilling a Chat Model

For distilling chat or instruction-tuned models:

```bash
python llm_kd.py \
  --teacher_model "meta-llama/Llama-2-7b-chat-hf" \
  --student_model "meta-llama/Llama-2-2b-chat-hf" \
  --task_type "causal" \
  --dataset "HuggingFaceH4/ultrachat_200k" \
  --max_length 1024 \
  --batch_size 2 \
  --gradient_accumulation_steps 16 \
  --temperature 2.0
```

### Memory-Constrained Environments

For systems with limited GPU memory:

```bash
python llm_kd.py \
  --teacher_model "EleutherAI/gpt-j-6b" \
  --student_model "EleutherAI/gpt-neo-1.3b" \
  --task_type "causal" \
  --dataset "openwebtext" \
  --max_length 512 \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --vocab_chunk_size 16 \
  --fp16
```

### Sequence-to-Sequence Tasks

For distilling seq2seq models like T5 or BART:

```bash
python llm_kd.py \
  --teacher_model "t5-large" \
  --student_model "t5-small" \
  --task_type "seq2seq" \
  --dataset "cnn_dailymail" \
  --max_length 512 \
  --temperature 3.0 \
  --optimizer "adafactor"
```

## Technical Details

The toolkit implements knowledge distillation using the following steps:

1. **Dataset Processing**: Handles various dataset formats and converts them to a consistent format based on task type
2. **Model Setup**: Loads teacher and student models, freezes teacher parameters
3. **Loss Calculation**: Implements chunked KL divergence to handle large vocabulary sizes efficiently
4. **Training Loop**: Manages the training process with gradient accumulation and checkpointing
5. **Optimization**: Applies learning rate scheduling and gradient clipping

## Dependencies

- torch
- transformers
- datasets
- tqdm

## License

This project is open-source and available under the MIT License.