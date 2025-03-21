# GRAFT - GPU-optimized Reduction and Fine-Tuning Toolkit

GRAFT is a comprehensive toolkit for optimizing large language models (LLMs) through various techniques including knowledge distillation, parameter-efficient fine-tuning, pruning, and quantization.

## Repository Structure

```
GRAFT_examples/
├── LICENSE
├── llm_kd/             # Knowledge Distillation toolkit
│   ├── llm_kd.py
│   ├── README.md
│   └── requirements.txt
├── llm_peft/           # Parameter-Efficient Fine-Tuning toolkit
│   ├── args.json
│   ├── args_ptuning.json
│   ├── config.py
│   ├── llm_peft.py
│   └── requirements.txt
├── llm_prune/          # Pruning and Quantization toolkit
│   ├── pruned_llm.py
│   ├── pruned_n_distilled_llm.py
│   ├── pruned_n_rebuilt_llm.py
│   ├── prune.py
│   ├── README.md
│   ├── README_ko.md
│   ├── requirements.txt
│   ├── simple_prune_1.py
│   └── simple_prune_2.py
└── README.md
```

## Overview

GRAFT provides a set of tools to make large language models more efficient and deployable in resource-constrained environments. The toolkit includes three main components:

### 1. Knowledge Distillation (`llm_kd`)

Transfer knowledge from larger teacher models to smaller student models while maintaining performance.

**Key features:**
- Support for causal language models (GPT, OPT, Llama) and sequence-to-sequence models (T5, BART)
- Temperature-scaled knowledge distillation
- Memory-efficient chunked KL divergence computation
- Gradient accumulation for effective larger batch sizes
- Mixed precision training

### 2. Parameter-Efficient Fine-Tuning (`llm_peft`)

Fine-tune LLMs with minimal parameter updates using state-of-the-art PEFT methods.

**Supported techniques:**
- LoRA (Low-Rank Adaptation)
- P-Tuning
- Prefix Tuning
- Prompt Tuning
- Adapter-based methods

### 3. Pruning and Quantization (`llm_prune`)

Reduce model size and computational requirements through pruning and quantization.

**Key features:**
- Multiple pruning strategies (unstructured, structured, global, iterative)
- Quantization options (dynamic, static, INT8)
- Performance evaluation and benchmarking
- Side-by-side quality comparison

## Installation

Each toolkit component has its own requirements. Navigate to the specific directory and install dependencies:

```bash
cd llm_kd
pip install -r requirements.txt
```

## Usage

Each module includes detailed documentation and examples in its respective README.md file.

### Knowledge Distillation

```bash
python llm_kd/llm_kd.py \
  --teacher_model "facebook/opt-6.7b" \
  --student_model "facebook/opt-1.3b" \
  --task_type "causal" \
  --dataset "wikitext" \
  --output_dir "./distilled_model"
```

### Pruning and Quantization

```bash
python llm_prune/pruned_n_rebuilt_llm.py \
  --model_name "facebook/opt-1.3b" \
  --pruning_method iterative \
  --iterative_amount 0.4 \
  --iterative_steps 3 \
  --quantize \
  --evaluate
```

## Requirements

- Python 3.8+
- PyTorch 1.13+
- Transformers 4.26+
- Datasets 2.0+
- CUDA-capable GPU (recommended for larger models)

## License

This project is open-source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use GRAFT in your research, please cite:

```
@software{graft_toolkit,
  author = {GRAFT Contributors},
  title = {GRAFT: GPU-optimized Reduction and Fine-Tuning Toolkit},
  url = {https://github.com/username/GRAFT_examples},
  year = {2025},
}
```
