# GRAFT - Generative AI Refinement & Fine-Tuning Toolkit

GRAFT is a comprehensive toolkit for optimizing large language models (LLMs) through various techniques including knowledge distillation, parameter-efficient fine-tuning, pruning, and quantization. This toolkit enables researchers and developers to create more efficient and deployable LLMs while maintaining performance.

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
- Dataset handling for instruction-tuning, chat, and text formats

### 2. Parameter-Efficient Fine-Tuning (`llm_peft`)

Fine-tune LLMs with minimal parameter updates using state-of-the-art PEFT methods.

**Supported techniques:**
- LoRA (Low-Rank Adaptation)
- Prefix Tuning
- Prompt Tuning
- P-Tuning
- IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

**Key features:**
- Comprehensive training pipeline
- Automatic dataset handling with split creation
- Perplexity evaluation and sample generation
- JSON-based configuration system
- Support for local and HuggingFace datasets

### 3. Pruning and Quantization (`llm_prune`)

Reduce model size and computational requirements through pruning and quantization.

**Key features:**
- Multiple pruning strategies (unstructured, structured, global, iterative)
- Quantization options (dynamic, static, INT8)
- Performance evaluation and benchmarking
- Side-by-side quality comparison
- Advanced text generation with sampling controls

## Installation

Each toolkit component has its own requirements. Navigate to the specific directory and install dependencies:

```bash
# For Knowledge Distillation
cd llm_kd
python3 -m venv llm_kd_env
source llm_kd_env/bin/activate
(llm_kd_env) pip install -r requirements.txt

# For Parameter-Efficient Fine-Tuning
cd llm_peft
python3 -m venv llm_peft_env
source llm_peft_env/bin/activate
(llm_peft_env) pip install -r requirements.txt

# For Pruning and Quantization
cd llm_prune
python3 -m venv llm_prune_env
source llm_prune_env/bin/activate
(llm_prune_env) pip install -r requirements.txt
```

## Usage Examples

### Knowledge Distillation

```bash
python llm_kd/llm_kd.py \
  --teacher_model "facebook/opt-6.7b" \
  --student_model "facebook/opt-1.3b" \
  --task_type "causal" \
  --dataset "wikitext" \
  --output_dir "./distilled_model"
```

### Parameter-Efficient Fine-Tuning

```bash
# Using LoRA
python llm_peft/llm_peft.py --config llm_peft/args.json

# Using P-Tuning
python llm_peft/llm_peft.py --config llm_peft/args_ptuning.json
```

Example configuration (LoRA):
```json
{
  "model_name": "meta-llama/Llama-2-7b-hf",
  "dataset_config": {
    "name": "databricks/databricks-dolly-15k",
    "text_column": "context"
  },
  "output_dir": "./output/llama2-lora",
  "epochs": 3,
  "batch_size": 4,
  "peft_method": "lora",
  "peft_config": {
    "r": 16,
    "alpha": 32,
    "target_modules": ["q_proj", "v_proj"]
  },
  "eval_strategy": "steps",
  "save_strategy": "steps",
  "fp16": true
}
```

### Pruning and Quantization

```bash
python llm_prune/pruned_n_rebuilt_llm.py \
  --model_name "facebook/opt-1.3b" \
  --pruning_method iterative \
  --iterative_amount 0.4 \
  --iterative_steps 3 \
  --target_modules "mlp" "attention" \
  --quantize \
  --evaluate
```

## Technical Details

### Knowledge Distillation
Implements knowledge distillation using temperature-scaled KL divergence to transfer knowledge from a larger teacher model to a smaller student model. The process involves:

1. Dataset processing for consistency across different formats
2. Model setup with frozen teacher parameters
3. Chunked KL divergence for memory efficiency
4. Training loop with gradient accumulation
5. Optimization with learning rate scheduling

### Parameter-Efficient Fine-Tuning
PEFT methods enable adaptation of large models with updates to only a small subset of parameters:

1. LoRA adds low-rank matrices to transform layers
2. Prefix Tuning adds trainable prefix vectors to transformer layers
3. Prompt Tuning adds trainable embeddings to the input
4. P-Tuning adds learnable continuous prompts with an encoder
5. IA³ modulates activations with learned vectors

The module handles dataset preparation, training, and evaluation with perplexity comparison.

### Pruning and Quantization
Implements various techniques to reduce model size and inference time:

1. Unstructured pruning removes individual weights
2. Structured pruning removes entire neurons/filters
3. Global pruning operates across all layers
4. Iterative pruning gradually increases sparsity
5. Quantization reduces precision of weights and activations

## Requirements

- Python 3.8+
- PyTorch 1.13+
- Transformers 4.26+
- Datasets 2.0+
- PEFT library (for fine-tuning)
- CUDA-capable GPU (recommended for larger models)

## Use Cases

- **Edge Deployment**: Create smaller models suitable for mobile and edge devices
- **Cost Reduction**: Lower compute and memory requirements for inference
- **Domain Adaptation**: Efficiently adapt models to specific domains or tasks
- **Research**: Experiment with various optimization techniques
- **Production**: Optimize models for production deployment with reduced latency

## License

This project is open-source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use GRAFT in your research, please cite:

```
@software{graft,
  author = {tenace@etri.re.kr},
  title = {GRAFT: Generative AI Refinement & Fine-Tuning Toolkit},
  url = {https://github.com/HyunwooCho/GRAFT_examples},
  year = {2025},
}
```
