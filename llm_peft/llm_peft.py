import sys
import argparse
import logging
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PrefixTuningConfig, PromptTuningConfig, PromptEncoderConfig, IA3Config
from datasets import load_dataset
from tqdm import tqdm
from config import settings

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class PEFTTrainer:
    """
    A trainer class for fine-tuning language models using various PEFT (Parameter-Efficient Fine-Tuning) methods.
    """
    def __init__(
        self, model_name: str, dataset_config: dict, output_dir: str, epochs: int,
        batch_size: int, peft_method: str, peft_config: dict, eval_strategy: str,
        save_strategy: str, logging_steps: int, save_total_limit: int, fp16: bool,
        max_length: int = 512
    ):
        """
        Initializes the trainer with model and training parameters.
        
        Args:
            model_name (str): Pre-trained model name.
            dataset_config (dict): Dataset configuration containing name, subset and column to use.
            output_dir (str): Directory to save the model.
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.
            peft_method (str): PEFT method to use (lora, prefix_tuning, prompt_tuning, p_tuning, ia3).
            peft_config (dict): Configuration for the chosen PEFT method.
            eval_strategy (str): Evaluation strategy.
            save_strategy (str): Save strategy.
            logging_steps (int): Logging frequency in steps.
            save_total_limit (int): Limit on total saved checkpoints.
            fp16 (bool): Whether to use mixed precision training.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.model_name = model_name
        self.dataset_config = dataset_config
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.peft_method = peft_method
        self.peft_config = peft_config
        self.eval_strategy = eval_strategy
        self.save_strategy = save_strategy
        self.logging_steps = logging_steps
        self.save_total_limit = save_total_limit
        self.fp16 = fp16
        self.max_length = max_length

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=settings.huggingface_token, 
                trust_remote_code=True
            )
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, token=settings.huggingface_token,
                trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Successfully loaded model: {model_name}")
            self._apply_peft()
            
            # Load both raw and tokenized datasets
            self.raw_dataset, self.train_dataset, self.eval_dataset, self.test_dataset = self._load_and_tokenize_dataset() 
            logger.info(f"Dataset columns: {self.train_dataset.column_names}")
        except Exception as e:
            logger.error(f"Failed to initialize model or tokenizer: {e}")
            raise

    def _apply_peft(self):
        """Configures and applies the selected PEFT method to the model."""
        logger.info(f"Applying {self.peft_method} PEFT method")
        
        if self.peft_method == "lora":
            config = LoraConfig(
                r=self.peft_config.get("r", 8),
                lora_alpha=self.peft_config.get("alpha", 16),
                lora_dropout=self.peft_config.get("dropout", 0.05),
                bias=self.peft_config.get("bias", "none"),
                target_modules=self.peft_config.get("target_modules", ["q_proj", "v_proj"])
            )
        elif self.peft_method == "prefix_tuning":
            config = PrefixTuningConfig(
                task_type="CAUSAL_LM",
                num_virtual_tokens=self.peft_config.get("num_virtual_tokens", 20),
                prefix_projection=self.peft_config.get("prefix_projection", False),
                encoder_hidden_size=self.peft_config.get("encoder_hidden_size", None)
            )
        elif self.peft_method == "prompt_tuning":
            config = PromptTuningConfig(
                task_type="CAUSAL_LM",
                num_virtual_tokens=self.peft_config.get("num_virtual_tokens", 20),
                prompt_tuning_init=self.peft_config.get("prompt_tuning_init", "RANDOM"),
                tokenizer_name_or_path=self.model_name
            )
        elif self.peft_method == "p_tuning":
            config = PromptEncoderConfig(
                task_type="CAUSAL_LM",
                num_virtual_tokens=self.peft_config.get("num_virtual_tokens", 20),
                encoder_hidden_size=self.peft_config.get("encoder_hidden_size", 128),
                encoder_reparameterization_type=self.peft_config.get("encoder_reparameterization_type", "MLP"),
                encoder_num_layers=self.peft_config.get("encoder_num_layers", 2)
            )
        elif self.peft_method == "ia3":
            config = IA3Config(
                task_type="CAUSAL_LM",
                target_modules=self.peft_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
                feedforward_modules=self.peft_config.get("feedforward_modules", ["down_proj", "up_proj"])
            )
        else:
            raise ValueError(f"Unknown PEFT method: {self.peft_method}")
            
        self.finetuned_model = get_peft_model(self.base_model, config)
        logger.info(f"{self.peft_method} configuration applied successfully.")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.finetuned_model.parameters() if p.requires_grad)}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.finetuned_model.parameters())}")
    
    def _load_and_tokenize_dataset(self):
        """
        Loads the dataset and tokenizes the text data.
        
        Returns:
            Tuple: Raw dataset and tokenized train/validation/test datasets.
        """
        dataset_name = self.dataset_config.get("name")
        dataset_subset = self.dataset_config.get("subset", None)
        text_column = self.dataset_config.get("text_column", "text")
        split_names = self.dataset_config.get("splits", {
            "train": "train",
            "validation": "validation",
            "test": "test"
        })
        
        # Load the dataset using either dataset name or path
        try:
            if dataset_name.startswith("./") or "/" in dataset_name:
                # Local dataset
                dataset = load_dataset("json", data_files=dataset_name)
                # If local dataset doesn't have splits, create them manually
                if "train" not in dataset:
                    train_test = dataset["train"].train_test_split(test_size=0.1)
                    valid_test = train_test["test"].train_test_split(test_size=0.5)
                    dataset = {
                        "train": train_test["train"],
                        "validation": valid_test["train"],
                        "test": valid_test["test"]
                    }
            else:
                # HuggingFace dataset
                if dataset_subset:
                    dataset = load_dataset(dataset_name, dataset_subset)
                else:
                    dataset = load_dataset(dataset_name)
                    
            # Create a reference to raw dataset
            raw_dataset = dataset
                
            logger.info(f"Dataset loaded: {dataset_name}")
            logger.info(f"Dataset splits: {dataset.keys()}")
            logger.info(f"Dataset columns: {next(iter(dataset.values())).column_names}")
        
            # Ensure the dataset has the required splits
            for required_split in ["train", "validation", "test"]:
                actual_split = split_names.get(required_split)
                if actual_split not in dataset:
                    logger.warning(f"Missing {required_split} split. Creating from existing data.")
                    # If missing validation/test splits, create them from train
                    if required_split in ["validation", "test"] and "train" in dataset:
                        test_size = 0.1
                        if required_split == "validation" and "test" not in dataset:
                            test_size = 0.2  # Take 20% for validation if no test split
                        elif required_split == "test" and "validation" not in dataset:
                            test_size = 0.2  # Take 20% for test if no validation split
                        
                        if required_split == "validation":
                            split = dataset["train"].train_test_split(test_size=test_size)
                            dataset["train"] = split["train"]
                            dataset["validation"] = split["test"]
                        else:  # test
                            if "validation" in dataset:
                                # Split validation to get test
                                split = dataset["validation"].train_test_split(test_size=0.5)
                                dataset["validation"] = split["train"]
                                dataset["test"] = split["test"]
                            else:
                                # Split train to get test
                                split = dataset["train"].train_test_split(test_size=test_size)
                                dataset["train"] = split["train"]
                                dataset["test"] = split["test"]
                
            # Check if the text column exists
            for split in dataset.values():
                if text_column not in split.column_names:
                    available_columns = split.column_names
                    logger.error(f"Text column '{text_column}' not found. Available columns: {available_columns}")
                    # Try to find a suitable alternative
                    text_candidates = ["text", "content", "document", "sentence", "passage"]
                    for candidate in text_candidates:
                        if candidate in available_columns:
                            text_column = candidate
                            logger.info(f"Using '{text_column}' as text column instead.")
                            break
                    else:
                        raise ValueError(f"No suitable text column found in dataset. Available columns: {available_columns}")
                break

            def tokenize_function(examples):
                """Tokenizes dataset text with truncation and padding."""
                texts = examples[text_column]
                tokenized_inputs = self.tokenizer(
                    texts, truncation=True, padding="max_length", max_length=self.max_length
                )
                tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
                return tokenized_inputs
            
            # Remove problematic columns that might conflict with tokenizer output
            remove_columns = [col for col in next(iter(dataset.values())).column_names 
                             if col not in ["labels", "attention_mask", "token_type_ids"]]
            
            tokenized_datasets = {}
            for key, split in dataset.items():
                tokenized_datasets[key] = split.map(
                    tokenize_function, 
                    batched=True, 
                    remove_columns=remove_columns
                )
            
            logger.info("Dataset tokenized successfully.")
            return raw_dataset, tokenized_datasets["train"], tokenized_datasets["validation"], tokenized_datasets["test"]
            
        except Exception as e:
            logger.error(f"Failed to load or tokenize dataset: {e}")
            raise

    def train(self):
        """
        Trains the model using the specified PEFT configuration.
        
        This method sets up the training arguments, initializes the Trainer, 
        starts the training process, and saves the fine-tuned model and tokenizer.
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy=self.eval_strategy,
            save_strategy=self.save_strategy,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            logging_dir="./logs",
            logging_strategy="steps",
            logging_steps=self.logging_steps,
            disable_tqdm=False,
            report_to="none",
            num_train_epochs=self.epochs,
            save_total_limit=self.save_total_limit,
            fp16=self.fp16,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.finetuned_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )

        logger.info("Training started...")        
        trainer.train()
        logger.info("Training completed successfully.")

        self.finetuned_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Model and tokenizer saved to {self.output_dir}.")

    def _calculate_ppl(self, model, tokenizer, dataset, max_samples=100):
        """Calculates perplexity (PPL) for a dataset."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        text_column = self.dataset_config.get("text_column", "text")
        test_data = self.raw_dataset["test"].select(range(min(max_samples, len(self.raw_dataset["test"]))))

        with torch.no_grad():
            for sample in tqdm(test_data[text_column]):
                # Skip empty samples
                if not sample or not sample.strip():
                    continue
                    
                inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=self.max_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]

        if total_tokens == 0:
            logger.warning("No tokens processed during perplexity calculation!")
            return float('inf')
            
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity

    def test(self):
        """Evaluates and compares perplexity (PPL) between base and fine-tuned models."""
        logger.info("Calculating perplexity for base model...")
        base_ppl = self._calculate_ppl(self.base_model, self.tokenizer, self.raw_dataset["test"])
        
        logger.info("Calculating perplexity for fine-tuned model...")
        fine_tuned_ppl = self._calculate_ppl(self.finetuned_model, self.tokenizer, self.raw_dataset["test"])
        
        logger.info(f"PPL Comparison - Base: {base_ppl:.2f}, {self.peft_method.upper()}: {fine_tuned_ppl:.2f}")
        improvement = ((base_ppl - fine_tuned_ppl) / base_ppl) * 100
        logger.info(f"Improvement: {improvement:.2f}%")
        
        # Sample generation for qualitative evaluation
        prompt = self.dataset_config.get("sample_prompt", "The quick brown fox")
        self._generate_samples(prompt)
        
        results = {
            "base_ppl": base_ppl,
            "fine_tuned_ppl": fine_tuned_ppl,
            "improvement_percentage": improvement,
            "model": self.model_name,
            "peft_method": self.peft_method
        }
        
        # Save test results to file
        with open(f"{self.output_dir}/test_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        return base_ppl, fine_tuned_ppl
        
    def _generate_samples(self, prompt, num_samples=3, max_new_tokens=100):
        """Generates sample text using base and fine-tuned models for comparison."""
        logger.info(f"Generating samples using prompt: '{prompt}'")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate with base model
        logger.info("Base model output:")
        self.base_model.to(device)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        base_outputs = self.base_model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.7, 
            top_p=0.9,
            num_return_sequences=num_samples
        )
        
        for i, output in enumerate(base_outputs):
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            logger.info(f"Sample {i+1}: {text}")
        
        # Generate with fine-tuned model
        logger.info(f"Fine-tuned model ({self.peft_method}) output:")
        self.finetuned_model.to(device)
        tuned_outputs = self.finetuned_model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.7, 
            top_p=0.9,
            num_return_sequences=num_samples
        )
        
        for i, output in enumerate(tuned_outputs):
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            logger.info(f"Sample {i+1}: {text}")


def main():
    """Parses command-line arguments and starts training."""
    parser = argparse.ArgumentParser(description="Fine-tune language models with PEFT methods")
    parser.add_argument("--config", type=str, default="./args.json", help="Path to JSON config file")
    args = parser.parse_args()
    
    config = load_json(args.config)
    trainer = PEFTTrainer(**config)
    trainer.train()
    trainer.test()

def load_json(json_path):
    """Loads configuration from a JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    main()