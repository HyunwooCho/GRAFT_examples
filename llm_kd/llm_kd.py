import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import logging
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    Adafactor, 
    DataCollatorWithPadding, 
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    get_scheduler
)
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, max_length=512, task_type="causal", split="train", num_samples=None):
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.max_length = max_length
        self.data = self.load_and_preprocess(dataset_name, split, num_samples)

    def load_and_preprocess(self, dataset_name, split, num_samples):
        # Load the dataset
        if os.path.exists(dataset_name):
            try:
                dataset = load_dataset("json", data_files=dataset_name)[split]
            except:
                dataset = load_dataset(dataset_name)[split]
        else:
            dataset = load_dataset(dataset_name)[split]
        
        # Limit number of samples if specified
        if num_samples is not None and num_samples > 0:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        formatted_data = []
        for entry in dataset:
            # Handle different dataset formats
            if "instruction" in entry and "output" in entry:
                input_text = entry["instruction"]
                if "input" in entry and entry["input"]:
                    input_text += "\n" + entry["input"]
                formatted_data.append((input_text, entry["output"]))
            elif "messages" in entry:
                input_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in entry["messages"][:-1]])
                output_text = entry["messages"][-1]["content"]
                formatted_data.append((input_text, output_text))
            elif "task" in entry and "input" in entry and "output" in entry:
                task = entry["task"]
                if task == "qa":
                    input_text = f"Question: {entry['input']} Answer:"
                elif task == "summarization":
                    input_text = f"Summarize the following text:\n{entry['input']}"
                else:
                    input_text = entry["input"]
                formatted_data.append((input_text, entry["output"]))
            elif "text" in entry:
                # For pure text datasets, try to split into chunks
                text = entry["text"]
                # Use a simple heuristic to split the text
                formatted_data.append((text[:len(text)//2], text[len(text)//2:]))
        
        logger.info(f"Loaded {len(formatted_data)} examples from dataset")
        return formatted_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, output_text = self.data[idx]
        
        if self.task_type == "causal":
            # For causal models, we concatenate input and output
            full_text = f"{input_text} {output_text}"
            encodings = self.tokenizer(
                full_text, 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            
            # Convert to PT tensor and remove batch dimension added by return_tensors="pt"
            input_ids = encodings["input_ids"].squeeze(0)
            attention_mask = encodings["attention_mask"].squeeze(0)
            
            # Create labels (for causal, labels are the same as input_ids)
            labels = input_ids.clone()
            
            # For causal LM, mask the labels for the input portion (only calculate loss on output)
            input_only_encoding = self.tokenizer(
                input_text, 
                padding=False, 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            input_length = input_only_encoding["input_ids"].size(1)
            # Set labels for the input portion to -100 (ignored in loss computation)
            if input_length < labels.size(0):
                labels[:input_length] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
        elif self.task_type == "seq2seq":
            # For seq2seq models, input and output are separated
            input_encodings = self.tokenizer(
                input_text, 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            
            output_encodings = self.tokenizer(
                output_text, 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            
            # Remove batch dimension added by return_tensors="pt"
            input_ids = input_encodings["input_ids"].squeeze(0)
            attention_mask = input_encodings["attention_mask"].squeeze(0)
            labels = output_encodings["input_ids"].squeeze(0)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

class ModelSetup:
    def __init__(self, args):
        """
        Initialize the ModelSetup class with command line arguments.
        
        Args:
            args: Command line arguments containing model configuration
        """
        self.args = args
        self.tokenizer = None
        self.teacher_model = None
        self.student_model = None
        self.device = None
        
    def setup_output_directory(self):
        """Set up the output directory if specified."""
        if self.args.output_dir:
            os.makedirs(self.args.output_dir, exist_ok=True)
            logger.info(f"Output directory created at {self.args.output_dir}")
            
    def load_tokenizer(self):
        """Load tokenizer from the teacher model."""
        logger.info(f"Loading tokenizer from {self.args.teacher_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.teacher_model)
        
        # Add padding token if not already there
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Setting pad_token to eos_token: {self.tokenizer.pad_token}")
        
        return self.tokenizer
    
    def load_models(self):
        """Load teacher and student models based on task type."""
        # Load teacher model
        logger.info(f"Loading teacher model from {self.args.teacher_model}")
        torch_dtype = torch.float16 if self.args.fp16 else None
        
        if self.args.task_type == "causal":
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.args.teacher_model,
                torch_dtype=torch_dtype
            )
            
            logger.info(f"Loading student model from {self.args.student_model}")
            self.student_model = AutoModelForCausalLM.from_pretrained(
                self.args.student_model,
                torch_dtype=torch_dtype
            )
            
        elif self.args.task_type == "seq2seq":
            self.teacher_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.args.teacher_model,
                torch_dtype=torch_dtype
            )
            
            logger.info(f"Loading student model from {self.args.student_model}")
            self.student_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.args.student_model,
                torch_dtype=torch_dtype
            )
        else:
            raise ValueError(f"Unsupported task type: {self.args.task_type}")
        
        return self.teacher_model, self.student_model
    
    def prepare_teacher_model(self):
        """Set teacher model to eval mode and freeze parameters."""
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
    def print_model_stats(self):
        """logger.info model sizes and compression ratio."""
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        
        logger.info(f"Teacher model size: {teacher_params/1e6:.2f}M parameters")
        logger.info(f"Student model size: {student_params/1e6:.2f}M parameters")
        logger.info(f"Compression ratio: {teacher_params/student_params:.2f}x")
    
    def setup_device(self):
        """Set up and return the device (GPU/CPU)."""
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        logger.info(f"Using device: {self.device}")
        return self.device
    
    def move_models_to_device(self):
        """Move teacher and student models to the device."""
        self.teacher_model.to(self.device)
        self.student_model.to(self.device)
    
    def setup(self):
        """Run the complete setup process and return all required objects."""
        self.setup_output_directory()
        self.load_tokenizer()
        self.load_models()
        self.prepare_teacher_model()
        self.print_model_stats()
        self.setup_device()
        self.move_models_to_device()
        
        return self.tokenizer, self.teacher_model, self.student_model, self.device

class KnowledgeDistillationTrainer:
    def __init__(self, args, student_model, teacher_model, tokenizer, optimizer, scheduler, dataloader, device):
        """
        Initialize the KnowledgeDistillationTrainer class.
        
        Args:
            args: Command line arguments containing training configuration
            student_model: The student model to be trained
            teacher_model: The teacher model for knowledge distillation
            tokenizer: Tokenizer for text processing
            optimizer: Optimizer for parameter updates
            scheduler: Learning rate scheduler
            dataloader: DataLoader containing training data
            device: Device (CPU/GPU) to use for training
        """
        self.args = args
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.device = device
        self.global_step = 0
        
    def train_epoch(self, epoch):
        """
        Train the student model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average loss for the epoch
        """
        self.student_model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Process batch and compute loss
            loss = self._process_batch(batch, step)
            
            # Update progress bar
            current_loss = loss.item() * self.args.gradient_accumulation_steps
            epoch_loss += current_loss
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
            
            # Log progress every log_interval steps
            if self.args.log_interval > 0 and step % self.args.log_interval == 0:
                tqdm.write(f"Epoch {epoch+1}/{self.args.epochs}, Batch {step}/{len(self.dataloader)}, Loss: {current_loss:.4f}")
        
        # Compute average epoch loss
        avg_loss = epoch_loss / len(self.dataloader)
        logger.info(f"Epoch {epoch+1}/{self.args.epochs}, Average Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _process_batch(self, batch, step):
        """
        Process a single batch through the models and compute the loss.
        
        Args:
            batch: The current batch data
            step: Current step within the epoch
            
        Returns:
            loss: The computed loss value (already scaled for gradient accumulation)
        """
        # Only zero gradients when we're going to update
        if (step + 1) % self.args.gradient_accumulation_steps == 0 or step == len(self.dataloader) - 1:
            self.optimizer.zero_grad()
        
        # Forward pass for student
        student_outputs = self.student_model(**batch, output_hidden_states=True)
        
        # Forward pass for teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**batch, output_hidden_states=True)
        
        # for debugging
        # student_logits = student_outputs.logits
        # teacher_logits = teacher_outputs.logits
        # logger.info(f"Student logits range: {student_logits.min().item()} to {student_logits.max().item()}")
        # logger.info(f"Teacher logits range: {teacher_logits.min().item()} to {teacher_logits.max().item()}")

        # Compute KL loss in chunks to save memory
        loss = compute_kl_loss_chunk(
            student_outputs.logits, 
            teacher_outputs.logits, 
            temperature=self.args.temperature,
            attention_mask=batch.get("attention_mask", None),
            chunk_size=self.args.vocab_chunk_size
        )
        
        # Scale loss for gradient accumulation
        loss = loss / self.args.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Only update every gradient_accumulation_steps or at the end
        if (step + 1) % self.args.gradient_accumulation_steps == 0 or step == len(self.dataloader) - 1:
            self._update_parameters()
        

        return loss
    
    def _update_parameters(self):
        """Apply gradient clipping and update model parameters."""
        # Apply gradient clipping if enabled
        if self.args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.args.clip_grad_norm)
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        
        # Update global step
        self.global_step += 1
    
    def save_checkpoint(self, epoch):
        """
        Save model checkpoint if configured.
        
        Args:
            epoch: Current epoch number
        """
        if self.args.save_every > 0 and (epoch + 1) % self.args.save_every == 0:
            checkpoint_path = f"{self.args.output_dir}/checkpoint-epoch-{epoch+1}"
            self.student_model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def train(self):
        """
        Run the full training loop for the specified number of epochs.
        
        Returns:
            all_losses: List of average losses for each epoch
        """
        all_losses = []
        
        for epoch in range(self.args.epochs):
            # Train for one epoch
            avg_loss = self.train_epoch(epoch)
            all_losses.append(avg_loss)
            
            # Save checkpoint if enabled
            self.save_checkpoint(epoch)
        
        return all_losses

def get_optimizer(optimizer_name, parameters, lr, weight_decay=0.01):
    if optimizer_name == "adamw":
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    elif optimizer_name == "adafactor":
        return Adafactor(
            parameters, 
            scale_parameter=True, 
            relative_step=True, 
            warmup_init=True, 
            lr=lr if lr is not None else None
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_collator(tokenizer, model, task_type):
    if task_type == "seq2seq":
        return DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, return_tensors="pt")
    elif task_type == "causal":
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        # Fallback to basic padding
        return DataCollatorWithPadding(tokenizer, padding=True, return_tensors="pt")

def compute_kl_loss_chunk(student_logits, teacher_logits, temperature, attention_mask=None, chunk_size=32):
    """
    Compute KL divergence loss between student and teacher model outputs in memory-efficient chunks
    with improved numerical stability.
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    device = student_logits.device
    
    # Check for NaN or inf in input
    if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
        logger.warning("WARNING: NaN or inf detected in student_logits")
        student_logits = torch.nan_to_num(student_logits, nan=0.0, posinf=100, neginf=-100)
    
    if torch.isnan(teacher_logits).any() or torch.isinf(teacher_logits).any():
        logger.warning("WARNING: NaN or inf detected in teacher_logits")
        teacher_logits = torch.nan_to_num(teacher_logits, nan=0.0, posinf=100, neginf=-100)
    
    # Initialize loss
    loss = torch.tensor(0.0, device=device)
    
    # Create masks based on attention_mask
    if attention_mask is not None:
        # Expand to match batch_size x seq_len x 1
        mask = attention_mask.unsqueeze(-1).float()
        non_pad_tokens = attention_mask.float().sum() + 1e-6  # Avoid division by zero
    else:
        mask = torch.ones((batch_size, seq_len, 1), device=device)
        non_pad_tokens = batch_size * seq_len
    
    # Process in chunks along the vocabulary dimension
    for i in range(0, vocab_size, chunk_size):
        end_idx = min(i + chunk_size, vocab_size)
        
        # Get logits for the current chunk
        student_chunk = student_logits[:, :, i:end_idx].clone()
        teacher_chunk = teacher_logits[:, :, i:end_idx].clone()
        
        # Clamp values for numerical stability
        student_chunk = torch.clamp(student_chunk, min=-100, max=100)
        teacher_chunk = torch.clamp(teacher_chunk, min=-100, max=100)
        
        # Apply temperature
        student_chunk = student_chunk / temperature
        teacher_chunk = teacher_chunk / temperature
        
        # Compute probabilities with improved numerical stability
        student_probs_chunk = nn.functional.log_softmax(student_chunk, dim=-1)
        teacher_probs_chunk = nn.functional.softmax(teacher_chunk, dim=-1)
        
        # Add small epsilon to avoid log(0)
        teacher_probs_chunk = teacher_probs_chunk + 1e-10
        
        # Check for NaN after softmax
        if torch.isnan(student_probs_chunk).any() or torch.isnan(teacher_probs_chunk).any():
            logger.warning(f"WARNING: NaN detected after softmax in chunk {i}")
            continue  # Skip this chunk
        
        # Compute loss for this chunk
        chunk_loss = nn.functional.kl_div(
            student_probs_chunk, 
            teacher_probs_chunk, 
            reduction="none",
            log_target=False
        )
        
        # Sum across vocab dimension (current chunk)
        chunk_loss = chunk_loss.sum(dim=-1, keepdim=True)
        
        # Check for NaN in loss
        if torch.isnan(chunk_loss).any():
            logger.warning(f"WARNING: NaN detected in loss for chunk {i}")
            continue  # Skip this chunk
        
        # Add to total loss (weighted by mask)
        loss = loss + (chunk_loss * mask).sum()
    
    # Normalize by number of non-pad tokens
    loss = loss / non_pad_tokens
    
    # Apply temperature scaling factor
    loss = loss * (temperature ** 2)
    
    # Final NaN check
    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning("WARNING: Final loss is NaN or inf, returning small positive value")
        return torch.tensor(0.1, device=device, requires_grad=True)
    
    return loss

def main(args):
    # Print summary of what will be done
    logger.info("="*50)
    logger.info("LLM Knowledge Distillation Tool")
    logger.info("="*50)
    logger.info(f"Teacher Model: {args.teacher_model}")
    logger.info(f"Student Model: {args.student_model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Task type: {args.task_method}")
    logger.info("="*50)

    # Initialize models and run setup
    models = ModelSetup(args)
    tokenizer, teacher_model, student_model, device = models.setup()
    
    # Initialize optimizer
    optimizer = get_optimizer(
        args.optimizer, 
        student_model.parameters(), 
        args.learning_rate, 
        args.weight_decay
    )
    
    # Load and prepare dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = TextDataset(
        args.dataset, 
        tokenizer, 
        max_length=args.max_length, 
        task_type=args.task_type,
        num_samples=args.num_samples
    )
    
    # Get appropriate data collator
    collate_fn = get_collator(tokenizer, student_model, args.task_type)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Set up lr scheduler
    num_training_steps = len(dataloader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Starting training for {args.epochs} epochs")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    # Initialize trainer and run training
    trainer = KnowledgeDistillationTrainer(
        args, student_model, teacher_model, tokenizer, 
        optimizer, scheduler, dataloader, device
    )
    losses = trainer.train()
    
    # Save final model
    if args.output_dir:
        student_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Saved final model to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation for LLMs")

    # Model arguments
    parser.add_argument("--teacher_model", type=str, required=True, help="Pretrained teacher model name or path")
    parser.add_argument("--student_model", type=str, required=True, help="Pretrained student model name or path")
    parser.add_argument("--task_type", type=str, choices=["causal", "seq2seq"], required=True, help="Task type: causal or seq2seq")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name on Hugging Face")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd", "adafactor"], default="adamw", help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for KL divergence loss")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Max norm for gradient clipping (0 to disable)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--vocab_chunk_size", type=int, default=32, 
                        help="Chunk size for vocabulary dimension in KL loss computation")
    parser.add_argument("--fp16", action="store_true", 
                        help="Use mixed precision training")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, 
                        help="Ratio of total training steps used for warmup")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="The scheduler type to use")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to use from dataset (for testing)")
    
    # Miscellaneous arguments
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval (batches)")
    parser.add_argument("--output_dir", type=str, default="distilled_model", help="Directory to save the distilled model")
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs (0 to disable)")
    
    args = parser.parse_args()
    main(args)
