"""
Ehnacements from simple llm_prune.py

1. Pruning sequence: 
 - applying both unstructured and structured pruning sequentially
 - might be overly aggressive and could significantly degrade model performance
 (solution) implementing an option to choose one method or the other.

2. Evaluation metrics: 
 - no evaluation of model performance before and after pruning
 (solution) Adding perplexity, accuracy, or other metrics would help measure the impact of optimizations

3. Gradient-based pruning: 
 - (solution) adding global pruning or magnitude-based pruning with gradients for better results

4. Quantization improvements:
 - Dynamic quantization is applied to the entire model
 - (solution) adding static quantization options 
              adding support for different precision levels (int8, int4)
              adding quantization-aware training

5. Error handling: 
 - (solution) Add error handling for when pruning or quantization fails

6. Layer targeting: 
 - (solution) Allow specifying which layers to prune rather than applying it to all linear layers

7. Progressive pruning: 
 - (solution) Implement iterative pruning where you gradually increase the pruning amount

8. Save original model: 
 - (solution) Add an option to save the original model for comparison

9. Model loading checks: 
 - (solution) Verify if the device has enough memory before loading large models

10. Temperature/sampling parameters: 
 - (solution) Add generation parameters like temperature and top_p for better text generation control
"""
import torch
import argparse
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils import prune
from datasets import load_dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrunedLLM:
    def __init__(self, model_name, device, cache_dir=None):
        """Initialize model and tokenizer"""
        self.device = device
        self.model_name = model_name
        self.original_model = None  # Store original model for comparison
        
        try:
            logger.info(f"Loading model {model_name} on {device}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            
            # Store original model params size
            self.original_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Model loaded successfully. Parameter count: {self.original_size:,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def backup_original_model(self):
        """Create a copy of the original model for comparison"""
        try:
            logger.info("Backing up original model")
            self.original_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
        except Exception as e:
            logger.error(f"Failed to backup original model: {str(e)}")
            
    def apply_unstructured_pruning(self, amount=0.3, target_modules=None):
        """Apply Unstructured Pruning (L1 norm)"""
        if amount <= 0:
            logger.info("Skipping unstructured pruning (amount <= 0)")
            return
            
        try:
            logger.info(f"Applying unstructured pruning with amount={amount}")
            pruned_count = 0
            
            for name, module in self.model.named_modules():
                # Skip if target_modules is specified and this module isn't in the list
                if target_modules and not any(target in name for target in target_modules):
                    continue
                    
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=amount)
                    prune.remove(module, 'weight')  # Make pruning permanent
                    pruned_count += 1

            current_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Unstructured pruning applied to {pruned_count} modules")
            logger.info(f"Model size after pruning: {current_size:,} parameters " +
                      f"({(1 - current_size/self.original_size) * 100:.2f}% reduction)")
                      
        except Exception as e:
            logger.error(f"Error during unstructured pruning: {str(e)}")
            raise
    
    def apply_structured_pruning(self, amount=0.3, n=2, dim=0, target_modules=None):
        """Apply Structured Pruning (Ln norm)
        
        Args:
            amount: Amount to prune
            n: Ln norm to use (1=L1, 2=L2, etc.)
            dim: 0 for row pruning (output neurons), 1 for column pruning (input connections)
            target_modules: List of module name patterns to target
        """
        if amount <= 0:
            logger.info("Skipping structured pruning (amount <= 0)")
            return
            
        try:
            logger.info(f"Applying structured pruning with amount={amount}, n={n}, dim={dim}")
            pruned_count = 0
            
            for name, module in self.model.named_modules():
                # Skip if target_modules is specified and this module isn't in the list
                if target_modules and not any(target in name for target in target_modules):
                    continue
                    
                if isinstance(module, torch.nn.Linear):
                    prune.ln_structured(module, name='weight', amount=amount, n=n, dim=dim)
                    prune.remove(module, 'weight')  # Make pruning permanent
                    pruned_count += 1

            current_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Structured pruning applied to {pruned_count} modules")
            logger.info(f"Model size after pruning: {current_size:,} parameters " +
                      f"({(1 - current_size/self.original_size) * 100:.2f}% reduction)")
                      
        except Exception as e:
            logger.error(f"Error during structured pruning: {str(e)}")
            raise
    
    def apply_global_pruning(self, amount=0.3, target_modules=None):
        """Apply Global Pruning across all target modules"""
        if amount <= 0:
            logger.info("Skipping global pruning (amount <= 0)")
            return
            
        try:
            logger.info(f"Applying global pruning with amount={amount}")
            parameters_to_prune = []
            
            for name, module in self.model.named_modules():
                # Skip if target_modules is specified and this module isn't in the list
                if target_modules and not any(target in name for target in target_modules):
                    continue
                    
                if isinstance(module, torch.nn.Linear):
                    parameters_to_prune.append((module, 'weight'))
            
            logger.info(f"Applying global pruning to {len(parameters_to_prune)} modules")
            
            if parameters_to_prune:
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=amount
                )
                
                # Make pruning permanent
                for module, name in parameters_to_prune:
                    prune.remove(module, name)

                current_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                logger.info(f"Model size after global pruning: {current_size:,} parameters " +
                          f"({(1 - current_size/self.original_size) * 100:.2f}% reduction)")
            else:
                logger.warning("No modules were selected for global pruning")
                
        except Exception as e:
            logger.error(f"Error during global pruning: {str(e)}")
            raise
    
    def apply_iterative_pruning(self, final_amount=0.5, steps=5, pruning_method="unstructured", target_modules=None):
        """Apply pruning in multiple iterations for better results"""
        if final_amount <= 0 or steps <= 0:
            logger.info("Skipping iterative pruning (invalid parameters)")
            return
            
        try:
            logger.info(f"Applying iterative pruning: method={pruning_method}, final_amount={final_amount}, steps={steps}")
            
            # Calculate amount per step (using formula that compounds correctly)
            # (1-x)^n = (1-target) => x = 1 - (1-target)^(1/n)
            amount_per_step = 1 - (1 - final_amount) ** (1 / steps)
            
            for step in range(1, steps + 1):
                logger.info(f"Pruning iteration {step}/{steps} with amount {amount_per_step:.4f}")
                
                if pruning_method == "unstructured":
                    self.apply_unstructured_pruning(amount=amount_per_step, target_modules=target_modules)
                elif pruning_method == "structured":
                    self.apply_structured_pruning(amount=amount_per_step, target_modules=target_modules)
                elif pruning_method == "global":
                    self.apply_global_pruning(amount=amount_per_step, target_modules=target_modules)
                else:
                    raise ValueError(f"Unknown pruning method: {pruning_method}")
                
                # Optionally evaluate after each step
                # self.evaluate_model(...)
                
        except Exception as e:
            logger.error(f"Error during iterative pruning: {str(e)}")
            raise
    
    def quantize_model(self, quantization_type="dynamic", dtype=torch.qint8, target_modules=None):
        """Apply Quantization to the model
        
        Args:
            quantization_type: "dynamic" or "static"
            dtype: Quantization precision (torch.qint8 or torch.quint8)
            target_modules: List of module types to quantize, defaults to Linear modules
        """
        if target_modules is None:
            target_modules = {torch.nn.Linear}
        
        try:
            logger.info(f"Applying {quantization_type} quantization with dtype {dtype}")
            
            if quantization_type == "dynamic":
                self.model = torch.quantization.quantize_dynamic(
                    self.model, 
                    target_modules,
                    dtype=dtype
                )
                logger.info("Dynamic quantization applied successfully")
            elif quantization_type == "static":
                # For static quantization, you'd need a calibration dataset
                # This is a simplified version
                model_fp32 = self.model
                model_fp32.eval()
                model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                model_prepared = torch.quantization.prepare(model_fp32)
                # Would need calibration here with representative inputs
                self.model = torch.quantization.convert(model_prepared)
                logger.info("Static quantization applied successfully")
            else:
                raise ValueError(f"Unknown quantization type: {quantization_type}")
            
        except Exception as e:
            logger.error(f"Error during model quantization: {str(e)}")
            raise
    
    def evaluate_model(self, dataset="wikitext", subset="wikitext-2-raw-v1", split="test", 
                      max_samples=100, stride=512, max_length=1024):
        """Evaluate model performance using perplexity"""
        try:
            logger.info(f"Evaluating model performance on {dataset}/{subset}")
            
            # Load evaluation dataset
            eval_dataset = load_dataset(dataset, subset, split=split)
            
            # Tokenize the dataset
            encodings = self.tokenizer("\n\n".join(eval_dataset["text"]), return_tensors="pt")
            
            # Limit to max_samples
            if max_samples and len(encodings.input_ids[0]) > max_samples * stride:
                max_length = max_samples * stride
            else:
                max_length = len(encodings.input_ids[0])
            
            # Create sequence windows for evaluation
            seq_len = encodings.input_ids.size(1)
            nlls = []
            prev_end_loc = 0
            
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
                target_ids = input_ids.clone()
                
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss * trg_len
                
                nlls.append(neg_log_likelihood)
                prev_end_loc = end_loc
                
                if end_loc == seq_len:
                    break
            
            ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
            logger.info(f"Perplexity: {ppl.item():.2f}")
            return ppl.item()
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            logger.warning("Continuing without evaluation...")
            return float('inf')
    
    def compare_with_original(self, prompt, max_length=50, **gen_kwargs):
        """Compare original and pruned/quantized model outputs"""
        if self.original_model is None:
            logger.warning("Original model not available for comparison")
            return
            
        try:
            logger.info("Comparing outputs between original and optimized models")
            
            # Generate with original model
            orig_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                orig_start = time.time()
                orig_output = self.original_model.generate(
                    **orig_inputs, 
                    max_length=max_length,
                    **gen_kwargs
                )
                orig_time = time.time() - orig_start
            orig_text = self.tokenizer.decode(orig_output[0], skip_special_tokens=True)
            
            # Generate with pruned/quantized model
            new_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                new_start = time.time()
                new_output = self.model.generate(
                    **new_inputs, 
                    max_length=max_length,
                    **gen_kwargs
                )
                new_time = time.time() - new_start
            new_text = self.tokenizer.decode(new_output[0], skip_special_tokens=True)
            
            # Compare and report
            logger.info(f"Original model generation time: {orig_time:.2f}s")
            logger.info(f"Optimized model generation time: {new_time:.2f}s")
            logger.info(f"Speed improvement: {(orig_time/new_time - 1) * 100:.2f}%")
            
            logger.info("\nOriginal model output:")
            logger.info(orig_text[:100] + "..." if len(orig_text) > 100 else orig_text)
            
            logger.info("\nOptimized model output:")
            logger.info(new_text[:100] + "..." if len(new_text) > 100 else new_text)
            
            return {
                "original": {
                    "text": orig_text,
                    "time": orig_time
                },
                "optimized": {
                    "text": new_text,
                    "time": new_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error during model comparison: {str(e)}")
            return None
    
    def generate_text(self, prompt, max_length=50, num_return_sequences=1, **gen_kwargs):
        """Generate text with more configurable parameters"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            generation_config = {
                "max_length": max_length,
                "num_return_sequences": num_return_sequences,
                "no_repeat_ngram_size": gen_kwargs.get("no_repeat_ngram_size", 0),
                "do_sample": gen_kwargs.get("do_sample", True),
                "top_k": gen_kwargs.get("top_k", 50),
                "top_p": gen_kwargs.get("top_p", 0.95),
                "temperature": gen_kwargs.get("temperature", 0.8),
            }
            
            with torch.no_grad():
                start_time = time.time()
                output = self.model.generate(**inputs, **generation_config)
                generation_time = time.time() - start_time
            
            generated_texts = [self.tokenizer.decode(output[i], skip_special_tokens=True) 
                              for i in range(len(output))]
            
            logger.info(f"Text generated in {generation_time:.2f}s")
            return generated_texts, generation_time
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return [f"Error: {str(e)}"], 0
    
    def save_model(self, save_path):
        """Save the optimized model and tokenizer"""
        try:
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"Saving model to {save_path}")
            
            # Save model and tokenizer
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Save metadata about optimization process
            metadata = {
                "original_size": self.original_size,
                "final_size": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "model_name": self.model_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save metadata as JSON
            import json
            with open(os.path.join(save_path, "pruning_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Model and metadata saved successfully to {save_path}")
            logger.info(f"Size reduction: {(1 - metadata['final_size']/metadata['original_size']) * 100:.2f}%")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Enhanced Pruning and Quantization for LLMs")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="facebook/opt-1.3b", 
                       help="Pretrained model name")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], 
                       default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Directory to cache models")
    parser.add_argument("--save_path", type=str, default="pruned_llm",
                       help="Path to save pruned model")
                       
    # Pruning configuration
    parser.add_argument("--backup_original", action="store_true",
                       help="Keep original model in memory for comparison")
    parser.add_argument("--pruning_method", type=str, 
                       choices=["unstructured", "structured", "global", "iterative", "none"],
                       default="unstructured", help="Pruning method to use")
    parser.add_argument("--unstructured_amount", type=float, default=0.3,
                       help="Amount for unstructured pruning (0-1)")
    parser.add_argument("--structured_amount", type=float, default=0.3,
                       help="Amount for structured pruning (0-1)")
    parser.add_argument("--structured_dim", type=int, choices=[0, 1], default=0,
                       help="Dimension for structured pruning (0=rows, 1=columns)")
    parser.add_argument("--global_amount", type=float, default=0.3,
                       help="Amount for global pruning (0-1)")
    parser.add_argument("--iterative_amount", type=float, default=0.5,
                       help="Final amount for iterative pruning (0-1)")
    parser.add_argument("--iterative_steps", type=int, default=5,
                       help="Number of steps for iterative pruning")
    parser.add_argument("--iterative_method", type=str, 
                       choices=["unstructured", "structured", "global"],
                       default="unstructured", help="Method for iterative pruning")
    parser.add_argument("--target_modules", type=str, nargs="+", default=None,
                       help="Target module patterns (e.g., 'attention' 'mlp')")
                       
    # Quantization configuration
    parser.add_argument("--quantize", action="store_true", help="Apply quantization")
    parser.add_argument("--quantization_type", type=str, choices=["dynamic", "static"],
                       default="dynamic", help="Quantization type")
                       
    # Evaluation configuration
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model after optimization")
    parser.add_argument("--eval_dataset", type=str, default="wikitext",
                       help="Dataset for evaluation")
    parser.add_argument("--eval_subset", type=str, default="wikitext-2-raw-v1",
                       help="Subset of evaluation dataset")
                       
    # Generation configuration
    parser.add_argument("--test_prompt", type=str, default="Once upon a time,",
                       help="Prompt for testing the model")
    parser.add_argument("--max_length", type=int, default=100,
                       help="Maximum length for generation")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p for nucleus sampling")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k for sampling")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                       help="Number of sequences to generate")
                       
    args = parser.parse_args()
    
    # Print summary of what will be done
    logger.info("="*50)
    logger.info("LLM Pruning and Quantization Tool")
    logger.info("="*50)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Pruning method: {args.pruning_method}")
    if args.quantize:
        logger.info(f"Quantization: {args.quantization_type}")
    logger.info("="*50)
    
    # Check for available memory if using CUDA
    if args.device == "cuda" and torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU total memory: {total_mem:.2f} GB")
    
    # Initialize model
    try:
        llm = PrunedLLM(args.model_name, args.device, cache_dir=args.cache_dir)
        
        # Backup original model if requested
        if args.backup_original:
            llm.backup_original_model()
            
        # Apply pruning based on selected method
        if args.pruning_method == "unstructured":
            llm.apply_unstructured_pruning(
                amount=args.unstructured_amount,
                target_modules=args.target_modules
            )
        elif args.pruning_method == "structured":
            llm.apply_structured_pruning(
                amount=args.structured_amount,
                dim=args.structured_dim,
                target_modules=args.target_modules
            )
        elif args.pruning_method == "global":
            llm.apply_global_pruning(
                amount=args.global_amount,
                target_modules=args.target_modules
            )
        elif args.pruning_method == "iterative":
            llm.apply_iterative_pruning(
                final_amount=args.iterative_amount,
                steps=args.iterative_steps,
                pruning_method=args.iterative_method,
                target_modules=args.target_modules
            )
        elif args.pruning_method == "none":
            logger.info("Skipping pruning as requested")

        # Apply quantization if requested
        if args.quantize:
            llm.quantize_model(
                quantization_type=args.quantization_type,
                dtype=torch.qint8
            )
            
        # Evaluate model if requested
        if args.evaluate:
            ppl = llm.evaluate_model(
                dataset=args.eval_dataset,
                subset=args.eval_subset
            )
            logger.info(f"Final perplexity: {ppl:.2f}")
            
        # Generate text with the optimized model
        gen_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "do_sample": True if args.temperature > 0 else False
        }
        
        # Compare with original if available
        if args.backup_original:
            comparison = llm.compare_with_original(
                args.test_prompt,
                max_length=args.max_length,
                **gen_kwargs
            )
        
        # Generate final text
        logger.info("\nGenerating text with optimized model:")
        generated_texts, _ = llm.generate_text(
            args.test_prompt,
            max_length=args.max_length,
            num_return_sequences=args.num_return_sequences,
            **gen_kwargs
        )
        
        for i, text in enumerate(generated_texts):
            logger.info(f"\nGenerated text {i+1}:")
            logger.info(text)
            
        # Save the optimized model
        llm.save_model(args.save_path)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
        
    logger.info("Process completed successfully")
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)