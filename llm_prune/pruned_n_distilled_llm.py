"""
PyTorch의 prune.remove 함수는 단순히 가중치를 '0'으로 만드는 것이지, 실제로 모델의 구조(행렬 크기)를 변경하지 못함
구조적 프루닝은 행이나 열 전체를 프루닝하므로 이론적으로 해당 차원을 제거할 수 있음
그러나 실제 구현에서는 새로운 모델과 레이어를 명시적으로 재설계하여야 함

문제점:
1. 모델 구조를 동적으로 변경하는 것은 복잡함. 특히 체인 구조가 아닌 경우에는 더욱 그러함
2. 프루닝으로 인해 레이어 간 차원이 일치하지 않게 되면 추가적인 조정이 필요함
3. 복잡한 모델(예. 트랜스포머)의 경우 더 많은 작업이 필요함

따라서 구조적 프루닝의 결과를 바탕으로 실제 구조를 변경하는 것은 가능하지만
그 과정은 모델 아키텍처에 따라 맞춤형으로 구현해야 함

구현 방향:
구조적 프루닝의 결과를 바탕으로 새 모델 아키텍처를 설계하고 지식 증류를 적용하는 방안

 add create_distilled_model()
 add _analyze_pruned_structure()
 add _create_compact_model()
 add _apply_knowledge_distillation()

 도전 과제:
 1. 모델 특화 구현 필요: _create_compact_model 함수를 모델 맞추형으로 구현해야 함
 2. 데이터 로더 필요: 지식 증류에는 학습 데이터 필요 self.train_dataloader 
 3. 태스크에 따른 손실 함수 조정: 분류 문제일 경우 CrossEntropyLoss 사용, 태스크에 맞게 손실 함수 조정 필요
 4. BERT/Tranformer 모델의 경우: 이런 복잡한 모델에서는 레이어 크기뿐 아니라 헤드 수, 어탠션 차원 등도 조정 필요
"""
import torch
import argparse
import time
import os
import copy
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

    def create_distilled_model(self, temperature=2.0, alpha=0.5, epochs=5):
        """
        구조적 프루닝 결과를 바탕으로 새 모델 설계 및 지식 증류 적용
        
        Args:
            temperature: 소프트 타겟 출력의 온도 파라미터 (높을수록 소프트한 확률 분포)
            alpha: 하드 타겟과 소프트 타겟 손실의 가중치 (0은 하드 타겟만, 1은 소프트 타겟만)
            epochs: 지식 증류에 사용할 학습 에포크 수
        """
        # 1. 구조적 프루닝을 적용하여 어떤 뉴런/연결이 중요한지 식별
        logger.info("Applying structural pruning to identify important connections...")
        self.apply_structured_pruning(amount=0.3, n=2, dim=0)  # 예시 파라미터
        
        # 2. 프루닝 결과 분석
        pruned_architecture = self._analyze_pruned_structure()
        
        # 3. 새 모델 아키텍처 생성
        student_model = self._create_compact_model(pruned_architecture)
        
        # 4. 지식 증류 적용
        self._apply_knowledge_distillation(
            teacher_model=self.model,  # 원래 프루닝된 모델을 교사로 사용
            student_model=student_model,
            temperature=temperature,
            alpha=alpha,
            epochs=epochs
        )
        
        # 5. 새 모델로 교체
        self.model = student_model
        
        # 6. 결과 로깅
        current_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Distilled model size: {current_size:,} parameters " +
                f"({(1 - current_size/self.original_size) * 100:.2f}% reduction)")
        
        return self.model

    def _analyze_pruned_structure(self):
        """
        프루닝된 모델 구조 분석하여 새 아키텍처 계획 수립
        """
        architecture_plan = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if hasattr(module, 'weight_mask'):
                    mask = module.weight_mask
                    
                    # 행 분석 (출력 뉴런)
                    row_importance = mask.sum(dim=1).cpu().numpy()
                    kept_rows = (row_importance > 0).sum()
                    
                    # 열 분석 (입력 연결)
                    col_importance = mask.sum(dim=0).cpu().numpy()
                    kept_cols = (col_importance > 0).sum()
                    
                    # 현재 레이어에 대한 새 크기 계획
                    architecture_plan[name] = {
                        'original_shape': module.weight.shape,
                        'new_out_features': kept_rows,
                        'new_in_features': kept_cols,
                        'row_importance': row_importance,
                        'col_importance': col_importance
                    }
        
        return architecture_plan

    def _create_compact_model(self, architecture_plan):
        """
        분석 결과를 바탕으로 더 작은 모델 생성
        """
        # 여기서는 모델 타입에 따라 구현이 달라질 수 있음
        # 예시: 간단한 MLP 모델의 경우
        
        if hasattr(self.model, "config"):
            # Hugging Face 모델인 경우
            new_config = copy.deepcopy(self.model.config)
            
            # 설정 업데이트 (모델 별로 다름)
            # 예: BERT 모델
            if hasattr(new_config, "hidden_size"):
                # 히든 레이어 크기 조정 등
                # (실제로는 더 복잡한 로직이 필요)
                pass
                
            new_model = type(self.model)(new_config)
        else:
            # 일반 PyTorch 모델의 경우: 새 레이어로 직접 모델 재구성
            # 예시 - 간단한 MLP 모델
            layers = []
            prev_size = None
            
            # architecture_plan을 사용하여 레이어 크기 결정
            for name, info in architecture_plan.items():
                in_size = info['new_in_features']
                out_size = info['new_out_features']
                
                if prev_size is not None and prev_size != in_size:
                    # 연속된 레이어 간 크기 불일치 해결
                    in_size = prev_size
                    
                layers.append(torch.nn.Linear(in_size, out_size))
                layers.append(torch.nn.ReLU())  # 활성화 함수 (모델에 맞게 조정)
                
                prev_size = out_size
                
            # 마지막 활성화 함수 제거 (필요시)
            if len(layers) > 0:
                layers.pop()
                
            new_model = torch.nn.Sequential(*layers)
        
        return new_model.to(self.device)

    def _apply_knowledge_distillation(self, teacher_model, student_model, 
                                    temperature=2.0, alpha=0.5, epochs=5):
        """
        지식 증류 적용하여 학생 모델 학습
        
        Args:
            teacher_model: 교사 모델 (원래 프루닝된 모델)
            student_model: 학생 모델 (새로 만든 작은 모델)
            temperature: 소프트 타겟의 온도
            alpha: 증류 손실과 원래 손실의 가중치
            epochs: 학습 에포크 수
        """
        # 최적화 도구 설정
        optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
        
        # 손실 함수
        ce_loss = torch.nn.CrossEntropyLoss()
        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        
        # 교사 모델을 평가 모드로 설정
        teacher_model.eval()
        
        # 데이터 로더 (여기서는 학습 데이터로더가 미리 정의되어 있다고 가정)
        train_loader = self.train_dataloader
        
        logger.info("Starting knowledge distillation...")
        
        for epoch in range(epochs):
            student_model.train()
            running_loss = 0.0
            
            for batch in train_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 교사 모델의 출력 (기울기 계산 없이)
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)
                
                # 학생 모델의 출력
                student_logits = student_model(inputs)
                
                # 소프트 타겟 생성
                soft_targets = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
                soft_prob = torch.nn.functional.log_softmax(student_logits / temperature, dim=1)
                
                # 두 손실 계산
                distillation_loss = kl_loss(soft_prob, soft_targets) * (temperature ** 2)
                student_loss = ce_loss(student_logits, targets)
                
                # 최종 손실 = 가중 평균
                loss = alpha * distillation_loss + (1 - alpha) * student_loss
                
                # 역전파 및 최적화
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # 에포크 결과 로깅
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
        
        logger.info("Knowledge distillation completed")

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