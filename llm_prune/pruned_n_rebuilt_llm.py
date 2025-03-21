"""
_analyze_pruned_structure() 함수 설명

1. 어텐션 헤드 분석: 
 - Transformer의 핵심인 어텐션 헤드의 중요도를 개별적으로 분석
 - 프루닝 결과를 통해 어떤 헤드가 중요한지 파악하고 새 모델에서는 중요한 헤드만 유지

2. 히든 차원 일관성: 
 - Transformer에서는 여러 컴포넌트 간에 차원이 일치해야 함
 - 예를 들어 Query, Key, Value 출력 차원이 모두 일치해야 하며, 레이어 간에도 출력 차원이 일치해야 함
 - 이를 위해 프루닝 결과를 바탕으로 일관된 차원을 결정

3. 레이어 중요도 분석: 
 - Transformer 모델에서는 어떤 레이어가 더 중요한지 분석하여 덜 중요한 레이어를 완전히 제거

4. 헤드 수 최적화: 
 - 새로운 히든 차원에 맞게 어텐션 헤드 수를 조정
 - 일반적으로는 기존 헤드 크기를 유지하는 방향으로 헤드 수를 결정

5. FFN 크기 분석: 
 - Transformer의 Feed-Forward Network 크기도 분석하여 새 모델에서의 최적 크기를 결정

6. 컴포넌트별 분류: 
 - Transformer는 다양한 컴포넌트(어텐션, FFN, 임베딩 등)로 구성되어 있으므로, 각 컴포넌트별로 분석 결과를 분류

_analyze_attention_heads() 함수 설명
: LLaMA 모델에 대한 어텐션 헤드 분석에서 고려해야 할 주요 차이점

1. 모델 구조의 차이: 
 - LLaMA는 self.model.layers에 레이어가 저장되며, 각 레이어 내에 self_attn 모듈이 있음
2. 어텐션 구현의 차이: 
 - LLaMA는 일반적으로 q_proj, k_proj, v_proj, o_proj로 구성된 어텐션을 사용
 - BERT와 같은 통합된 어텐션 가중치가 아닌 별도의 프로젝션을 사용
3. 그룹 쿼리 어텐션(GQA)/다중 쿼리 어텐션(MQA): 
 - LLaMA 모델(특히 최신 버전)은 KV 헤드 수가 쿼리 헤드 수보다 적을 수 있으므로, num_key_value_heads 설정 확인
4. 헤드 중요도 계산: 
 - LLaMA에서는 쿼리, 키, 값, 출력 프로젝션 모두에서 헤드의 중요도를 합산하여 종합적인 중요도를 계산
"""

import torch
import argparse
import time
import os
import copy
import re
import numpy as np
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

    def rebuild_model(self):
        """
        구조적 프루닝 결과를 바탕으로 새 모델 설계
        """
        # 1. 프루닝 결과 분석
        pruned_architecture = self._analyze_pruned_structure()
        
        # 2. 새 모델 아키텍처 생성
        self.model = self._create_compact_model(pruned_architecture)

    def _analyze_pruned_structure(self):
        """
        Transformer 모델의 프루닝된 구조 분석
        """
        architecture_plan = {}
        
        # 주요 분석 대상 컴포넌트들
        attention_components = []
        ffn_components = []
        embedding_components = []
        
        # 각 모듈 분석
        logger.info("Analyzing the pruned model's component modules")
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if hasattr(module, 'weight_mask'):
                    mask = module.weight_mask
                    
                    # 행/열 중요도 계산
                    row_importance = mask.sum(dim=1).cpu().numpy()
                    col_importance = mask.sum(dim=0).cpu().numpy()
                    
                    # 살아남은 뉴런/연결 수
                    kept_rows = (row_importance > 0).sum()
                    kept_cols = (col_importance > 0).sum()
                    
                    # 컴포넌트 분류 (이름 패턴 기반)
                    component_info = {
                        'original_shape': module.weight.shape,
                        'new_out_features': kept_rows,
                        'new_in_features': kept_cols,
                        'row_importance': row_importance,
                        'col_importance': col_importance
                    }
                    
                    architecture_plan[name] = component_info
                    
                    # Transformer 특화 컴포넌트 분류
                    if any(x in name for x in ['query', 'key', 'value', 'attention']):
                        attention_components.append((name, component_info))
                    elif any(x in name for x in ['ffn', 'intermediate', 'output']):
                        ffn_components.append((name, component_info))
                    elif any(x in name for x in ['embedding', 'embed']):
                        embedding_components.append((name, component_info))


        # Transformer 특화 분석 수행
        
        # 1. 어텐션 헤드 분석
        head_importance = self._analyze_attention_heads()
        
        # 2. 히든 차원 일관성 확인 및 조정
        hidden_size_analysis = self._analyze_hidden_dimensions(architecture_plan)

        # 3. 레이어별 중요도 분석
        layer_importance = self._analyze_layer_importance()
        
        # 종합 결과
        transformer_analysis = {
            'head_importance': head_importance,
            'hidden_size_analysis': hidden_size_analysis,
            'layer_importance': layer_importance,
            'attention_components': attention_components,
            'ffn_components': ffn_components,
            'embedding_components': embedding_components
        }
        
        architecture_plan['transformer_analysis'] = transformer_analysis
        
        return architecture_plan

    def _analyze_attention_heads(self):
        """
        어텐션 헤드의 중요도 분석
        """
        head_importance = {}
        
        # 모델 타입에 따라 적절한 방법으로 어텐션 헤드 중요도 계산
        # BERT 모델의 경우
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            logger.info("Analyzing attention heads of BERT model")
            for layer_idx, layer in enumerate(self.model.encoder.layer):
                if hasattr(layer, "attention"):
                    # 1. 어텐션 출력에 대한 마스크 분석
                    if hasattr(layer.attention.output, "dense") and hasattr(layer.attention.output.dense, "weight_mask"):
                        mask = layer.attention.output.dense.weight_mask
                        # 헤드 크기 추정 (일반적으로 hidden_size / num_heads)
                        hidden_size = mask.shape[1]
                        if hasattr(self.model.config, "num_attention_heads"):
                            num_heads = self.model.config.num_attention_heads
                        else:
                            # 일반적인 값으로 가정
                            num_heads = 12  # BERT-base 기본값
                        
                        head_size = hidden_size // num_heads
                        
                        # 각 헤드의 중요도 계산
                        head_importances = []
                        for head_idx in range(num_heads):
                            start_idx = head_idx * head_size
                            end_idx = (head_idx + 1) * head_size
                            head_mask = mask[:, start_idx:end_idx]
                            head_imp = head_mask.sum().item()
                            head_importances.append(head_imp)
                        
                        head_importance[f"layer_{layer_idx}"] = head_importances

        # LLaMA 모델인 경우
        elif hasattr(self.model, "layers"):  # LLaMA의 기본 구조
            logger.info("Analyzing attention heads of LLaMA model")
            for layer_idx, layer in enumerate(self.model.layers):
                # LLaMA의 경우 일반적으로 self_attn 모듈을 가짐
                if hasattr(layer, "self_attn"):
                    # LLaMA의 어텐션은 일반적으로 o_proj(출력 투영)을 가짐
                    if hasattr(layer.self_attn, "o_proj") and hasattr(layer.self_attn.o_proj, "weight_mask"):
                        mask = layer.self_attn.o_proj.weight_mask
                        
                        # LLaMA의 구성 파라미터 접근
                        if hasattr(self.model.config, "num_attention_heads"):
                            num_heads = self.model.config.num_attention_heads
                        elif hasattr(self.model.config, "num_heads"):
                            num_heads = self.model.config.num_heads
                        else:
                            # LLaMA 모델의 일반적인 값
                            num_heads = 32  # LLaMA의 기본 헤드 수 (모델 크기에 따라 다름)
                        
                        # LLaMA에서는 추가로 그룹 쿼리 어텐션(GQA) 또는 다중 쿼리 어텐션(MQA)을 사용할 수 있음
                        if hasattr(self.model.config, "num_key_value_heads"):
                            num_kv_heads = self.model.config.num_key_value_heads
                        else:
                            num_kv_heads = num_heads  # 기본적으로 전체 헤드 수와 동일
                        
                        # 어텐션 출력의 차원
                        hidden_size = mask.shape[0]  # 출력 차원
                        head_size = hidden_size // num_heads
                        
                        # 어텐션 출력 프로젝션에서의 헤드 중요도 계산
                        if hasattr(layer.self_attn, "q_proj") and hasattr(layer.self_attn.q_proj, "weight_mask"):
                            q_mask = layer.self_attn.q_proj.weight_mask
                            
                            # 각 헤드의 중요도 계산
                            head_importances = []
                            for head_idx in range(num_heads):
                                start_idx = head_idx * head_size
                                end_idx = (head_idx + 1) * head_size
                                
                                # 쿼리 어텐션의 해당 헤드 부분
                                q_head_mask = q_mask[start_idx:end_idx, :]
                                
                                # KV 헤드는 MQA/GQA에서 다르게 계산
                                kv_head_idx = head_idx % num_kv_heads
                                
                                # 출력 프로젝션에서의 해당 헤드 부분
                                o_head_mask = mask[:, start_idx:end_idx]
                                
                                # 헤드의 총 중요도 (입력과 출력 프로젝션 모두 고려)
                                head_imp = q_head_mask.sum().item() + o_head_mask.sum().item()
                                
                                # K, V 프로젝션도 있다면 추가
                                if hasattr(layer.self_attn, "k_proj") and hasattr(layer.self_attn.k_proj, "weight_mask"):
                                    k_mask = layer.self_attn.k_proj.weight_mask
                                    kv_start_idx = kv_head_idx * head_size
                                    kv_end_idx = (kv_head_idx + 1) * head_size
                                    k_head_mask = k_mask[kv_start_idx:kv_end_idx, :]
                                    head_imp += k_head_mask.sum().item()
                                    
                                if hasattr(layer.self_attn, "v_proj") and hasattr(layer.self_attn.v_proj, "weight_mask"):
                                    v_mask = layer.self_attn.v_proj.weight_mask
                                    kv_start_idx = kv_head_idx * head_size
                                    kv_end_idx = (kv_head_idx + 1) * head_size
                                    v_head_mask = v_mask[kv_start_idx:kv_end_idx, :]
                                    head_imp += v_head_mask.sum().item()
                                
                                head_importances.append(head_imp)
                        
                        # q_proj 마스크가 없는 경우, o_proj만 사용하여 대략적인 중요도 계산
                        else:
                            head_importances = []
                            for head_idx in range(num_heads):
                                start_idx = head_idx * head_size
                                end_idx = (head_idx + 1) * head_size
                                head_mask = mask[:, start_idx:end_idx]
                                head_imp = head_mask.sum().item()
                                head_importances.append(head_imp)
                        
                        head_importance[f"layer_{layer_idx}"] = head_importances
        
        # 다른 Transformer 계열 모델 (GPT 등)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            logger.info("Analyzing attention heads of GPT model")
            # GPT 스타일 모델
            for layer_idx, block in enumerate(self.model.transformer.h):
                if hasattr(block, "attn") and hasattr(block.attn, "c_proj") and hasattr(block.attn.c_proj, "weight_mask"):
                    mask = block.attn.c_proj.weight_mask
                    
                    # GPT 구성에서 헤드 수 가져오기
                    num_heads = getattr(self.model.config, "n_head", 12)  # 기본값 12
                    hidden_size = mask.shape[1]
                    head_size = hidden_size // num_heads
                    
                    # 각 헤드의 중요도 계산
                    head_importances = []
                    for head_idx in range(num_heads):
                        start_idx = head_idx * head_size
                        end_idx = (head_idx + 1) * head_size
                        head_mask = mask[:, start_idx:end_idx]
                        head_imp = head_mask.sum().item()
                        head_importances.append(head_imp)
                    
                    head_importance[f"layer_{layer_idx}"] = head_importances
        
        # OPT 모델 처리
        elif hasattr(self.model, "model") and hasattr(self.model.model, "decoder") and hasattr(self.model.model.decoder, "layers"):
            logger.info("Analyzing attention heads of OPT model")
            
            # OPT 모델에서 decoder 레이어 가져오기
            decoder_layers = self.model.model.decoder.layers
            
            # 모델 구성에서 헤드 수 가져오기
            if hasattr(self.model.config, "num_attention_heads"):
                num_heads = self.model.config.num_attention_heads
            else:
                # OPT-1.3B의 기본 헤드 수
                num_heads = 32  # 모델 크기에 따라 조정 필요
            
            for layer_idx, layer in enumerate(decoder_layers):
                # OPT의 셀프 어텐션 모듈 접근
                if hasattr(layer, "self_attn"):
                    # 출력 프로젝션 가중치 마스크 접근
                    if hasattr(layer.self_attn, "out_proj") and hasattr(layer.self_attn.out_proj, "weight_mask"):
                        mask = layer.self_attn.out_proj.weight_mask
                        
                        # 어텐션 출력의 차원
                        hidden_size = mask.shape[0]  # 출력 차원
                        head_size = hidden_size // num_heads
                        
                        # 어텐션 헤드 중요도 계산
                        head_importances = []
                        
                        # 각 헤드의 입력 및 출력 프로젝션에서의 중요도 계산
                        if hasattr(layer.self_attn, "q_proj") and hasattr(layer.self_attn.q_proj, "weight_mask"):
                            q_mask = layer.self_attn.q_proj.weight_mask
                            k_mask = layer.self_attn.k_proj.weight_mask if hasattr(layer.self_attn, "k_proj") else None
                            v_mask = layer.self_attn.v_proj.weight_mask if hasattr(layer.self_attn, "v_proj") else None
                            
                            for head_idx in range(num_heads):
                                # 각 헤드의 시작 및 끝 인덱스 계산
                                start_idx = head_idx * head_size
                                end_idx = (head_idx + 1) * head_size
                                
                                # 쿼리 프로젝션에서의 헤드 중요도
                                q_head_mask = q_mask[start_idx:end_idx, :]
                                q_importance = q_head_mask.sum().item()
                                
                                # 키 프로젝션에서의 헤드 중요도
                                k_importance = 0
                                if k_mask is not None:
                                    k_head_mask = k_mask[start_idx:end_idx, :]
                                    k_importance = k_head_mask.sum().item()
                                
                                # 값 프로젝션에서의 헤드 중요도
                                v_importance = 0
                                if v_mask is not None:
                                    v_head_mask = v_mask[start_idx:end_idx, :]
                                    v_importance = v_head_mask.sum().item()
                                
                                # 출력 프로젝션에서의 헤드 중요도
                                out_head_mask = mask[:, start_idx:end_idx]
                                out_importance = out_head_mask.sum().item()
                                
                                # 헤드의 총 중요도 계산
                                total_importance = q_importance + k_importance + v_importance + out_importance
                                head_importances.append(total_importance)
                        
                        # 단순히 출력 프로젝션만 사용한 대략적인 중요도 계산
                        else:
                            for head_idx in range(num_heads):
                                start_idx = head_idx * head_size
                                end_idx = (head_idx + 1) * head_size
                                head_mask = mask[:, start_idx:end_idx]
                                head_importance = head_mask.sum().item()
                                head_importances.append(head_importance)
                        
                        head_importance[f"layer_{layer_idx}"] = head_importances

        # 그 밖의 모델 처리
        else:
            logger.info("Unknown model architecture detected. Using generic attention head analysis.")
            self._print_attributes(self.model)

        return head_importance

    def _analyze_hidden_dimensions(self, architecture_plan):
        """
        히든 차원의 일관성 분석 및 조정된 차원 제안

        목적:
         - 모델의 다양한 레이어에서 사용되는 차원들을 분석
         - 모델 압축/재구성 시 일관된 차원 값을 제안
         - 어텐션 헤드 수를 추천
        """
        logger.info("Analyzing hidden dimension consistency and adjusting dimensions")
        hidden_sizes = {}
        ffn_sizes = {}
        
        # 모델 타입 확인
        model_type = getattr(self.model.config, "model_type", "unknown")
        logger.info(f"Analyzing dimensions for model type: {model_type}")
        
        # 레이어 간 차원 분석
        for name, info in architecture_plan.items():
            logger.debug(f"Analyzing layer: {name} with info: {info}")
            
            # OPT 모델의 경우 패턴이 다를 수 있음
            layer_idx = None
            
            # 일반적인 패턴 시도: 레이어 이름에서 인덱스 추출(layer.0, layers.0 등의 패턴 인식)
            layer_match = re.search(r'layer\.(\d+)|layers\.(\d+)', name)
            if layer_match:
                layer_idx = int(layer_match.group(1) if layer_match.group(1) else layer_match.group(2))
            
            # OPT 특화 패턴 (facebook/opt-1.3B 모델의 경우 decoder.layers.0 같은 특수 패턴 처리)
            if model_type == "opt" and not layer_idx:
                layer_match = re.search(r'decoder\.layers\.(\d+)', name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
            
            # 어텐션 관련 레이어 분석
            if layer_idx is not None:
                # 쿼리/키/밸류 처리: 쿼리, 키, 밸류 및 그 프로젝션 레이어 식별하여 그 출력 차원을 hidden_sizes 딕셔너리에 저장
                if any(x in name.lower() for x in ['query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj']):
                    if layer_idx not in hidden_sizes:
                        hidden_sizes[layer_idx] = []
                    hidden_sizes[layer_idx].append(info['new_out_features'])
                
                # FFN 처리: FFN, FC, MLP, 중간 레이어 식별하여 그 출력 차원을 ffn_sizes 딕셔너리에 저장
                elif any(x in name.lower() for x in ['intermediate', 'fc', 'ffn', 'mlp']):
                    if layer_idx not in ffn_sizes:
                        ffn_sizes[layer_idx] = []
                    ffn_sizes[layer_idx].append(info['new_out_features'])
        
        logger.info(f"Found hidden sizes: {hidden_sizes}")
        logger.info(f"Found FFN sizes: {ffn_sizes}")
        
        # 기본값 설정 - 모델의 원래 설정에서 가져옴
        default_hidden_size = getattr(self.model.config, "hidden_size", 768)
        default_ffn_size = getattr(self.model.config, "ffn_dim", default_hidden_size * 4)
        default_num_heads = getattr(self.model.config, "num_attention_heads", 12)
        
        # 일관된 히든 사이즈 결정
        consistent_hidden_size = default_hidden_size  # 기본값 사용
        if hidden_sizes:
            all_sizes = [size for sizes in hidden_sizes.values() for size in sizes]
            if all_sizes:
                consistent_hidden_size = int(np.mean(all_sizes))
        
        # 헤드 크기의 배수로 맞추기
        num_heads = default_num_heads
        if hasattr(self.model.config, "num_attention_heads"):
            num_heads = self.model.config.num_attention_heads
        
        # hidden_size가 너무 작지 않도록 보장
        consistent_hidden_size = max(consistent_hidden_size, 32)
        
        # 헤드 크기의 배수로 조정
        head_size = max(consistent_hidden_size // num_heads, 8)  # 최소 head_size 보장
        consistent_hidden_size = head_size * num_heads
        
        # 일관된 FFN 사이즈 결정
        consistent_ffn_size = default_ffn_size  # 기본값 사용
        if ffn_sizes:
            all_sizes = [size for sizes in ffn_sizes.values() for size in sizes]
            if all_sizes:
                consistent_ffn_size = int(np.mean(all_sizes))
        
        # FFN 사이즈가 너무 작지 않도록 보장
        consistent_ffn_size = max(consistent_ffn_size, consistent_hidden_size * 2)
        
        # 추천 헤드 수
        recommended_num_heads = self._recommend_num_heads(consistent_hidden_size)
        
        logger.info(f"Final dimensions - Hidden: {consistent_hidden_size}, FFN: {consistent_ffn_size}, Heads: {recommended_num_heads}")
        
        return {
            'per_layer_hidden_sizes': hidden_sizes,
            'per_layer_ffn_sizes': ffn_sizes,
            'consistent_hidden_size': consistent_hidden_size,
            'consistent_ffn_size': consistent_ffn_size,
            'recommended_num_heads': recommended_num_heads
        }

    def _analyze_layer_importance(self):
        """
        Transformer 레이어별 중요도 분석
        """
        layer_importance = {}
        
        # 레이어별 가중치 중요도 합산
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight_mask'):
                layer_match = re.search(r'layer\.(\d+)', name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    if layer_idx not in layer_importance:
                        layer_importance[layer_idx] = 0
                    
                    # 이 레이어의 중요도를 마스크 합으로 계산
                    importance = module.weight_mask.sum().item()
                    layer_importance[layer_idx] += importance
        
        # 중요도 기준 레이어 정렬
        sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'layer_importance': layer_importance,
            'sorted_layers': sorted_layers,
            'recommended_num_layers': self._recommend_num_layers(sorted_layers)
        }

    def _recommend_num_heads(self, hidden_size):
        """
        새 히든 사이즈에 적합한 어텐션 헤드 수 추천
        """
        # 기본 어텐션 헤드 수 가져오기
        if hasattr(self.model.config, "num_attention_heads"):
            original_num_heads = self.model.config.num_attention_heads
        else:
            original_num_heads = 12  # 일반적인 기본값
        
        # 새 히든 사이즈에 맞게 헤드 수 조정
        # 일반적으로 헤드 크기를 일정하게 유지하는 것이 좋음
        if hasattr(self.model.config, "hidden_size"):
            original_hidden_size = self.model.config.hidden_size
            original_head_size = original_hidden_size // original_num_heads
            
            # 새 히든 사이즈로 가능한 헤드 수 계산
            possible_num_heads = hidden_size // original_head_size
            
            # 적어도 1개 이상의 헤드 필요
            return max(1, possible_num_heads)
        else:
            # 히든 사이즈 정보가 없으면 히든 사이즈의 1/64 정도로 헤드 수 추정
            return max(1, hidden_size // 64)

    def _recommend_num_layers(self, sorted_layers):
        """
        중요도에 따라 유지할 레이어 수 추천
        """
        if not sorted_layers:
            return 0
        
        # 전체 중요도 합 계산
        total_importance = sum(imp for _, imp in sorted_layers)
        
        # 중요도가 누적 90%를 차지하는 레이어 수 계산
        cumulative_importance = 0
        for i, (_, imp) in enumerate(sorted_layers):
            cumulative_importance += imp
            if cumulative_importance / total_importance >= 0.9:
                return i + 1
        
        # 기본적으로 모든 레이어 유지
        return len(sorted_layers)

    def _create_compact_model(self, architecture_plan):
        """
        분석 결과를 바탕으로 더 작은 모델 생성
        """
        # 분석된 차원 정보 가져오기
        # dims_analysis = self._analyze_hidden_dimensions(architecture_plan)
        dims_analysis = architecture_plan['transformer_analysis']['hidden_size_analysis']
        consistent_hidden_size = dims_analysis['consistent_hidden_size']
        consistent_ffn_size = dims_analysis['consistent_ffn_size']
        recommended_num_heads = dims_analysis['recommended_num_heads']
        
        if hasattr(self.model, "config"):
            # Hugging Face 모델인 경우
            new_config = copy.deepcopy(self.model.config)
            model_type = getattr(new_config, "model_type", None)
            
            # 모델 타입에 따른 설정 업데이트
            if model_type == "bert":
                # BERT 모델 설정 업데이트
                logger.info("Updating configuration for the BERT model")

                new_config.hidden_size = consistent_hidden_size
                new_config.intermediate_size = consistent_ffn_size
                new_config.num_attention_heads = recommended_num_heads
                
            elif model_type == "llama" or model_type == "mistral" or "llama" in str(type(self.model)).lower():
                # LLaMA 계열 모델 설정 업데이트
                logger.info("Updating configuration for the LLaMA or Mistral model")

                new_config.hidden_size = consistent_hidden_size
                new_config.intermediate_size = consistent_ffn_size
                new_config.num_attention_heads = recommended_num_heads
                new_config.num_key_value_heads = recommended_num_heads  # GQA 지원 (필요시)
                
                # LLaMA 특화 설정
                if hasattr(new_config, "ffn_dim"):
                    new_config.ffn_dim = consistent_ffn_size
                
                # 헤드 차원 업데이트
                if hasattr(new_config, "head_dim"):
                    new_config.head_dim = consistent_hidden_size // recommended_num_heads
                else:
                    # 명시적 헤드 차원 설정이 없는 경우
                    pass
                    
            elif model_type == "gpt2" or model_type == "gpt_neo" or "gpt" in str(type(self.model)).lower():
                # GPT 계열 모델 설정 업데이트
                logger.info("Updating configuration for the GPT model")

                new_config.n_embd = consistent_hidden_size
                new_config.n_head = recommended_num_heads
                
                # GPT 계열 특화 FFN 설정
                if hasattr(new_config, "n_inner"):
                    new_config.n_inner = consistent_ffn_size
                elif hasattr(new_config, "ffn_dim"):
                    new_config.ffn_dim = consistent_ffn_size
            
            elif model_type == "opt" or "opt" in str(type(self.model)).lower():
                # OPT 모델 설정 업데이트
                logger.info("Updating configuration for the OPT model")
                
                # 기존 설정에서 기본값 가져오기
                original_hidden_size = getattr(self.model.config, "hidden_size", 768)
                original_ffn_size = getattr(self.model.config, "ffn_dim", original_hidden_size * 4)
                original_num_heads = getattr(self.model.config, "num_attention_heads", 12)
                
                # 분석값이 0이거나 너무 작으면 원래 값의 비율로 축소
                reduction_factor = self.compression_rate if hasattr(self, "compression_rate") else 0.75
                
                # 안전한 값 계산
                safe_hidden_size = consistent_hidden_size if consistent_hidden_size > 32 else int(original_hidden_size * reduction_factor)
                safe_ffn_size = consistent_ffn_size if consistent_ffn_size > 64 else int(original_ffn_size * reduction_factor)
                safe_num_heads = recommended_num_heads if recommended_num_heads > 0 else max(1, int(original_num_heads * reduction_factor))
                
                # 헤드 수에 맞게 hidden_size 조정
                safe_hidden_size = (safe_hidden_size // safe_num_heads) * safe_num_heads
                
                # 로깅
                logger.info(f"OPT dimensions - Using Hidden: {safe_hidden_size}, FFN: {safe_ffn_size}, Heads: {safe_num_heads}")
                
                # 설정 업데이트
                new_config.hidden_size = safe_hidden_size
                new_config.ffn_dim = safe_ffn_size
                new_config.num_attention_heads = safe_num_heads
                
                # head_dim 명시적 설정
                new_config.head_dim = safe_hidden_size // safe_num_heads
                
                # OPT 특화 설정
                if hasattr(new_config, "word_embed_proj_dim"):
                    new_config.word_embed_proj_dim = safe_hidden_size

            elif model_type == "t5" or "t5" in str(type(self.model)).lower():
                # T5 계열 모델 설정 업데이트
                logger.info("Updating configuration for the T5 model")

                new_config.d_model = consistent_hidden_size
                new_config.d_ff = consistent_ffn_size
                new_config.num_heads = recommended_num_heads
                
                # T5 인코더-디코더 특화 설정
                if hasattr(new_config, "d_kv"):
                    new_config.d_kv = consistent_hidden_size // recommended_num_heads
                    
            elif model_type == "roberta" or "roberta" in str(type(self.model)).lower():
                # RoBERTa는 BERT와 구조가 유사함
                logger.info("Updating configuration for the RoBERTa model")

                new_config.hidden_size = consistent_hidden_size
                new_config.intermediate_size = consistent_ffn_size
                new_config.num_attention_heads = recommended_num_heads
                
            elif model_type == "bart" or "bart" in str(type(self.model)).lower():
                # BART 모델 설정 업데이트
                logger.info("Updating configuration for the BART model")

                new_config.d_model = consistent_hidden_size
                new_config.encoder_ffn_dim = consistent_ffn_size
                new_config.decoder_ffn_dim = consistent_ffn_size
                new_config.encoder_attention_heads = recommended_num_heads
                new_config.decoder_attention_heads = recommended_num_heads
                
            else:
                # 기타 Transformer 기반 모델에 대한 일반적인 설정 업데이트
                logger.info("Updating configuration for an unspecified Transformer-based model")

                # 일반적인 파라미터 이름 변형들 시도
                for hidden_attr in ["hidden_size", "d_model", "n_embd", "hidden_dim"]:
                    if hasattr(new_config, hidden_attr):
                        setattr(new_config, hidden_attr, consistent_hidden_size)
                        break
                        
                for ffn_attr in ["intermediate_size", "ffn_dim", "d_ff", "n_inner"]:
                    if hasattr(new_config, ffn_attr):
                        setattr(new_config, ffn_attr, consistent_ffn_size)
                        break
                        
                for head_attr in ["num_attention_heads", "n_head", "num_heads"]:
                    if hasattr(new_config, head_attr):
                        setattr(new_config, head_attr, recommended_num_heads)
                        break
    
            # 임베딩 차원 조정 (필요한 경우)
            if hasattr(new_config, "vocab_size") and hasattr(new_config, "hidden_size"):
                # 임베딩 차원이 히든 차원과 일치해야 하는 경우
                pass
                
            # 새 모델 인스턴스 생성
            try:
                new_model = type(self.model)(new_config)
                
                # 모델의 가중치 초기화 (필요시)
                if hasattr(new_model, "init_weights"):
                    new_model.init_weights()
            except Exception as e:
                print(f"모델 생성 중 오류 발생: {e}")
                # 대체 초기화 방법 시도 
                new_model = self._initialize_model_from_scratch(new_config, model_type)
        else:
            # 일반 PyTorch 모델: 아키텍처 계획에 따라 레이어 구성
            
            def _is_transformer_like_structure(architecture_plan):
                # 아키텍처 계획이 Transformer 구조를 가지고 있는 지 확인
                transformer_components = ['query', 'key', 'value', 'attention', 'ffn', 'intermediate']
                for name in architecture_plan.keys():
                    if any(comp in name.lower() for comp in transformer_components):
                        return True
                return False
            
            if _is_transformer_like_structure(architecture_plan):
                # 구조화된 모델 타입 감지(Transformer component가 있을 경우)
                # Transformer 스타일 모델 구성 시도 (실제로는 더 복잡한 구현이 필요함)
                # 예: nn.MultiheadAttention 등을 사용한 구현 또는 자체 Transformer 클래스 구현

                # 레이어 수 추출
                dims = dims_analysis.get('per_layer_hidden_sizes')
                if dims:
                    num_layers = max(dims) + 1
                else:
                    num_layers = 1 

                # PyTorch의 기본 Transformer 인코더 사용하여 모델 생성
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=consistent_hidden_size,
                    nhead=recommended_num_heads,
                    dim_feedforward=consistent_ffn_size,
                    batch_first=True
                )

                new_model = torch.nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=num_layers
                )
            else:
                # 일반 MLP 또는 기타 구조(Transformer component가 없을 경우)
                # 순차적 레이어 구조의 모델 생성

                layers = []
                prev_size = None
                
                # 레이어 이름 정렬로 순서 유지
                sorted_layers = sorted(architecture_plan.items(), key=lambda x: x[0])
                
                for name, info in sorted_layers:
                    in_size = info['new_in_features']
                    out_size = info['new_out_features']
                    
                    if prev_size is not None and prev_size != in_size:
                        # 연속된 레이어 간 크기 불일치 해결
                        in_size = prev_size
                        
                    # 활성화 함수 결정
                    if 'activation' in info:
                        activation = info['activation']
                    elif 'gelu' in name.lower():
                        activation = 'gelu'
                    elif 'relu' in name.lower():
                        activation = 'relu'
                    else:
                        activation = 'linear'
                        
                    # 레이어 추가
                    layers.append(torch.nn.Linear(in_size, out_size))
                    
                    # 활성화 함수 추가
                    if activation == 'gelu':
                        layers.append(torch.nn.GELU())
                    elif activation == 'relu':
                        layers.append(torch.nn.ReLU())
                    elif activation == 'tanh':
                        layers.append(torch.nn.Tanh())
                    elif activation == 'sigmoid':
                        layers.append(torch.nn.Sigmoid())
                        
                    prev_size = out_size
                        
                # 마지막 활성화 함수가 필요 없는 경우 제거
                activation_layers = (torch.nn.ReLU, torch.nn.GELU, torch.nn.Tanh, torch.nn.Sigmoid)
                if len(layers) > 1 and isinstance(layers[-1], activation_layers):
                    layers.pop()
                    
                new_model = torch.nn.Sequential(*layers)                

        return new_model.to(self.device)

    def _initialize_model_from_scratch(self, config, model_type):
        """
        모델 타입에 따라 적절한 모델 클래스를 import하고 초기화
        """
        try:
            if model_type == "bert":
                from transformers import BertForPreTraining
                return BertForPreTraining(config)
            elif model_type == "llama" or "llama" in model_type:
                from transformers import LlamaForCausalLM
                return LlamaForCausalLM(config)
            elif model_type == "mistral":
                from transformers import MistralForCausalLM
                return MistralForCausalLM(config)
            elif "gpt" in model_type:
                from transformers import GPT2LMHeadModel
                return GPT2LMHeadModel(config)
            elif model_type == "t5":
                from transformers import T5ForConditionalGeneration
                return T5ForConditionalGeneration(config)
            elif model_type == "roberta":
                from transformers import RobertaForMaskedLM
                return RobertaForMaskedLM(config)
            elif model_type == "bart":
                from transformers import BartForConditionalGeneration
                return BartForConditionalGeneration(config)
            else:
                # 일반적인 모델 클래스 가져오기 시도
                from transformers import AutoModel
                return AutoModel.from_config(config)
        except Exception as e:
            print(f"모델 초기화 중 오류 발생: {e}")
            raise

    def _print_attributes(self, model):
        for attr in dir(model):
            if not attr.startswith('_'):
                try:
                    attr_value = getattr(model, attr)
                    logger.info(f"  - {attr}: {type(attr_value).__name__}")
                    
                    # 중요한 속성의 경우 더 자세한 정보 출력
                    if isinstance(attr_value, torch.nn.Module):
                        logger.info(f"    Module with {sum(p.numel() for p in attr_value.parameters())} parameters")
                except Exception as e:
                    logger.info(f"  - {attr}: Error accessing - {str(e)}")

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
                    # prune.remove(module, 'weight')  # Make pruning permanent
                    pruned_count += 1

            self.rebuild_model()
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