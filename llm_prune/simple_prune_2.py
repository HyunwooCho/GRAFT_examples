"""
sLLM을 만드는 과정

1. Pretrained LLM 불러오기: Hugging Face의 transformers 라이브러리를 활용하여 사전 학습된 모델을 로드
2. Pruning 기법 적용: 가중치 가지치기(weight pruning), 구조적 가지치기(structural pruning) 등을 활용하여 모델의 크기 축소
3. Quantization 적용: INT8 또는 FP16으로 변환하여 연산 속도를 높이고 메모리 사용량 절감
4. Sparsity-aware Fine-tuning: 가지치기 후 성능 저하를 보완하기 위한 추가 학습
5. 추론 성능 테스트: 모델 크기 비교, 속도 테스트, 정확도 평가

simple_prune.py 코드에 structured pruning을 추가
- ln_structured()를 활용해 뉴런 단위로 가지치기를 적용
- Unstructured Pruning + Structured Pruning 함께 적용

torch.nn.utils.prune.ln_structured()는 Structured Pruning(구조적 가지치기) 를 수행하는 PyTorch 유틸리티 함수
이 방식은 개별 가중치가 아닌, 전체 뉴런(Neuron) 또는 필터(Filter) 단위로 가지치기 수행

1. Structured Pruning의 핵심 개념
- Unstructured Pruning: 개별 가중치를 선택적으로 제거 (ex: l1_unstructured())
- Structured Pruning: 특정 뉴런이나 필터 전체를 제거하여 모델의 계산량을 줄임

ln_structured()의 핵심 동작
(1) 특정 축(dim)을 기준으로 뉴런(혹은 필터)의 Ln norm을 계산
(2) 가장 작은 amount 비율의 뉴런(필터)을 제거 --> 완전히 0이 되므로 모델 크기 감소 및 연산 속도 향상 가능

2. ln_structured() 함수 설명
'''
prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=1)
'''
매개변수
- module: 가지치기를 적용할 레이어 (torch.nn.Linear, torch.nn.Conv2d 등)
- name: 가지치기를 적용할 파라미터 (weight 또는 bias)
- amount: 가지치기할 뉴런(필터) 비율 (예: 0.3 → 전체 뉴런 중 30% 제거)
- n: L_n norm에서 n 값 (기본값: 2 → L2 norm)
  . n=1 → L1 norm(절대값 합)
  . n=2 → L2 norm (유클리드 거리)
  . n=∞ → 무한 norm (최대 절대값)
- dim: 가지치기를 적용할 축
  . dim=0 → 출력 뉴런(Feature) 단위로 가지치기
  . dim=1 → 입력 뉴런(Weight) 단위로 가지치기

3. 가지치기 예제
선형 계층에서 Structured Pruning 적용

'''
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# 예제 모델
linear = nn.Linear(10, 5)  # 입력 10개, 출력 5개

# 가지치기 전 가중치 확인
print("Before Pruning:\n", linear.weight)

# Structured Pruning (L2 노름 기준, 입력 뉴런 단위)
prune.ln_structured(linear, name='weight', amount=0.4, n=2, dim=1)

# 가지치기 후 가중치 확인
print("After Pruning:\n", linear.weight)
'''

결과:

- amount=0.4이므로 입력 뉴런의 40%가 제거됨 (즉, 10 * 40% = 4개의 뉴런이 제거됨)
- dim=1이므로 입력 뉴런 단위로 가지치기됨 (컬럼 단위로 0이 됨)

4. 가지치기 후 최적화 (영구 적용)
기본적으로 PyTorch의 가지치기는 마스크를 적용하는 방식
즉, module.weight_orig은 원래 가중치를 유지하며, module.weight_mask가 적용됨
이를 완전히 제거하려면 prune.remove()를 호출해야 함

'''
prune.remove(linear, 'weight')  # 구조적 가지치기 후 원본 가중치 삭제
'''
이렇게 하면 module.weight_orig이 삭제되고 module.weight만 남아 최적화

5. Unstructured Pruning vs Structured Pruning 비교
--------------------------------------------------------------------------------
방식	              | 제거 단위 / 속도 최적화 효과 / 모델 크기 축소 / 학습 후 성능 유지
---------------------|----------------------------------------------------------
Unstructured Pruning | 개별 가중치 / 낮음 / 크기만 감소 / 상대적으로 좋음
Structured Pruning   | 뉴런, 핕터 단위 / 높음 / 크기 & 연산량 감소 / 추가 학습 필요
--------------------------------------------------------------------------------
- Unstructured Pruning은 가중치를 랜덤하게 제거하므로 연산 속도 최적화 효과는 적음
- Structured Pruning은 전체 뉴런이나 필터를 제거하므로 속도 최적화 및 크기 감소 효과가 큼

6. L1 vs L2 Norm 선택 가이드
---------------------------------------------------------------------
Pruning 방법	   설명	                        사용 추천
---------------------------------------------------------------------
L1 Norm (n=1)	절대값의 합이 작은 뉴런 제거	     희소성(Sparsity) 극대화
L2 Norm (n=2)	유클리드 거리(L2 norm) 기반 제거	일반적인 구조적 가지치기
L∞ Norm (n=inf)	최대 절대값 기준 제거	           특정 중요 뉴런 유지 필요시
----------------------------------------------------------------------

"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils import prune

# 1. Pretrained LLM 불러오기
MODEL_NAME = "facebook/opt-1.3b"  # 원하는 모델로 변경 가능
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2. 가지치기 적용 함수
def apply_unstructured_pruning(model, amount=0.3):
    """ 모델의 선형 계층에 Unstructured Pruning 적용 """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # 가중치 제거

def apply_structured_pruning(model, amount=0.3):
    """ 모델의 선형 계층에 Structured Pruning 적용 (전체 뉴런 단위) """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=1)
            prune.remove(module, 'weight')  # 가중치 제거

# 가지치기 적용
apply_unstructured_pruning(model, amount=0.3)  # 30% Unstructured Pruning 적용
apply_structured_pruning(model, amount=0.3)    # 30% Structured Pruning 적용

# 3. 모델 양자화 적용
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 4. 테스트: 추론 속도 및 성능 비교
def generate_text(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt = "Once upon a time,"
print("Generated Text:", generate_text(prompt, model, tokenizer))

# 5. 저장 및 로드
model.save_pretrained("pruned_sllm")
tokenizer.save_pretrained("pruned_sllm")
