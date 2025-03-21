"""
sLLM을 만드는 과정

1. Pretrained LLM 불러오기: Hugging Face의 transformers 라이브러리를 활용하여 사전 학습된 모델을 로드
2. Pruning 기법 적용: 가중치 가지치기(weight pruning), 구조적 가지치기(structural pruning) 등을 활용하여 모델의 크기 축소
3. Quantization 적용: INT8 또는 FP16으로 변환하여 연산 속도를 높이고 메모리 사용량 절감
4. Sparsity-aware Fine-tuning: 가지치기 후 성능 저하를 보완하기 위한 추가 학습
5. 추론 성능 테스트: 모델 크기 비교, 속도 테스트, 정확도 평가

torch.nn.utils.prune.l1_unstructured() 함수
- L1 정규화 기반 가지치기(L1 Unstructured Pruning) 를 수행하는 PyTorch의 유틸리티 함수

1. L1 Unstructured Pruning이란?
Unstructured Pruning: 모델의 특정 뉴런이나 채널이 아닌, 개별 가중치(weight)를 기준으로 가지치기를 수행하는 방식
L1 기반: 절대값이 작은 가중치를 제거하는 방식 → 즉, 절대값이 작은 가중치부터 선택적으로 0으로 만들면서 모델을 경량화

2. 함수 설명
prune.l1_unstructured(module, name='weight', amount=0.3)

매개변수
module: 가지치기를 적용할 레이어 (예: torch.nn.Linear)
name: 가지치기를 적용할 파라미터 (대부분 weight)
amount: 가지치기 비율 (예: 0.3 → 전체 가중치의 30%를 0으로 설정)

작동 방식
module.weight에서 절대값이 가장 작은 30%의 가중치를 0으로 설정
하지만 실제로 module.weight가 변하는 것이 아니라, module.weight_mask라는 새로운 마스크 텐서가 생성됨
추론 시 마스크가 적용되어 가중치가 0이 되지만, 원래의 가중치는 module.weight_orig에 남아 있음

3. 가지치기 후 가중치 확인하기 (선택사항)
가지치기 전후의 가중치 비교 코드
'''
import torch.nn.utils.prune as prune

# 가지치기 전 가중치 확인
print("Before pruning:\n", model.transformer.h[0].mlp.c_fc.weight)

# 가지치기 적용
prune.l1_unstructured(model.transformer.h[0].mlp.c_fc, name='weight', amount=0.3)

# 가지치기 후 가중치 확인
print("After pruning:\n", model.transformer.h[0].mlp.c_fc.weight)
'''
위 코드를 실행하면 가지치기 후 일부 가중치가 0으로 설정된 것을 확인할 수 있음

4. Permanent Pruning (영구적 가지치기)
기본적으로 PyTorch의 prune 모듈은 가중치에 마스크만 적용하는 방식이라서, 원래 가중치 정보가 남아 있음
즉, module.weight_orig가 원래 가중치이고, module.weight는 weight_mask가 적용된 값
이걸 완전히 제거하려면 prune.remove()를 호출해야 함

prune.remove(model.transformer.h[0].mlp.c_fc, 'weight')

이렇게 하면 module.weight_orig이 삭제되고 module.weight만 남아서 모델이 더 가벼워짐

5. 가지치기 후 추가적인 최적화 (선택사항)
가지치기 후 성능을 유지하려면 파인튜닝을 수행하는 것이 좋음
가지치기된 모델을 다시 학습시키면, 살아남은 가중치가 중요도를 더 반영하여 최적화 됨

'''
# 학습 루프
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(3):
    for batch in dataloader:
        inputs, labels = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
'''

6. 다른 Pruning 기법과 비교
----------------------------------------------------------------
   Pruning 기법	     |             설명
--------------------|-------------------------------------------
l1_unstructured	    | 절대값이 작은 개별 가중치를 0으로 설정           
random_unstructured	| 랜덤하게 가중치를 0으로 설정                   
structured	        | 특정 뉴런이나 채널 단위로 가지치기                 
global_unstructured	| 모델 전체에서 가장 작은 가중치를 선택적으로 가지치기
----------------------------------------------------------------
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
def apply_pruning(model, amount=0.3):
    """ 모델의 선형 계층에 가지치기 적용 """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # 가중치 제거

apply_pruning(model, amount=0.3)  # 30% 가지치기 적용

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
