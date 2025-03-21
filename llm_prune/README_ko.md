# LLM Pruning Guide

## 🔍 LLM에서 주로 사용되는 Pruning 기법

| Pruning 방식           | 적용 대상                | 장점                                   | 단점                                    |
|-----------------------|----------------------|--------------------------------------|----------------------------------------|
| **Unstructured Pruning** | 선형 레이어 (MLP)      | 모델 크기 줄이기 용이, 성능 유지 가능 | 속도 최적화 효과 적음 (희소성만 증가)  |
| **Structured Pruning**  | 선형 레이어 (MLP) 및 어텐션 | 연산량 감소, 속도 향상 가능         | 성능 저하 가능성 높음, 추가 학습 필요  |

---

## 1️⃣ Unstructured Pruning (비구조적 가지치기)

### 📌 적용 대상
- `Linear` 레이어의 개별 가중치

### 📌 방법
- L1/L2 노름이 작은 개별 가중치를 제거 (ex: `l1_unstructured()`)

### 📌 장점
- 성능 유지가 상대적으로 용이

### 📌 단점
- 가중치 희소성은 증가하지만 연산량 감소 효과는 크지 않음

### 📌 LLM 적용 예시
- OPT, GPT, BERT 모델의 선형 레이어 (`torch.nn.Linear`)에 적용
- `weight_mask`만 0으로 설정되므로 희소성이 증가하지만, 계산량은 그대로 유지됨

---

## 2️⃣ Structured Pruning (구조적 가지치기)

### 📌 적용 대상
- `Linear` 레이어 (MLP 부분)
- `Attention`의 Heads (어텐션 헤드)

### 📌 방법
- L2 norm이 작은 뉴런(MLP)이나 어텐션 헤드를 제거 (ex: `ln_structured()`)

### 📌 장점
- 실제 계산량이 줄어 속도 최적화 효과 큼

### 📌 단점
- 성능 저하 가능성이 있어 추가 학습 필요

### 📌 2-1. 선형(MLP) 레이어 가지치기
- `Transformer` 모델의 `FFN (Feed-Forward Network)` 부분은 보통 큰 비중을 차지함
- 특정 뉴런을 제거하면 모델 크기를 줄일 수 있음
- `dim=1`로 입력 뉴런 단위 pruning → 연산량 감소

```python
prune.ln_structured(model.linear, name='weight', amount=0.3, n=2, dim=1)
```

### 📌 2-2. 어텐션 헤드(Attention Head) 가지치기
- `Multi-Head Attention`의 일부 헤드를 제거하는 방식
- 연구 결과에 따르면, 일부 어텐션 헤드는 기여도가 낮아 제거 가능
- `self-attention`에서 특정 Head의 전체 weight를 pruning하여 연산량 감소

```python
prune.ln_structured(model.self_attn, name='weight', amount=0.3, n=2, dim=0)  # Head 단위 pruning
```

- `dim=0`으로 지정하여 어텐션 헤드 전체를 제거

---

## 🔬 실제 연구 및 적용 사례

### 📖 1. Movement Pruning (DiffPruning)
- 논문: ["Movement Pruning: Adaptive Sparsity by Fine-Tuning"](https://arxiv.org/abs/2005.07683)
- **핵심 아이디어**: 중요도가 낮은 가중치를 제거하며 성능 유지
- **적용 방식**: Unstructured Pruning + Knowledge Distillation

### 📖 2. Pruning Attention Heads
- 논문: ["Are Sixteen Heads Really Better than One?"](https://arxiv.org/abs/1905.10650)
- **핵심 아이디어**: 일부 어텐션 헤드는 거의 기여하지 않음 → 제거 가능
- **적용 방식**: 중요도가 낮은 어텐션 헤드를 Structured Pruning으로 삭제

### 📖 3. Block-wise Pruning
- 논문: ["Block Pruning for Faster Transformers"](https://arxiv.org/abs/2203.07890)
- **핵심 아이디어**: 모델을 블록 단위로 나누고 가지치기
- **적용 방식**: MLP 및 Attention 블록을 개별적으로 Pruning

---

## 🔎 결론: LLM에서는 어디에 Pruning을 적용할까?

### ✅ 주로 사용하는 방법

1. **MLP (선형 레이어) Pruning**
   - 구조적 가지치기 (`ln_structured()`)를 적용해 연산량 감소

2. **Attention Head Pruning**
   - 성능에 영향이 적은 어텐션 헤드를 제거

### 🚀 추가적으로 고려할 점

- **작은 모델 (GPT-2, BERT 등)**: 선형 레이어만 가지치기하는 경우 많음
- **큰 모델 (GPT-3, OPT 등)**: 어텐션 + MLP 같이 가지치기하여 최적화

---

## 📌 참고
- PyTorch 공식 문서: [torch.nn.utils.prune](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.html)
- Hugging Face: [Pruning LLM Guide](https://huggingface.co/docs/transformers/pruning)

---

