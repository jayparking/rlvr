# GRPO Curriculum 연구 파이프라인

실행 진입점은 `python yj/train.py research ...` 하나입니다.
A100 80GB 8장 실행은 아래 **회사 서버: A100 80GB × 8장 실행 순서** 절을 먼저 참고하세요.
원본 `math_eval_a100.py`를 수정하지 않고 zero-shot CoT 프롬프트와 기본 평가 실행에 사용합니다.
연구 설정은 `yj/research.json`, 새 데이터·결과는 `yj/studies/grpo_v1/`에 저장합니다.

**현재 상태: 구현 및 CPU 검증, 실제 데이터 준비 완료. 연구 설계는 draft이며 GPU 본 학습·전체 평가는 미실행입니다.**
설계와 주 지표의 확정 여부는 workspace의 `protocol.lock.json`으로 판단합니다.
기존 `experiment.json`, `artifacts/seed42/`는 이전 V1/V2 설계의 기록입니다.

## 기존 조건과 충돌하는 부분 및 제안 기본안

| 항목 | 이전 구현 | 제안 기본안 |
|---|---|---|
| 시작 상태 | Base V1 / 외부 SimpleRL-Zoo V2 | S0=Base, S1/S2=동일 seed의 한 사전 RL 궤적에서 100/300 update |
| seed | 42 | 42, 43, 44 |
| 평가 | 공통 90문제 greedy | MATH500 avg@8 후보, greedy 별도 보조 |
| 손실 | beta=0.04, 일부 기본값 의존 | beta=0, BNPO 토큰 평균, scale_rewards=True |
| 잘림 마스킹 | TRL 0.16.1에 해당 설정 없음 | 로컬 손실에 completion 전체 마스킹 추가 |
| 프롬프트 | 학습에서 parse_question 생략 | 원본 평가기의 실제 프롬프트 생성문 공통 사용 |
| 실행 파일 | train.py | 같은 진입점 유지, 내부 연구 모듈만 분리 |

외부 V2를 S1/S2로 간주하면 같은 사전 RL 궤적의 체크포인트라는 조건을 만족하지 않습니다.
따라서 외부 V2는 새 주 비교에 포함하지 않는 안을 제안합니다.
사전 RL은 seed마다 별도 궤적을 만들고, 그 안에서 S1/S2를 추출합니다.
이 선택들과 주 지표는 `finalize`로 명시적으로 확정해야 GPU 학습이 가능합니다.
현재 workspace는 확정하지 않았습니다.

이전 `python yj/train.py train ...` 명령은 혼용 방지를 위해 새 경로를 안내하고 종료합니다.
기존 export/evaluate/summarize와 데이터는 보존합니다. 이전 결과는 새 통계에 섞지 않습니다.

## 데이터와 학습 흐름

```text
고정 revision의 공식 MATH train
 ├─ 본 학습 300문제: Easy 100 / Medium 100 / Hard 100
 ├─ 독립 validation 300문제: 각 난도 100
 └─ 별도 사전 RL 풀 1,200문제: 각 난도 400

seed마다:
 Base(S0) → 새 LoRA로 사전 RL → 100 update → 300 update
                                  ↓              ↓
                              Base에 병합     Base에 병합
                                  S1             S2

각 S0/S1/S2에서:
 새 LoRA + 새 optimizer + 새 scheduler
 ├─ Random 300 update
 └─ Curriculum 300 update
       ↓
  0 / 100 / 200 / 300 모델별 no-update probe + 공통 평가
```

난도는 Easy=Level 1–2, Medium=Level 3, Hard=Level 4–5입니다.
모든 seed·상태·학습 순서가 같은 본 학습 300문제를 사용합니다.
Random은 전체를 seed별 셔플하고, Curriculum은 같은 셔플에서 난도 순서를 안정적으로 정렬하여
Easy→Medium→Hard 및 각 구간 내부 셔플을 모두 보장합니다.

세 학습용 풀은 서로 겹치지 않습니다. 공식 MATH test 전체와 평가셋의 질문을
공백 제거 후 비교하여 학습 풀에서 제외합니다. 이는 의미상 유사 문제 탐지가 아닙니다.
난도를 알 수 없는 문제는 학습 표본에서 제외하지만 평가 중복 제외에는 사용합니다.
실제 준비 결과는 본 학습 300 / validation 300 / 사전 RL 1,200 / MATH500 500입니다.

사전 RL은 1,200문제 풀 전체를 seed별 셔플한 뒤, 기본 설정에서는 앞의 **300문제만** 학습합니다.
한 문제에 한 update이므로 100/300 update가 각각 S1/S2입니다.
풀 1,200개를 모두 학습했다는 의미가 아닙니다.
후보 시점은 `pre_rl_checkpoints`로 변경 가능하며 결과를 보기 전에 새 workspace에 확정해야 합니다.

사전 RL과 본 학습은 동일한 프롬프트, G, LR, 생성 길이, 손실 설정을 공유합니다.
S1/S2는 사전 RL LoRA를 Base에 병합한 전체 모델로 저장합니다.
본 실험은 이전 adapter나 optimizer 상태를 이어받지 않습니다.
같은 seed에서 새 LoRA 초기화 난수도 맞추며 초기 adapter hash와 빈 optimizer 상태를 기록합니다.
S1/S2의 동일 궤적 의존성은 state manifest와 run manifest에 남깁니다.

## 프롬프트·EOS·생성 조건

`research/runtime.py`는 원본 평가기의 실제 프롬프트 생성문을 AST로 읽어 실행합니다.
`parse_question → parse_ground_truth → construct_prompt` 순서와
zero-shot CoT, 공백, 참/거짓·예/아니오 후처리를 공통으로 적용합니다.
chat template을 추가하지 않습니다.

학습은 `add_special_tokens=False`를 사용하고, 기존 vLLM 텍스트 경로의
기본 인코딩과 토큰 ID가 동일한지 검사합니다. 자동 BOS/EOS 추가가 달라지면 실행을 거부합니다.
별도 평가 환경의 tokenizer 결과도 다시 비교합니다.

공통 stop 문자열:
`</s>`, `<|im_end|>`, `<|endoftext|>`, `\n\nQuestion:`.
원본 평가기의 Qwen 분기는 모델 경로에 `qwen2`가 있을 때
stop token ID 151645/151643을 사용합니다.
생성한 병합 경로에 `qwen2`를 유지하며 EOS ID를 검사합니다.
입력 길이가 초과하면 조용히 자르지 않고 실패합니다.

| 설정 | 학습 / probing | MATH500 avg@8 | Greedy |
|---|---:|---:|---:|
| 답변 수 | G=4 / k=8 | 8 | 1 |
| temperature | 0.8 | 0.8 | 0 |
| top_p | 0.95 | 0.95 | 1 |
| top_k | 비활성 | 비활성 | 비활성 |
| 최대 생성 토큰 | 512 | 2048 | 2048 |

학습과 평가 temperature는 설정에서 서로 다르게 정할 수 있습니다.
표는 현재 제안값이며 실험 전 고정합니다. 평가 샘플링 seed는 20261018로 공통입니다.

## 실제 학습 알고리즘과 손실

설치된 **TRL 0.16.1의 실제 `GRPOTrainer.compute_loss` 소스**를 확인했습니다.
정규화는 completion별 평균의 평균이 아니라 전체 유효 토큰 평균입니다.
이에 해당하는 `loss_type=bnpo`, `scale_rewards=True` 조합을 고정했습니다.

이 버전의 GRPOConfig에는 `loss_type`, `mask_truncated_completions` 필드가 없습니다.
지원하지 않는 옵션을 넘기거나 TRL을 조용히 업그레이드하지 않고,
해당 손실을 명시적인 로컬 GRPO 루프 `yj_grpo_v1_bnpo`로 구현했습니다.
CPU 테스트에서 원본 TRL 손실값 및 log-probability gradient와 수치 비교합니다.
`inspect-loss`는 버전·실제 함수 소스 hash·로컬 함수 hash를 출력합니다.

```text
정답 보상 r_i ∈ {0, 1}
A_i = (r_i - mean(r)) / (sample_std(r) + 1e-4)
ratio_it = exp(logp_it - stop_gradient(logp_it))

L = -sum(mask_it × min(ratio_it × A_i,
                     clip(ratio_it, 0.8, 1.2) × A_i)) / sum(mask_it)
```

- beta=0.0: KL 항과 reference model 없음.
- num_iterations=1: 같은 생성 그룹을 재사용해 여러 번 업데이트하지 않음.
- clipping epsilon_low=epsilon_high=0.2. 단일 iteration에서는 ratio=1이므로 clipping은 비활성.
- scale_rewards=True: 그룹 내 표본 표준편차(correction=1) 사용.
- mask_truncated_completions=True: 길이 제한으로 잘린 답변의 **모든 토큰 손실**을 0으로 만듦.
- 잘린 답변의 실제 정오답 보상은 그대로 저장함. 잘림을 보상 0으로 덮어쓰지 않음.
- advantage 정규화에는 잘린 답변을 포함한 전체 G개 보상을 사용함.
  따라서 잘린 답변은 직접 토큰 gradient는 없지만 다른 답변의 advantage에 간접 영향을 줄 수 있음.
- 전체가 잘리면 분모를 1로 보호하여 유한한 0 loss가 됨.
- all-correct/all-wrong 그룹의 reward-driven gradient가 0인지 테스트함.
  Adam의 과거 momentum이 있으면 gradient=0이어도 가중치 이동이 생길 수 있으므로
  이를 “optimizer update 없음”과 혼동하지 않음.

한 update는 **문제 1개 → 답변 G=4개 → 정오답 채점 → 그룹 advantage → backward →
optimizer.step 1회 → scheduler.step 1회**입니다.
gradient accumulation이나 여러 문제를 한 update로 묶는 동작은 없습니다.
AdamW LR=5e-6, weight_decay=0, constant scheduler, gradient clipping=1입니다.
생성과 학습의 dropout을 비활성화하고 CUDA 학습에서는 gradient checkpointing을 사용합니다.
run manifest와 매 step 로그에 설정·보상·잘림·유효 토큰·실제 update 수를 기록합니다.

## 단계별 평가와 no-update probing

학습은 300 update까지 연속 진행하고 100/200/300 adapter를 저장합니다.
**단계별 평가를 위해 학습을 중단했다가 optimizer를 다시 만드는 방식이 아닙니다.**
학습 완료 후 저장된 시점별 모델을 평가하므로 중간 평가가 학습 난수나 순서에 영향을 주지 않습니다.
step 0은 새 LoRA가 아직 변화를 주지 않은 시작 모델입니다.

| step | Curriculum 진행 | Random 진행 | 공통 평가 |
|---:|---|---|---|
| 0 | 학습 전 | 학습 전 | 본 학습 300문제 probe + MATH500 |
| 100 | Easy 완료 | 섞인 100문제 완료 | 동일 |
| 200 | Medium 완료 | 섞인 200문제 완료 | 동일 |
| 300 | Hard 완료 | 300문제 완료 | 동일 |

probing은 매 시점 **본 학습 300문제 전체 × k=8**을 생성합니다.
temperature, top_p, top_k, 최대 생성 길이, EOS/stop은 학습과 같습니다.
별도 추론 모델에서 실행하며 optimizer를 만들거나 학습 상태를 불러오지 않습니다.
HF 경로는 inference_mode와 파라미터 전후 hash·gradient 부재 검사를 수행합니다.
A100/vLLM 경로는 별도 추론 프로세스입니다.

문제마다 다음을 저장합니다.

- 정답 수 c, p=c/8, all-wrong / mixed / all-correct 분류.
- 예상 학습 그룹 유효 확률 `1-p^G-(1-p)^G` (G=4).
- 8개 표본의 모든 G개 부분집합을 이용한 U-statistic 보조 추정치.
- 원시 답변·정오답·종료 사유·잘림·채점 오류.
- 전체 문제의 그룹 비율과 평균 확률.

p는 k=8의 잡음이 큰 추정치입니다. 위 유효 확률은 iid 보상 가정이며
잘림 마스킹 이후의 실제 gradient 활성 확률까지 표현하지는 않습니다.
모델 내용 hash, 데이터, 프롬프트·tokenizer 계약, 생성 설정, seed, backend·버전으로 캐싱합니다.
완료 receipt와 결과 hash가 맞는 경우에만 재사용합니다.
동일한 S0 baseline은 seed·학습 순서 사이에서 재사용하며 S1/S2 baseline도 두 순서에서 공유합니다.

## 평가 지표·validation·보조 평가셋

주 지표 후보: **최종 step 300의 MATH500 avg@8**.

- avg@8: 각 문제의 8개 답변 정답 비율을 문제 전체에 평균. 전체 4,000답변의 정답률과 같음.
- pass@8: 각 문제에서 8개 중 하나라도 정답이면 1. 이를 500문제에 평균.
- greedy accuracy: temperature=0으로 **별도 생성한** 1개 답변의 정답률.

예: 8개 중 2개가 맞은 문제의 avg@8 기여도는 0.25, pass@8 기여도는 1입니다.
Greedy를 avg@8의 첫 답변으로 대체하지 않습니다.
중간 step 0/100/200는 변화 과정 분석용이며 최종 step 300을 주 비교로 사용합니다.

`diagnose`는 S0/S1/S2를 본 학습과 겹치지 않는 validation 300문제에서 k=8로 평가하고,
사전 RL 학습 로그의 all-wrong/mixed/all-correct 및 잘림 비율과 함께 저장합니다.
체크포인트 차이가 작아도 유의성 조건으로 후보를 자동 교체하지 않습니다.
변경이 필요하면 결과를 본 뒤 기존 확증 실험인 것처럼 이어가지 않고 새 설계·workspace로 분리합니다.

보조 평가셋은 `evaluation.auxiliary_sets`에 아래 형태로 추가할 수 있습니다.
각 JSONL 행은 `problem`과 `solution` 또는 `answer`를 포함해야 합니다.

```json
{"name":"independent_math","path":"datasets/independent_math.jsonl",
 "sha256":"파일의 SHA256","provenance":"제작 시점·출처",
 "contamination_risk_note":"사전학습 데이터에 포함될 위험이 낮다고 판단하는 근거"}
```

현재 검증된 저오염 보조셋을 임의로 선정하거나 포함하지 않았습니다.
사용자가 확보한 비공개·신규 제작 문제 등을 연결할 수 있는 기능만 제공합니다.
단순히 출처 설명이 있다는 이유로 저오염을 보장하지 않습니다.

## 통계와 오염 진단

`statistics`는 각 상태에서 Curriculum-Random 정답률 차이와
차이의 차이 S1−S0, S2−S0, S2−S1을 percentage point로 계산합니다.
먼저 seed 궤적을 재표집하고, 그 안의 모든 상태·학습 순서를 함께 유지합니다.
각 반복에서 공통 문제 인덱스를 함께 재표집하여 문제 pairing과 상태 간 의존성도 유지합니다.
공통 평가 문제를 여러 seed에 사용하므로 문제 draw는 seed 간에도 공유하는 paired cluster 방식입니다.

기본 5,000회, percentile 95% CI이며 seed별 accuracy·차이와 문제별 원시 점수를 함께 저장합니다.
단 세 개 seed는 학습 궤적이 세 개뿐이므로 CI가 불안정하거나 실제 불확실성을 과소평가할 수 있습니다.
문제 bootstrap 횟수를 늘려도 독립적인 학습 seed가 늘지는 않습니다.
유의성을 기준으로 체크포인트를 선택하지 않습니다. backend·버전이 섞인 주 비교도 거부합니다.

`contamination`은 MATH500 질문의 **공백 기준 단어 수 앞 60%**를 원문 그대로 입력하고
나머지 질문을 이어 쓰게 합니다. 수학 풀이용 CoT를 넣는 평가와는 목적이 다른 진단입니다.
S0/S1/S2를 같은 prefix와 설정으로 비교합니다.
경계 공백을 제거한 Exact Match, 공백 정규화 EM, 공백 토큰 LCS 기반 ROUGE-L F1을 저장합니다.
ROUGE-L은 자체 whitespace tokenization이며 다른 라이브러리의 stemming/tokenization 점수와 다를 수 있습니다.
복원 성능만으로 오염을 확정하거나 완전히 배제할 수 없습니다.

## 실행 방법

모든 명령은 `math_eval` 폴더에서 실행합니다.
Linux의 단일 BF16 CUDA GPU를 대상으로 하며, 본 학습은 CPU fallback을 허용하지 않습니다.
`--workspace`를 바꿀 때는 **research 다음, 하위 명령 앞**에 둡니다.

학습 환경은 기존 A100 평가 환경과 분리합니다.

```bash
python3.11 -m venv yj/.venv-train
yj/.venv-train/bin/python -m pip install -r yj/requirements-train.txt
source yj/.venv-train/bin/activate

# GPU 사용 없음: 제안 설정·충돌·비용 확인
python yj/train.py research plan
python yj/train.py research inspect-loss

# 새 서버에서 데이터 준비. 현재 작업 공간에는 이미 준비되어 있음.
python yj/train.py research prepare
# 소스 데이터 캐시가 있을 때만:
# python yj/train.py research prepare --offline
```

기존 데이터가 있으면 prepare는 덮어쓰지 않습니다.
설정 수정 시 `yj/research.json`을 수정하고 다른 workspace에 prepare합니다.

```bash
python yj/train.py research --workspace yj/studies/grpo_v2 prepare --config yj/research.json
```

설계 충돌과 주 지표를 검토한 후 아래 명령으로 확정합니다.
이 명령 자체는 GPU를 사용하지 않으며, 이번 작업에서는 실행하지 않았습니다.

```bash
python yj/train.py research finalize --primary math500.avg@8 --accept-design
python yj/train.py research pretrain --seed 42 --dry-run
python yj/train.py research train --seed 42 --state S0 --mode curriculum --dry-run
```

설정·데이터·연구 코드 hash를 잠그므로 확정 후 결과를 본 상태에서 코드를 고쳐 같은 실험으로 계속할 수 없습니다.
아래부터는 사용자가 GPU 서버에서 명시적으로 실행하는 장시간 작업입니다.
`EVAL_PY`는 기존 `math_eval_a100.py`가 실행되는 Python의 실제 경로로 바꿉니다.

```bash
export CUDA_VISIBLE_DEVICES=0
EVAL_PY=/absolute/path/to/a100-eval-env/bin/python

# seed별 사전 RL과 독립 validation
for seed in 42 43 44; do
  python yj/train.py research pretrain --seed "$seed"
  python yj/train.py research diagnose --seed "$seed" --python "$EVAL_PY"
done

# 3 states × 2 orders × 3 seeds = 18개 본 학습
for seed in 42 43 44; do
  for state in S0 S1 S2; do
    for mode in random curriculum; do
      python yj/train.py research train --seed "$seed" --state "$state" --mode "$mode"
      python yj/train.py research probe --seed "$seed" --state "$state" --mode "$mode" --python "$EVAL_PY"
      python yj/train.py research evaluate --seed "$seed" --state "$state" --mode "$mode" --python "$EVAL_PY"
    done
  done
done

python yj/train.py research statistics --step 300

# 추가 진단; 이 명령도 전체 MATH500 생성 작업
for seed in 42 43 44; do
  python yj/train.py research contamination --seed "$seed" --backend hf
done
```

probe/evaluate 기본 시점은 0/100/200/300 모두입니다. 일부만 실행하려면 `--steps 100`처럼 지정합니다.
GPU 학습이 실패한 run은 자동으로 덮어쓰거나 다른 상태로 재개하지 않습니다.
완료한 run의 probing/evaluation은 다시 실행할 수 있으며 완료 캐시를 재사용합니다.
원본 adapter가 있으면 `export --run-dir ... --step 100`으로 병합 모델을 다시 만들 수 있습니다.

### vLLM 직접 LoRA 평가와 디스크 정리

기본 경로는 병합 모델을 **원본 A100 evaluator**로 평가합니다.
직접 LoRA 경로는 원본에 없는 기능이므로 별도 vLLM worker에서 실행합니다.
`--python`은 A100과 직접 vLLM 경로 모두에서 평가 환경을 지정합니다.
worker는 요청별로 종료하여 GPU 자원을 반환하며, tokenizer 일치도 다시 검사합니다.

```bash
# 먼저 작은 호환성 비교: 이 역시 GPU 실행이므로 수동으로 실행
python yj/train.py research compare-backends \
  --run-dir yj/studies/grpo_v1/runs/S0_seed42_random --step 100 \
  --limit 10 --python "$EVAL_PY"

# 선택적으로 직접 adapter 평가
python yj/train.py research evaluate --seed 42 --state S0 --mode random \
  --steps 100 --backend vllm-lora --python "$EVAL_PY"

# 삭제 전 대상 확인: 기본 dry-run
python yj/train.py research cleanup \
  --run-dir yj/studies/grpo_v1/runs/S0_seed42_random --step 100

# 위 경로를 실제 삭제하기로 명시적으로 결정한 경우에만 --delete 추가
```

비교 결과는 greedy 텍스트 일치율·정오답 일치율로 저장합니다.
직접 LoRA와 병합 BF16은 수치적 차이가 있을 수 있으므로 동등성을 미리 가정하지 않습니다.
현재 GPU에서 두 경로의 호환성을 실측하지 않았습니다.
공식 비교는 모든 상태·seed·순서에 같은 backend와 버전을 사용해야 합니다.
주 통계는 확정된 evaluation.backend와 다른 backend 결과도 거부합니다.
위 직접 LoRA 명령은 선택적 점검 예시이며, 이를 주 평가로 사용할 경우 새 workspace 준비·확정 전에 backend를 변경합니다.

cleanup은 해당 시점의 probing과 모든 평가셋 avg@8·greedy 완료 receipt,
모델 identity, 원본 adapter와 Base 존재를 확인한 뒤 파생 본 학습 병합 모델만 대상으로 합니다.
기본은 dry-run이며 `--delete`가 있어야 삭제합니다.
데이터·로그·캐시·원본 LoRA·optimizer 체크포인트·Base·S1/S2 시작 모델은 삭제하지 않습니다.

## 기본안 비용 예상

| 항목 | 예상 |
|---|---:|
| 사전 RL update | 900 |
| 본 실험 update | 5,400 |
| 학습 생성 답변 | 25,200 |
| 본 실험 checkpoint 평가 요청 | 72 |
| baseline 중복 제외 모델 상태 | 61 |
| probing 답변: 캐시 전 / baseline 재사용 후 | 172,800 / 146,400 |
| MATH500 avg@8+greedy 답변: 캐시 전 / 재사용 후 | 324,000 / 274,500 |
| 사전 RL validation 답변 | 16,800 |
| 질문 복원 진단 답변 | 3,500 |
| 전체 병합 가중치를 보관할 때 | 약 183 GB |
| adapter + Adam 상태 | 약 3.6 GB |

가중치 1.5B×2bytes, LoRA 5M×4bytes 등 설정의 근사치입니다.
다운로드 캐시, tokenizer, 데이터·원시 생성 로그, 병합 임시 공간, CPU/GPU 메모리는 별도입니다.
실제 모델 parameter 수와 dtype에 따라 달라집니다.
`plan`은 생성 토큰 수 상한도 출력하지만 측정 처리량 없이 GPU 시간/금액을 단정하지 않습니다.
평가 완료 병합 모델을 순서대로 정리하거나 검증된 직접 LoRA 경로를 쓰면 보관량을 줄일 수 있습니다.

## 파일·결과 위치와 검증

| 파일 | 역할 |
|---|---|
| `train.py` | 기존 진입점, research 명령 연결 |
| `research.json` | 검토·확정할 연구 설정 |
| `research/protocol.py` | 데이터 분리, 셔플, hash, 확정, 비용 |
| `research/runtime.py` | 원본 프롬프트 계약, 생성·마스킹·GRPO 손실 |
| `research/workflow.py` | 새 LoRA·optimizer, 학습·저장·병합, 계보 |
| `research/inference.py` | no-update probe, 평가, 복원 진단, 캐시·정리 |
| `research/metrics.py` | avg/pass, 그룹 진단, paired bootstrap |
| `research/cli.py` | 실행 명령과 충돌 보고 |
| `test_research.py` | 연구 단위·CPU 통합 테스트 |

workspace 주요 결과:
`data/manifest.json`, `protocol.lock.json`, `states/seed42/S1.json`,
`pre_rl/seed42/run.json`, `runs/S0_seed42_random/training.jsonl`,
`runs/.../results/*.json`, `cache/<hash>/result.json`,
`diagnostics/validation_seed42.json`, `diagnostics/contamination_seed42.json`,
`statistics/step300/{bootstrap.json,raw_problem_scores.json,seed_results.csv}`.

CPU 검증 명령:

```bash
python -m unittest discover -s yj -p 'test_research.py' -v
YJ_RESEARCH_INTEGRATION=1 TOKENIZERS_PARALLELISM=false \
  python -m unittest discover -s yj -p 'test_research.py' -v
```

2026-10-18 CPU 실행 결과: 새 연구 테스트 15개와 기존 회귀 테스트 10개, 총 **25개 통과**, skip/failure/error 0, 6.978초.
Python 3.13 임시 검증 환경에서 실행했습니다. 전체 회귀 테스트를 켜는 명령은 다음과 같습니다.

```bash
YJ_RESEARCH_INTEGRATION=1 YJ_TRAINING_INTEGRATION=1 TOKENIZERS_PARALLELISM=false \
  python -m unittest discover -s yj -p 'test_*.py' -v
```

검증 기록은 `studies/grpo_v1/verification/cpu-verification.json`에 저장했습니다.
통합 테스트는 작은 무작위 Qwen2 모델을 로컬에서 만들어 실행하며 모델 다운로드나 GPU가 필요 없습니다.
업데이트 테스트의 학습 신호에는 합성 교대 보상을 사용하므로 실제 수학 성능 향상을 검증한 것은 아닙니다.
실제 G=4의 1회 update, LoRA 변화, checkpoint 저장·병합·새 초기화,
TRL 손실/gradient 일치, uniform 보상 gradient=0, 잘림 마스킹,
k=8 no-update 생성·캐시, 셔플·분리, 통계, 안전한 파생 모델 정리를 검사합니다.

아직 확인하지 못한 항목:
실제 1.5B tokenizer의 전체 데이터 길이와 양쪽 환경 인코딩,
A100 BF16 메모리·처리량·장시간 안정성, 실제 학습 성능 향상,
vLLM 직접 LoRA/병합 실측 호환성, MATH500 실측 점수·CI 및 오염 진단 결과.
이 항목들을 검증하기 전에는 실험이 잘 학습되거나 Curriculum이 우수하다고 결론 내릴 수 없습니다.


## 추가 재검증: 2026-10-18

이전 25개 테스트 통과 이후 경계 조건을 추가하여 총 31개로 확장했습니다.
작은 CPU 테스트 모델의 초기화 seed와 PYTHONHASHSEED를 42/43/44로 바꿔
독립 프로세스에서 세 번 실행했고 **31 × 3 = 93개 실행 모두 통과**했습니다.
각 회차는 6.905초 / 6.984초 / 6.650초였으며 skip·failure·error는 없습니다.
이는 실제 1.5B 연구 모델을 세 seed로 학습한 결과가 아닙니다.

재검증 중 다음 문제를 수정하거나 보호 검사를 보강했습니다.

- 보조 평가셋에 기존 idx가 있으면 원본 loader가 재정렬하는 문제: 임시 평가 입력의 idx를 연속 번호로 지정.
- 실제 답변이 7개인 결과를 avg@8 통계로 읽을 수 있는 문제: 설정 k와 실제 개수를 비교.
- CPU float32와 CUDA BF16 결과의 HF 캐시 충돌: 장치·dtype·GPU 이름·CUDA runtime을 캐시 조건에 포함.
- 음수 clipping, 0 생성 길이, 소수 G, NaN, 잘못된 복원 단위 등: 실행 전 설정 검증 추가.
- 일부 평가 의존 파일이 protocol 코드 hash에서 빠진 부분: evaluate/data_loader/executor/trajectory/model_utils 포함.
- 병합 모델 상위 폴더가 symlink일 때 삭제 대상 경계가 바뀌는 경우: 삭제 거부.
- CPU 병합 시 PEFT가 CUDA를 자동 선택하는 동작: adapter를 CPU에 명시적으로 로드.

실제 데이터를 임시 폴더에 두 번 재생성했고 두 결과와 기존 데이터의 파일 hash가 모두 같았습니다.
본 학습 300 + 사전 RL 1,200 + validation 300 + MATH500 500 = **2,300문제 전체**에서
프롬프트 문자열과 기준 정답을 원본 평가기 방식과 비교해 일치를 확인했습니다.
실제 Qwen tokenizer의 토큰 ID와 GPU/vLLM 검증은 여전히 별도 실행이 필요합니다.

사전 RL → S1/S2 병합 → 새 LoRA/optimizer → 본 학습 저장까지 연결하는 CPU 테스트도 추가했습니다.
이 테스트는 CUDA 허용 검사만 mock하고 작은 모델 및 단축된 1/3 checkpoint를 사용합니다.
모든 모델과 optimizer는 실제로 CPU에서 작동합니다.

원본 math_eval_a100.py는 그대로이며 연구 설계도 draft를 유지했습니다.
상세 보고서·실패 재현 로그·최종 반복 로그는
`studies/grpo_v1/verification/recheck-20261018/`에 있습니다.

```bash
for seed in 42 43 44; do
  YJ_RESEARCH_INTEGRATION=1 YJ_TRAINING_INTEGRATION=1 \
  YJ_TEST_MODEL_SEED="$seed" PYTHONHASHSEED="$seed" TOKENIZERS_PARALLELISM=false \
    python -m unittest discover -s yj -p 'test_*.py' -v
done
```

## 회사 서버: A100 80GB × 8장 실행 순서

아래는 **한 Linux 서버의 할당된 A100 80GB 8장**, 같은 로컬 디스크, 같은 사용자 계정을 기준으로 합니다.
여러 서버/NFS/MIG 환경용 분산 실행기가 아닙니다.
GPU 한 장에 독립 실험 하나를 배정합니다. `torchrun`/DDP로 8장을 한 학습에 묶지 않으므로
기존 G=4, 문제당 optimizer step 1회, 300 update 조건을 유지합니다.

| 순서 | 작업 | 전체 작업 수 | 동시 GPU 작업 |
|---|---|---:|---:|
| 1 | 데이터 준비·서버 환경·tokenizer 점검·설계 확정 | 1 | 학습·생성 없음 |
| 2 | 사전 RL seed 42/43/44, S1/S2 병합 | 3 | 최대 3 |
| 3 | S0/S1/S2 독립 validation 진단 | seed별 3 | 최대 3 |
| 4 | S0/S1/S2 × Random/Curriculum × 3 seeds | 18 | 최대 8 |
| 5 | 단계별 결과를 이용한 최종 paired bootstrap | 1 | CPU |
| 6 | 선택적 질문 복원 진단 | seed별 3 | 최대 3 |

본 실험 작업 하나는 `train → probe → evaluate` 순서로 진행합니다.
같은 GPU에서 이 세 명령을 겹쳐 실행하지 않습니다.
빈 GPU에 다음 실험을 배정하므로 18개를 수동으로 8/8/2개로 나눌 필요가 없습니다.
`probe/evaluate`는 각 작업 안에서 0/100/200/300 시점을 처리합니다.
동일 baseline 캐시를 기다리는 동안 일부 GPU가 잠시 쉬는 것은 정상입니다.

이번 병렬화 보강:
동일 run의 학습·병합·평가·정리 사이 잠금, 동일 probing/evaluation 요청의 단일 생성,
Base 다운로드/등록 잠금, 고유 임시 파일을 통한 JSON 원자적 저장을 추가했습니다.
실제 GPU UUID로 장치를 지정하고, 사용 중인 GPU·중복 GPU 별칭·할당 범위를 벗어난 GPU를 거부합니다.
한 launcher 안에서 GPU별 작업은 순차 실행됩니다.
오류가 나면 새 작업 배정과 후속 단계를 중단하며, 다른 GPU에서 이미 실행 중인 명령은 완료될 수 있습니다.
자동으로 원본 checkpoint나 로그를 삭제하지 않습니다.

### 1. 파일과 Python 환경 준비

회사 서버에는 `math_eval` 폴더 전체를 옮깁니다.
`math_eval_a100.py`, parser/grader 등 기존 평가 의존 파일, `data/math500/test.jsonl`,
`yj/research/`, `yj/train.py`, `yj/run_a100x8.sh`, 설정·requirements 파일이 필요합니다.
새 실행 파일 중 Git에 아직 추가되지 않은 파일도 있으므로 단순 git clone만으로 모두 전달됐다고 가정하지 않습니다.
실제 학습 결과에는 절대 경로가 기록되므로 회사 서버에서 새 workspace로 시작합니다.

```bash
cd /회사서버/프로젝트/math_eval

python3.11 -m venv yj/.venv-train
yj/.venv-train/bin/python -m pip install -r yj/requirements-train.txt

export TRAIN_PY="$PWD/yj/.venv-train/bin/python"
export EVAL_PY="/기존_A100_평가환경의_절대경로/bin/python"
export YJ_WORKSPACE="yj/studies/a100x8_v1"
export YJ_GPU_IDS="0 1 2 3 4 5 6 7"
```

`EVAL_PY`는 기존 `math_eval_a100.py`와 vLLM이 실행되는 환경입니다.
학습용 TRL/torch 환경과 분리합니다. 이 스크립트가 평가 환경을 자동 설치하거나 덮어쓰지 않습니다.
`YJ_GPU_IDS`의 숫자는 `nvidia-smi`에 보이는 GPU index입니다. 실행 시 UUID로 변환합니다.
기존 `CUDA_VISIBLE_DEVICES`가 있으면 그 할당 범위를 존중합니다.
이전에 한 장만 지정한 셸이면 회사에서 실제 할당받은 장치 범위를 확인한 뒤 환경을 맞춥니다.
Slurm/컨테이너가 정한 범위를 임의로 넓히지 않습니다.

### 2. 데이터·환경 점검 후 설계 확정

```bash
"$TRAIN_PY" yj/train.py research --workspace "$YJ_WORKSPACE" prepare

"$TRAIN_PY" yj/train.py research --workspace "$YJ_WORKSPACE" server-check \
  --gpus 0 1 2 3 4 5 6 7 --python "$EVAL_PY" --download-base

# 제안 설계와 MATH500 avg@8을 채택하기로 확정한 후
"$TRAIN_PY" yj/train.py research --workspace "$YJ_WORKSPACE" finalize \
  --primary math500.avg@8 --accept-design

# 실제 GPU 작업 없이 18개 작업과 명령 확인
bash yj/run_a100x8.sh plan
```

이미 준비한 동일 workspace에는 prepare를 반복하지 않습니다.
`server-check --download-base`는 pinned Base를 다운로드하고 다음을 확인합니다.

- 실제 A100 이름·80GB 메모리·장치 UUID, GPU별 CUDA 인식과 BF16 지원.
- 학습 패키지 버전·실제 TRL 손실 소스, 평가 환경의 원본 evaluator import.
- 본 학습·사전 RL 풀·validation·MATH500 및 보조셋 전체의 실제 tokenizer 입력 길이.
- 양쪽 Python 환경의 프롬프트/token ID/EOS 일치.
- 디스크 여유 공간과 호스트 RAM, 기존 dry-run 비용 추정.

이 명령은 학습이나 답변 생성을 하지 않습니다.
결과는 `$YJ_WORKSPACE/verification/server-check.json`에 저장합니다.
환경·코드·설정과 일치하는 성공 기록 및 tokenizer 검사 없이 launcher의 `--execute`는 진행되지 않습니다.
설정 변경이 필요하면 finalize 전에 새 설정으로 새 workspace를 준비합니다.

전체 병합 가중치 보관 추정은 약 183GB이고 adapter/Adam 약 3.6GB, 다운로드·데이터·로그·임시 공간은 별도입니다.
8개 CPU 병합이 겹칠 수 있으므로 서버의 RAM과 로컬 SSD 여유도 확인합니다.
실제 GPU 최고 메모리와 벽시계 실행시간은 아직 측정하지 않았습니다.

### 3. 첫 seed 하나를 확인하고 전체 실행

먼저 seed 42 사전 RL 하나를 GPU 0에서 실행하는 명령입니다.
이는 작은 smoke test가 아니라 **실제 300 update 사전 RL 한 실행**입니다.

```bash
"$TRAIN_PY" yj/train.py research --workspace "$YJ_WORKSPACE" launch \
  --phase pretrain --gpus 0 --limit 1 --python "$EVAL_PY" --execute
```

이 실행이 완료되면 아래 한 명령으로 나머지 사전 RL부터 통계까지 순서대로 진행할 수 있습니다.

```bash
bash yj/run_a100x8.sh all --execute --resume-completed
```

회사 정책에 맞는 tmux 또는 배치 작업 안에서 실행하면 SSH 연결 종료와 분리해서 관리할 수 있습니다.
위 `all`은 다음 네 단계를 **앞 단계 성공 후** 실행합니다.

```bash
bash yj/run_a100x8.sh pretrain --execute --resume-completed
bash yj/run_a100x8.sh diagnose --execute
bash yj/run_a100x8.sh main --execute --resume-completed
"$TRAIN_PY" yj/train.py research --workspace "$YJ_WORKSPACE" statistics --step 300
```

단계마다 직접 상태를 확인하려면 위 명령을 하나씩 실행하면 됩니다.
`all` 명령과 개별 명령을 동시에 실행하지 않습니다.
`--execute`를 빼면 계획만 출력하고 GPU 작업을 시작하지 않습니다.

질문 복원 진단은 `all`에 포함하지 않았으며, 필요할 때 별도로 실행합니다.

```bash
bash yj/run_a100x8.sh contamination --execute
```

### 4. 진행 확인·중단 후 재실행

```bash
nvidia-smi
ls "$YJ_WORKSPACE/launches"
```

각 실행 폴더의 `plan.json`, `status.json`, `<job>_<command번호>.log`에서
GPU UUID, 실행 명령, 완료 개수, 종료 코드를 확인합니다.
본 실험 로그의 command 0/1/2는 보통 train/probe/evaluate입니다.
학습을 건너뛴 재실행은 명령 번호가 달라지므로 `plan.json`을 기준으로 확인합니다.

`--resume-completed`는 다음 동작만 수행합니다.

- 완료된 학습의 프로토콜·step 수·Base·원본 adapter hash를 검증하고 학습을 건너뜀.
- 완료된 사전 RL의 병합/상태 등록이 중단됐다면 원본 adapter에서 복구.
- probing/evaluation은 완료 캐시를 재사용하고 미완료 생성은 다시 실행.
- **실패한 학습을 optimizer checkpoint부터 자동 재개하지 않음.**
  `failed/training` 상태의 run은 명시적으로 거부하므로 원인을 확인해야 함.

실험이 중단됐다고 데이터·run 폴더를 자동 삭제하지 않습니다.
일반적인 Ctrl+C/SIGTERM에는 launcher가 소유한 자식 process group을 종료합니다.
회사 작업 관리자가 강제 종료한 경우에는 재실행 전에 `nvidia-smi`와 작업 상태를 확인합니다.
완료 병합 모델 삭제는 기존 `cleanup --delete`를 별도로 명시한 경우에만 가능합니다.

이번 로컬 검증은 CPU에서 수행했습니다. 실제 A100 8장 위의 학습 안정성·vLLM 결과·메모리 사용량은
회사 서버의 server-check와 첫 실험으로 확인해야 합니다.


A100 8장 실행기 로컬 재검증 결과: CPU 테스트 40개를 작은 모델 초기화 seed 42/43/44로
세 번 실행해 **120개 실행 모두 통과**했습니다(10.432초 / 9.395초 / 9.573초).
실제 CPU 프로세스 8개의 잠금/파일 저장과 모의 GPU 슬롯 8개에 대한 18개 작업 배분,
작은 실제 HF 모델의 probing 동시 요청 8개가 한 번만 생성하는 동작을 확인했습니다.
셸 wrapper의 순서와 실패 시 중단도 모의 실행기로 확인했습니다.
이는 실제 GPU 8장 실측이 아닙니다. 상세 기록은
`studies/grpo_v1/verification/a100x8-20261018/report.json`과 같은 폴더의 로그에 있습니다.
