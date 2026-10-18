# MATH 난도 순서 비교 실험: 데이터 준비

새 코드는 모두 `yj/`에 있으며 기존 `math_eval_a100.py`와 `yj_dff_test/`는 수정하지 않습니다.
요청의 `TH`는 MATH로 해석했습니다.

## 고정 조건

- seed: 42
- 공식 train에서 Level 1–2 / 3 / 4–5 각각 100개, 총 300개
- 공식 test에서 같은 구간 각각 30개, 총 90개
- 무작위/커리큘럼은 동일한 300문제를 한 번씩 사용합니다. 커리큘럼은 쉬움 100개 → 중간 100개 → 어려움 100개이며 각 구간 내부는 무작위입니다.
- V1과 V2 각각 random/curriculum, 총 4개 독립 실행 설정을 저장합니다. 모두 원래 시작 모델에서 시작하며 다른 실행의 체크포인트를 이어받지 않습니다.

| 소스 | 고정 revision |
| --- | --- |
| Qwen/Qwen2.5-1.5B | `8faed761d45a263340a0528343f099c05c9a4323` |
| hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo | `c014edac2e4bbaba644d7af74dc22d4526ec51c9` |
| EleutherAI/hendrycks_math | `21a5633873b6a120296cce3e2df9d5550074f4a3` |

모델 SHA는 Hugging Face API에서 확인했습니다. 원본 출처:
[V1](https://huggingface.co/Qwen/Qwen2.5-1.5B),
[V2](https://huggingface.co/hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo),
[MATH 공식 분할 미러](https://huggingface.co/datasets/EleutherAI/hendrycks_math).
로컬 `data/math/train.jsonl`에는 `test/...` 문제 ID가 있어 사용하지 않습니다.
고정된 미러의 train 7,500개/test 5,000개를 읽고 전체 공식 test와 내용이 중복되는 학습 문제를 제외합니다.
문제 내용에서 공백을 제거한 SHA256으로 중복을 확인합니다(의미상 중복 탐지는 아님).
미러에는 원래 문제 파일명이 없어 고정 revision의 split/subject/행 번호로 문제 ID를 만듭니다.

## 실행

현재 `math_eval` 폴더에서:

```bash
python yj/prepare_data.py
python -m unittest discover -s yj -p 'test_*.py'
```

기본 출력은 `yj/artifacts/seed42/`입니다. 이미 있으면 덮어쓰지 않습니다.
다시 검증하려면 캐시를 이용해 별도 경로로 실행합니다.

```bash
python yj/prepare_data.py --offline --output yj/artifacts/seed42_repeat
```

필요한 라이브러리는 `yj/requirements.txt`에 있습니다. 다운로드 캐시도 `yj/.cache/`에만 저장합니다.

## 저장 파일

- `train_selected.jsonl`, `test_selected.jsonl`: 선택한 문제 ID, 원문, 풀이, 난도, 출처
- `train_random.jsonl`, `train_curriculum.jsonl`: 순서가 지정된 학습 문제
- `eval_data/math/test.jsonl`: 기존 A100 평가기에 바로 넣는 공통 평가 데이터
- `config.json`, `manifest.json`: seed, 소스 revision, 그룹별 개수, 원본 및 출력 파일 SHA256
- `runs.json`: V1/V2 × random/curriculum의 독립 시작 모델과 파일 경로(이 파일 기준 상대 경로)

이 단계는 데이터와 실험 설정을 준비합니다. RL 학습기나 GPU 학습 실행은 포함하지 않습니다.
학습기 연결 시 `runs.json`의 model/tokenizer revision을 전달하고 준비된 순서에 추가 shuffle을 적용하지 않아야 합니다.

## 기존 A100 평가 코드 사용

기존 평가기는 `--revision`을 지원하지 않으므로 정확한 revision으로 내려받은
시작 모델 로컬 경로나 학습이 끝난 모델의 로컬 경로를 전달합니다.
아래 `MODEL_PATH`는 해당 경로로 설정해야 합니다.

```bash
python math_eval_a100.py \
  --model_name_or_path "$MODEL_PATH" \
  --data_names math --data_dir yj/artifacts/seed42/eval_data \
  --split test --num_test_sample -1 --prompt_type cot \
  --seed 42 --temperature 0 --n_sampling 1 \
  --use_vllm --save_outputs --output_dir yj/runs/eval
```

각 모델/학습 방식마다 `--output_dir`을 별도로 지정합니다.
