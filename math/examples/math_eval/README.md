# GRPO Curriculum 연구 파이프라인

현재 실행 안내와 연구 설계는 [yj/README.md](yj/README.md)에 정리되어 있습니다.
설정은 [yj/research.json](yj/research.json), 실행 진입점은 `python yj/train.py research ...`입니다.

S0=Base, S1/S2=같은 사전 RL 궤적의 체크포인트라는 제안 설계로,
각 상태에서 새 LoRA를 초기화하여 Random/Curriculum을 비교합니다.
0/100/200/300 update마다 저장한 모델에 대해 학습 300문제 no-update probing과
MATH500 avg@8·별도 greedy 평가를 지원합니다.

현재는 draft 설계입니다. 데이터 준비와 CPU 검증을 수행했으며 GPU 본 학습·전체 평가는 실행하지 않았습니다.
기존 V1/V2·90문제 설계와의 충돌 및 주 지표를 검토하고 `finalize`로 확정한 뒤 실행합니다.
기존 `yj/experiment.json`과 `yj/artifacts/seed42/`는 이전 설계의 기록으로 보존합니다.
원본 [math_eval_a100.py](math_eval_a100.py)는 수정하지 않았습니다.

```bash
python yj/train.py research plan
```
