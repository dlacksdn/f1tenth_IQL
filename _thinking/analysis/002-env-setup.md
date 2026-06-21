# 002 — 환경설정 (IQL 학습 venv) 세팅 기록

> 2026-06-22. plan(plan/001) §2의 전용 venv를 실제로 구축한 기록. 모든 버전·동작은 **실측 검증**됨.
> append-only. 엄밀(버전·경로·명령) + 쉽게(표·이유). 새 세션이 환경을 그대로 재현/이해할 수 있게.

---

## 0. 한 줄
**IQL 학습용 전용 venv를 `/home/dlacksdn/f1tenth_IQL/.venv`(python3.8.10)에 구축, torch 2.4.1+cu124·cuda True
(RTX 4060 Ti)·numpy 1.24.4·pyrallis·tensorboard 설치·검증 완료.** 검증된 RL_project/Dreamer 스택은 무접촉
(격리). **평가만** 기존 RL_project venv를 빌려 쓴다(f110_gym 때문).

## 1. venv 위치·생성
- 경로: **`/home/dlacksdn/f1tenth_IQL/.venv`** (`.gitignore`로 git 제외 — repo 비대화 방지).
- Python: **3.8.10** (시스템 유일 버전, RL_project와 동일 → 호환성 안전).
- 생성: `python3.8 -m venv /home/dlacksdn/f1tenth_IQL/.venv` → pip 20.0.2 부트스트랩 → **pip 25.0.1**로 업그레이드.
- 디스크: **5.1 GB** (884 GB 여유 중 — torch는 pip 캐시서 설치돼 재다운로드 거의 없었음).

## 2. 설치 패키지 (실측 버전)
| 패키지 | 버전 | 용도 |
|---|---|---|
| **torch** | **2.4.1+cu124** | IQL MLP 학습 (RL_project와 정확히 동일 빌드) |
| nvidia-*-cu12 | 12.4.x (cudnn 9.1.0) | torch 번들 CUDA 런타임 |
| **numpy** | **1.24.4** | 데이터 처리 (RL_project와 동일, <2 핀) |
| **pyrallis** | 0.3.1 | CORL iql.py의 config 데이터클래스 (import 호환용) |
| **tqdm** | 4.68.3 | 진행바 |
| **tensorboard** | 2.14.0 | 학습 곡선 로깅 |
| protobuf | 5.29.6 | (tensorboard 의존 — §5 격리 이유) |

설치 명령(재현용):
```bash
PY=/home/dlacksdn/f1tenth_IQL/.venv/bin/python
$PY -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
$PY -m pip install "numpy==1.24.4" tqdm pyrallis tensorboard
```

## 3. 검증 결과 (전부 PASS)
- `torch.cuda.is_available()` = **True**, `get_device_name(0)` = **NVIDIA GeForce RTX 4060 Ti** (드라이버 596.21, 8 GB).
- **GPU matmul** (1024² @ cuda) = OK, 결과 유한.
- **TensorBoard SummaryWriter** = OK (이벤트 파일 기록됨 — protobuf 5.29.6에서도 정상).
- pyrallis / tqdm / numpy import = OK.

## 4. 일부러 제외한 것 + 이유
CORL의 `requirements`는 **torch 1.11+cu113 + d4rl + mujoco-py + gym0.23 + jax + wandb**를 핀하지만 전부 설치 안 함:
- **d4rl / mujoco-py / gym0.23 / jax**: D4RL 벤치마크 전용. 우리는 자체 npz 데이터를 쓰고 d4rl 경로를 우회하므로 불필요
  (cu124에서 빌드도 안 됨). → train_iql.py에서 `d4rl/gym/wandb` import를 **stub**해 CORL 클래스만 재사용(plan §2).
- **wandb**: 네트워크·로그인 필요 → 순수 offline 위배. TensorBoard + JSONL로 대체.

## 5. ★ 2-venv 아키텍처 + 격리 근거
| 용도 | venv | 비고 |
|---|---|---|
| **IQL 학습** | **이 새 venv** (`f1tenth_IQL/.venv`) | 격리. 순수 offline(env 무접촉). |
| **IQL 평가** | **기존 `f1tenth_RL_project/.venv`** | f110_gym이 거기 있음. eval_iql.py를 그 venv로 실행(설치 0, 읽기 전용 사용). |

**왜 학습을 격리했나(이번 설치가 증명):** tensorboard가 **protobuf 5.29.6**을 끌어왔다. 만약 이걸 검증된
RL_project venv(Dreamer 학습·데이터 수집·평가 인프라)에 설치했다면 protobuf/의존성 충돌로 그 스택이 깨질
수 있었다 — 그 스택이 깨지면 데이터 재생성·평가가 다 막히는 재앙. **격리 비용은 디스크 5GB뿐, 안전 이득은 큼.**
(CLAUDE.md "현 모델과 다른 모델의 학습/평가 독립" 규칙 준수.)

## 6. 사용법
```bash
# 학습 (GPU): 반드시 run_in_background (foreground+CUDA=exit144)
/home/dlacksdn/f1tenth_IQL/.venv/bin/python /home/dlacksdn/f1tenth_IQL/train_iql.py ...

# 평가 (CPU, f110_gym): RL_project venv로 별도 프로세스
/home/dlacksdn/f1tenth_RL_project/.venv/bin/python /home/dlacksdn/f1tenth_IQL/eval_iql.py ...
```

## 7. git
- `.venv/`는 `.gitignore`에 포함(이미 push된 .gitignore에 `.venv/`·`__pycache__/`·`*.pyc`·`*Zone.Identifier`·
  `*.pt`·`checkpoints/` 등재). venv는 repo에 안 올라간다.

## 8. 다음 단계 (plan/001 기준)
1. **CORL vendor**: `/home/dlacksdn/CORL/algorithms/offline/{iql.py, any_percent_bc.py, td3_bc.py}` →
   `f1tenth_IQL/vendor/CORL/` 복사(상류 무수정).
2. **`f1tenth_data.py`**: npz → CORL 5-key dict (obs 133D, action [-1,1], reward γ=0.999, terminals=is_terminal).
3. **`train_iql.py`**: iql.py 클래스 import(+d4rl/gym/wandb stub) + 교체점 5개 + 촘촘 ckpt.
4. **`eval_iql.py`**: RL_project venv, f110 2랩.
5. **D0 스모크** → D1 BC → D2 IQL@cap10 → D3 IQL@cap10+15+20.

## 참조
- 계획: [[001-iql-execution-plan]] (plan) · 현황: [[001-status-synthesis]] (analysis) · 목표: [[001-goal]] (goal)
- 동일 스택 출처: `f1tenth_RL_project/.venv` (torch 2.4.1+cu124, numpy 1.24.4 — 검증된 박스 구성).
