# 001 — IQL 파이프라인 구현 + D2(cap10) 실험 결과

> 2026-06-22. plan(plan/001)의 5개 산출물을 실제 구현·검증하고, 첫 실제 실험 D2(IQL@cap10)를
> GPU 학습→평가까지 돌린 기록. 모든 수치는 **실측**. append-only, 엄밀 + 쉽게.
> 참조: 현황 [[001-status-synthesis]] · 목표 [[001-goal]] · 계획 [[001-iql-execution-plan]] · 환경 [[002-env-setup]]

---

## 0. 한 줄 요약
**IQL 학습→f110 평가 전 파이프라인을 구현·검증 완료.** 그 과정에서 **action frame 불일치**라는
잠복 정합성 버그를 실측 발견·수정(harmonization). 첫 실험 **D2(IQL@cap10, 300k step)**: 12개
체크포인트 중 **2랩 완주 0개(G1 미달)**. 그러나 **best(step 50k)는 lap 1을 27.96s에 완주**(= cap10
기록 56.14s와 동일 페이스) 후 **lap 2 도중 충돌** — "첫코너 실패"가 아닌 **lap2 견고성 + 심한 과적합**
문제. 첫코너 병목은 넘었고, 목표에 의외로 근접.

---

## 1. 구현 산출물 (전부 동작 검증)

| 파일 | 역할 | 검증 |
|---|---|---|
| `vendor/CORL/iql.py` 등 | CORL IQL 원본(상류 무수정 vendor) | 클래스 재사용 |
| `f1tenth_data.py` | npz→CORL 5-key dict + **action 조화** | D2=102,246 / D3=591,121 transition, 보상재구성 오차 0 |
| `train_iql.py` | IQL 학습(CORL 클래스 재사용) | CPU·**GPU(CUDA)** 학습, resume, NaN가드, 촘촘 ckpt, JSONL/TB |
| `eval_iql.py` | f110 단일 정책 평가(2랩) | Oschersleben 완주판정·시간비교·JSON |
| `eval_sweep.py` | 한 run의 여러 ckpt 일괄 평가(env 1회 빌드) | D2 12개 스윕 |

### 교체점 5개 (plan §4 — CORL을 우리 문제로 적응)
1. **데이터**: `gym.make`+`d4rl.qlearning_dataset` → `f1tenth_data.load_f1tenth_dataset`(npz, obs133/act2).
2. **import**: vendored iql.py 최상단 `import d4rl, gym, wandb` 를 **stub**.
   - 학습 venv: `d4rl/gym/wandb` 전부 MagicMock(단 `gym.Env`는 주석평가 위해 진짜 `object`).
   - 평가 venv(RL_project): **gym은 real 유지**(f110_gym 의존!), `d4rl/wandb/pyrallis`만 stub.
3. **eval 분리**: 인라인 eval/env 제거 → 평가는 별도 프로세스(eval_iql.py).
4. **촘촘 ckpt**: eval과 분리, 번호본 보존 + latest 롤링 + obs_norm.npz + config.json.
5. **로깅**: wandb → JSONL + TensorBoard.

### 2-venv 실행 (격리 — [[002-env-setup]])
```bash
# 학습 (GPU, 반드시 background): f1tenth_IQL/.venv
.venv/bin/python train_iql.py --datasets cap10_full --device cuda --out_dir runs/<name> ...
# 평가 (CPU, f110_gym): RL_project/.venv
/home/dlacksdn/f1tenth_RL_project/.venv/bin/python eval_sweep.py --run_dir runs/<name>
```

---

## 2. ★ 핵심 발견 — action frame 불일치(harmonization) [가장 중요]

### 문제
저장된 action은 정규화 [-1,1]인데, 수집 시 `NormalizeActions`가 **수집 시점 v_max**로 [-1,1]→물리속도를
매핑한다. 따라서 **같은 정규화 action 값이 데이터셋마다 다른 물리속도**를 의미한다.

**실측 복원**(분산 0, p10=p90=median): `state[t][4]·20 = denorm(action[t][1], v_max_collect)` 관계로 역산.

| 데이터셋 | 수집 v_max | action[1]=+1 의 물리속도 |
|---|---|---|
| cap10_full, cap10 | 10 | 10 m/s |
| cap15 | 15 | 15 m/s |
| cap20 | 20 | 20 m/s |

이걸 모르면:
- **D2(cap10)**: eval을 기본 v_max=20으로 돌리면 정책의 "+0.5"가 7.5→12.5 m/s로 **2배 빨라져 조용히 크래시**.
- **D3(혼합)**: 같은 (state,action)이 데이터셋마다 다른 (next,reward) → **Bellman 타깃 모순 → stitching ill-posed**.

### 해법 — 공통 frame으로 재정규화
speed action만 단일 공통 frame(`COMMON_V_MAX=20`)으로 조화. state는 이미 고정 /20이라 일관, steer(S_MIN/S_MAX
공통)도 불변.
```
a1' = (a1+1)·(v_collect − V_MIN)/(v_common − V_MIN) − 1      (V_MIN=−5)
```
eval은 항상 env v_max=20으로 굴려 공통-frame action을 올바른 물리속도로 환원.

**교차검증**(통과): 조화된 cap10/15/20의 raw +1 → eval(v_max=20) 물리속도 = **정확히 10/15/20 m/s**.
조화 후 speed 범위: D2(cap10) [−0.94, **0.20**](=10 m/s 상한), D3 [−1.0, **1.0**](cap20=20 m/s).

> 이게 **G3(56초 경신)이 원리적으로 가능해지는 전제**다 — 직선=cap20 고속(20 m/s), 코너=cap10 감속을
> **한 frame에서 stitch** 할 수 있게 됨.

### 데이터 정렬 규약(이미 [[001-status-synthesis]]서 확정, 재확인)
DreamerV3 replay 규약: `action[t]/reward[t]`는 obs[t]를 **만든** 행동·obs[t] **도착** 보상.
→ 표준 transition: `s_t=obs[:-1], a_t=action[1:], r_t=reward[1:], s'=obs[1:], done=is_terminal[1:]`.
(corr(action[t].speed, state[t][4])=0.99 ≫ 0.48 로 실측 확정.)

### f110_gym 수정(PDF p.30-31) — 이미 반영됨
`base_classes.py:432-434` `check_collision`의 충돌 latching 제거(`np.maximum` 주석 → `self.collisions =
new_collisions`)는 **RL_project f110_gym에 이미 적용**된 상태. 우리 eval이 2-venv로 그 f110_gym을
재사용하므로 추가 작업 불필요(소스 무수정 규칙도 준수).

---

## 3. D2 실험 — IQL @ cap10

### 설정
```
datasets=cap10_full(102,246 transition: 완주30/충돌10)  obs=133  act=2(조화 v_common=20)
γ=0.999  β=3.0  iql_τ=0.7  actor=Gaussian  collision_penalty=-50  lap_bonus=+100(원본)
max_steps=300,000  batch=256  ckpt=25k마다(12개)  학습 2512.8s(~42분, ~120 it/s, RTX 4060 Ti)
out=runs/d2_iql_cap10/
```
- collision_penalty=-50: 사용자 결정(원본 보상 충실은 Dreamer 제약일 뿐, 우리 목표엔 무관 — 우리 보상
  재구성은 독립이라 왜곡 우려 없음. 충돌회피 강화가 목표에 유리).
- γ=0.999: +100 lap 보상을 value로 bootstrap(0.99는 lap에 blind). 학습 중 q_loss가 +100/-50 샘플
  배치에서 스파이크 → value가 희소보상 전파 중인 증거.

### 결과 — 스윕(12 ckpt × 1 ep, eval 결정론적이라 1ep 충분)

| step | 완주율 | 생존 step(s) | 비고 |
|---|---|---|---|
| 25k | 0 | 259 (5.2s) | |
| **50k** | **0** | **2422 (48.4s)** | **★ best: lap1 완주 27.96s, lap2 중 충돌** |
| 75k | 0 | 55 | 급붕괴 |
| 100k | 0 | 71 | |
| 125k | 0 | 173 | |
| 150k | 0 | 315 | |
| 175k / 200k | 0 | 803 | |
| 225k | 0 | 247 | |
| 250k | 0 | 702 | |
| 275k | 0 | 304 | |
| 300k | 0 | 271 (5.4s) | 충돌직전 조향 -1.0 포화(벽으로 직진) |

**2랩 완주 0/12 (G1 미달).**

### best(50k) 정밀 진단
- **lap 1 완주 = 27.96s** → ×2 = **55.9s ≈ cap10 기록 56.14s 동일 페이스**. progress_reward_sum=484.7(≈1.7랩 분량).
- speed action ~0.18(≈9.5 m/s, cap10 거의 최고속) — 정체 아님, 제 속도로 주행.
- steer 활발([-0.98,0.98]), 첫 action 정상(코너 진입). **lap 2 도중 단발 실수로 충돌**.

---

## 4. 분석 — 왜 2랩 완주 0인가

1. **심한 과적합**: 50k ≫ 75k(즉시붕괴) ≫ 300k(조향 포화). 300k=750 epoch(102k transition)은 과다.
   value_loss 0.0003(1k)→0.65(100k)→1.84(300k) 상승, actor 분산 붕괴(actor_loss 큰 음수).
   **이른 시점(≈50k)이 최적**, 이후 퇴화.
2. **lap2 견고성**: 첫코너 병목은 **넘었다**(이전 BC는 첫코너 0%였음). 50k는 lap1을 기록페이스로 완주.
   2랩 실패는 lap2 누적오차/단발실수. 결정론 정책이라 한 번 어긋나면 복구 불가.
3. **표현/구조 한계 가능성**: min-pool lidar(1080→128) + feedforward MLP는 Dreamer(순환 world model)
   대비 시간맥락·정밀도 부족. lap2의 특정 코너에서 한계.

→ 파이프라인·정합성은 정상. 병목은 **(a) 과적합 억제 + (b) lap2 견고성**.

---

## 5. 다음 단계 (제안)

1. **이른-체크포인트 정밀 스윕**: ckpt_freq=5k로 재학습(또는 0~80k 구간 집중) → 50k 근방 sweet spot 정밀 탐색.
   조기종료가 핵심임을 D2가 보여줌. (best 정책을 그래도 확보)
2. **D3 스티칭** (목표 본진): cap10+cap15+cap20(591k, 충돌 672). crash 데이터가 value에 "어디서 죽는지"를
   학습 → **lap2 견고성↑** + 직선 고속(cap20 frame) stitch. D2의 near-miss가 정확히 D3의 동기.
3. (옵션) 과적합 억제: max_steps↓, actor_dropout, 또는 정규화. BC 베이스라인(any_percent_bc)으로
   IQL>BC 확인.

---

## 6. 재현 / 파일 위치
- 학습 로그: `runs/d2_iql_cap10.log` · 체크포인트: `runs/d2_iql_cap10/checkpoint_*.pt`(보존)
- 스윕 결과: `runs/d2_iql_cap10/sweep_f1tenth_Oschersleben.json`
- 정규화 통계: `runs/d2_iql_cap10/obs_norm.npz` · 설정: `runs/d2_iql_cap10/config.json`
- GPU 스모크(검증용): `runs/_d0_gpu_smoke/`
```bash
# 재평가
/home/dlacksdn/f1tenth_RL_project/.venv/bin/python eval_sweep.py --run_dir runs/d2_iql_cap10
```
