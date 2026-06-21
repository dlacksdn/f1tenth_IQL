# 001 — 현황 종합 (f1tenth_IQL 프로젝트 SSOT)

> 2026-06-21. 새 프로젝트 `f1tenth_IQL` 착수 시점의 전체 현황 종합. 이전 프로젝트(`f1tenth_planning_with_diffusion`)
> _thinking 53개 문서(5에이전트 정독) + 직접 데이터/정책 측정으로 확정. append-only.
> 데이터·정책·평가·미해결 병목·재사용 자산을 한 곳에 못박는다.

---

## 0. 한 줄
**LeWM→Diffuser 두 모델을 차례로 폐기(둘 다 BC/prior 완주 0%)하고, 정책 기반 Offline RL = IQL로 피벗.**
데이터(완주 cap10 + 고속충돌 cap15/20)·dreamer 정책·평가 인프라는 동결 보존·재사용. **진짜 미해결 병목은
알고리즘이 아니라 "폐루프 첫 코너 정밀도(covariate shift)"** — IQL의 승부수는 in-sample value 학습이 이 벽을
넘느냐다.

## 1. 전체 아크
- **LeWM(JEPA)**: reward-free goal-reaching, "빠르게"가 목적함수 아님 → 부적합, 폐기.
- **Diffuser(diffusion planner)**: P4 학습 성공(loss 0.003·value corr 0.98)했으나 **모든 데이터처방·K스윕·
  obs해상도에서 BC/prior 완주 0%**. 근본 = 생성·open-loop planner가 고속 폐루프에 부적합 + 충돌오염 prior +
  첫 코너 정밀도 + value의 "속도 선호(완주 아님)" 결함(D3). → 019 폐기.
- **피벗(019)** → 정책 기반 Offline RL. **020**(TD3+BC 우선) → **021 적대검수+직접검증** → **IQL이 더 안전한
  1순위 앵커**로 정정. → **현재: `f1tenth_IQL`에서 IQL을 CORL 기반으로 새로 구현.**

## 2. 데이터셋 위치 (확정·동결·gitignore) — `/home/dlacksdn/f1tenth_RL_project/runs/crash_data/`
직접 전수 측정(dt=0.02s, 2랩 기준):

| 디렉터리 | npz | 2랩완주 | 충돌 | 2랩 최速 | 역할 |
|---|---|---|---|---|---|
| **cap10_full** | 40 | **30** | 10 | **56.14s** | ★ 완주 정답 토대(베이스라인 1.9배) |
| cap10 | 21 | 0 | 21 | — | 충돌 보강 |
| cap8_jitter | 300 | 26 | 274 | 68.9s | 마진/커버리지(cap10 정책 v_max=8 rollout) |
| cap5_full | 31 | 22 | 9 | 114.2s(느림) | 베이스라인보다 느림 |
| cap5 | 13 | 0 | 13 | — | |
| **cap15** | 371 | **0** | 371 | — | 고속 충돌 재료(완주 0) |
| **cap20** | 291 | **0** | 291 | — | 최고속 충돌 재료(완주 0) |
| cap8_probe·cap5_lowsnap·_probe·_smoke | 20·0·2·1 | — | — | — | 타진/빈/소량 |

- **스키마(키)**: `lidar(T,1080)`·`state(T,5)`·`action(T,2,[-1,1])`·`reward`·`log_reward_{progress,collision,lap,
  reverse,diverged}`·`pose(T,3)`·`v_max(상수)`·`is_terminal`·`is_last`·`discount`·`log_lap_time_s`·`log_completed`·`logprob`.
- **state5 = [vel_x, vel_y, yaw_rate, prev_steer, prev_speed]**. ★ **state[4]는 "현재속도"(동적, 캡에서 v_max/20
  도달)** — 021의 "state[4]=v_max 상수 조건화"는 오류. v_max는 별도 상수 키이며 **action 역정규화에만** 쓰임.
- **action 역정규화(평가용)**: steer=`(a0+1)/2·(0.4189−(−0.4189))+(−0.4189)` (tier무관), speed=`(a1+1)/2·(v_max−
  (−5))+(−5)` (V_MIN=−5 고정, v_max=tier별 5/10/15/20). npz action은 정규화 입력값 → 역정규화 1회 정당.
- **reward 이미 저장됨**: progress(~0.2~0.4/step) + **lap +100/랩** + **충돌 −10**. IQL이 그대로 쓰거나 재구성 가능.
- **핵심**: 107.16s보다 빠른 2랩 완주는 cap10(30)·cap8_jitter(26)뿐, **전부 v_max≤10. cap15/20 안전완주 0개.**

## 3. Dreamer policy 위치 (확정·재사용 가능) — `/home/dlacksdn/f1tenth_RL_project/runs/*_oschersleben/`
| tier | 채택 ckpt | 결정론 2랩 lap | 역할 |
|---|---|---|---|
| **cap5** | `cap5_oschersleben/step_25k.pt` | **107.16s** | ★ behavior policy = 베이스라인 |
| **cap10** | `cap10_oschersleben/step_45k.pt`(=policy_best_lap26.2s) | **53.66s** | sweet spot(무진동) |
| **cap15** | `cap15_oschersleben/step_105k.pt`(best 18.0s/랩) | ~37.3s | 고속 reference |
| **stage2(cap20급)** | `stage2_oschersleben/policy_best_lap16.6s_step85k.pt` | ~35s | 최고속(무제한). ※완주 시연용 아님(no-expert) |

- ckpt: `step_Nk.pt`(148M, world model 포함) vs `policy_*.pt`(47M, actor만). 추가수집은 `collect_crash_data.py
  --ckpt --v_max`로 재사용. 데이터의 114/56s는 stochastic 수집이라 결정론 eval(107.16/53.66)보다 약간 느림(모순 아님).

## 4. ★ 핵심 미해결 병목 = 폐루프 첫 코너 정밀도
- BC는 **모든 prior·K·obs해상도에서 완주 0/10**. cap10(9.6m/s 풀스로틀, margin0) = 첫 코너 ~66step 즉발충돌;
  cap5(마진 큼, 저속) = 점진 표류. lidar256은 생존 1.4~1.8배 늘렸으나 완주 0.
- 원인 = closed-loop compounding error + covariate shift(완주 코너 데이터가 cap10 30ep로 얇음) + 충돌오염 prior.
- **IQL 가설**: 명시적 BC prior 없이 in-sample value/advantage로 직접 학습 → mixture-averaging·prior오염을 회피,
  이 벽을 넘을 가능성. (단 covariate shift는 알고리즘 공통 위험 — 미검증.)

## 5. RL 설계 함정 (검증됨 — IQL critic에 직결)
- **γ**: γ=0.99면 value가 "완주"를 못 봄(lap +100이 할인으로 소멸). **γ=0.999 필요**. 단 γ=0.999에서도 "cap20
  충돌고속 RTG > cap5 완주" → **value는 "완주"가 아니라 "속도"를 보상**(D3, 완화될 뿐 소멸 안 됨).
- **충돌 페널티**: 원본 −10은 약함(RTG 부호 역전 0%). **−50에서 88%, −200에서 100%** 역전. ≈−50이 임계.
  (단 offline RTG 통계 ≠ closed-loop 회피 — 논리 비약 미해소.)
- **보상 스케일**: lap +100 스파이크 = progress(~0.27)의 370배 → 정규화/클립 검토.

## 6. 데이터 레버 + 평가 v_max 명확화
- **고속(v_max≥15) 안전데이터 수집 불가**(정책 100% 충돌, 닭-달걀). **합법·유효 레버 = v_max≤10 커버리지
  보강**(시작점 jitter/섭동; cap8_jitter가 그 시도) → 코너 정밀도 직격.
- **평가 v_max**: eval env는 **V_MAX=20(헤드룸)** — 차가 20까지 낼 수 있다는 것이지 20 강제가 아님. 정책이
  ~10m/s 명령하면 56s로 주행. `eval_gate.py`에 `--v_max` 캡 일치 옵션 존재. → "v_max=20 벽"은 사실상 해소.

## 7. IQL로 가져갈 자산 vs 폐기
**재사용:** 데이터 로더 로직(`vendor/diffuser/diffuser/datasets/f1tenth.py` — 133D=lidar128 min-pool+state5,
action 역정규화 L97-110, `F1TENTH_MODE`, timeouts=`is_last&~is_terminal`, max_path_length≥2807) · 평가
(`f1tenth_RL_project/scripts/eval_gate.py`, **RL_project `.venv`에서**, 2랩·V_MAX20·baseline·`--v_max`) · 수집
인프라(`collect_crash_data.py --ckpt --v_max`, CPU). **IQL 학습 venv는 별도 신설**(타 프로젝트 venv 차용 금지).
**폐기:** Diffuser 궤적생성/value guidance/K-MPC/normalizer 글루, LeWM 라인 전체.

## 8. 절대 규칙 (계승 + 신규)
- git/구현/_thinking 저장은 **사용자 지시 시만**. _thinking append-only. 로그·모델·데이터 폐기 금지.
- **다른 모델 ↔ Dreamer 독립**(공용코드 수정 시 하위호환+스모크). 코어/물리/env reward/`_STATE_SCALE`/맵 무변경.
  **f110_gym 소스(RL_project 공유 editable) 수정 금지.**
- GPU=run_in_background(foreground+CUDA=exit144), kill 단독, 평가=RL_project `.venv`·V_MAX20·2랩·Oschersleben.
- **★ 신규(f1tenth_IQL CLAUDE.md): 체크포인트 저장 분기를 촘촘하게**(중간 끊겨도 손실 최소). 한글 답변.

## 9. 참조
- 이전 프로젝트 핵심 문서: `f1tenth_planning_with_diffusion/_thinking/plan_new/{019,020,021}`
- 데이터/정책: `f1tenth_RL_project/runs/crash_data/` · `runs/*_oschersleben/`
- IQL 구현: CORL `/home/dlacksdn/CORL/algorithms/offline/iql.py` (공신력 1위 PyTorch IQL, → vendor 예정).
- raws: `f1tenth_IQL/_thinking/raws/{IQL.pdf, f110_env.py, map_easy3.*, AIE4003 PDF}`. 과제 PDF p29~ = env 코드수정 가이드.
