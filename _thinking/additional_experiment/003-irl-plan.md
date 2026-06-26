# 003 — 역강화학습(IRL, 과제 Topic-2) 실험 계획·맥락

> 2026-06-24. 새 세션 인수인계용 **맥락/설계 정리**(아직 아무것도 실행 안 함, 설계 공간 열려 있음).
> 본 보고서 프로젝트(IQL)와 **별개**의 탐색 라인. append-only. 한글.
> 선행 맥락: [[001-no-completion-stitching]] · [[002-rconly-ablation]] · 과제스펙 `raws/AIE4003_RL_F1TENTH.pdf` **p8**

---

## 0. 무엇을/왜 (과제 Topic-2, PDF p8)
**역강화학습(Inverse RL) for Autonomous Driving.**
- 목표: 주행 데이터로 **보상함수(reward)를 학습**하고, 그 보상으로 더 나은 주행 정책을 학습.
- 핵심: ① expert 데이터로 **숨겨진 보상함수 추정**, ② 학습된 보상 기반 RL, ③ **사람이 설계한 reward 없이** 정책 학습.
- 기대: 수작업 reward 대비 **일반화↑**, **새 트랙에서도 안정 주행**, 학습된 reward의 효과 분석.
- (대비) 지금까지 한 IQL/offline RL은 *사람이 설계한* reward(progress+collision+lap)를 썼다. IRL은 그 reward를 **데이터에서 학습**으로 대체한다.

## 1. 지금까지의 맥락 (요약)
- 이 폴더(`f1tenth_IQL`)는 **offline RL(IQL)** 본 과제 프로젝트. Oschersleben 2랩을 충돌 데이터 스티칭으로 34.32s 완주(cap10 56.14s 기록 경신, G3 달성). 보고서 작성·push 완료(`_thinking/report/`).
- `additional_experiment/` 라인(본 과제와 별개): **001** = 완주 토대(cap10)·1랩 주행조차 없는 데이터만으로 2랩 완주 가능함을 입증(7/30, 560k seed 3/4, ~34.5s). **002** = rc-only ablation(rc가 enabler, 단독이면 취약 1/30).
- **003(본 문서) = 또 다른 별개 라인: IRL.** 위 IQL/offline 작업과 독립.

## 2. 가용 자원 (실측 grounding)
- **전문가(완주) 데모 — 현재 디스크엔 *느린* 것만**:
  `/home/dlacksdn/f1tenth_RL_project/runs/crash_data/` 의 cap10_full 30ep(2랩 56s, 9.2m/s), cap8_jitter 26ep(69s),
  cap5_full 22ep(114s). 전부 느림.
- **빠른 전문가(Dreamer best, lap 16.6s ≈ 2랩 ~33s) 완주는 미저장** — 그러나 정책 파일 있음:
  `/home/dlacksdn/f1tenth_RL_project/runs/stage2_oschersleben/KEEP/KEEP_oscher_lapbest16.6s_step85k_FULL.pt`
  (+ `KEEP_policy_best_lap16.6s_step85k.pt`). → **즉석 롤아웃 수집 가능**. 수집 스크립트 `f1tenth_RL_project/scripts/collect_crash_data.py`(--ckpt/--v_max/--start-jitter, 단 기본은 충돌만 저장 → 완주 저장 옵션·로직 확인 필요).
- **IRL 인프라는 없음**: `vendor/CORL`에 iql.py·td3_bc.py·any_percent_bc.py(BC)뿐. GAIL/AIRL/판별자/보상모델 코드 0.
  → **보상 학습 부분은 신규 구현 필요**(IQL "데이터만 바꿔 재학습"과 차원이 다른 작업).

## 3. 설계 공간 (미확정 — 사용자 결정 대기)
**(A) 오프라인 "보상 학습 → IQL"** [재사용 多, 내 추천 1차]
1. 전문가 데모 E vs 잡다 데이터(crash_data)를 구분하는 **판별자 D(s,a)**(작은 MLP, 133D obs+2D act) 학습.
2. **학습 보상** R(s,a)=log D−log(1−D) (GAIL/AIRL식).
3. 기존 **IQL 파이프라인 그대로**, 저장된 사람-설계 보상 대신 **R(s,a)로 교체**(`f1tenth_data.py` 보상 훅).
4. 평가: 2랩 완주/시간 — *사람 설계 reward 없이* 좋은 주행 복원되나?
- 신규 코드: 판별자 학습기 + 로더 보상 덮어쓰기. **중간 규모.** 한계: 판별자를 정책 롤아웃과 반복 갱신하지 않는 *1-shot* 근사(정통 GAIL/AIRL은 온라인 반복).

**(B) 온라인 AIRL/GAIL** [정통, 큰 빌드]
- 판별자 ↔ 온라인 RL(PPO/SAC, f110 상호작용) 반복. AIRL은 **전이 가능한 보상** 복원 → **새 트랙 일반화**(Topic-2 핵심 기대)에 직결. 단 온라인 RL 루프 신규 구축(대공사). Topic-2는 온라인 허용이라 규칙상 OK.

**내 추천(미확정)**: (A) 오프라인 보상학습→IQL 로 1차 + 전문가는 **빠른 Dreamer(16.6s) 새로 수집**. 흥미로우면 (B) 온라인으로 키워 전이보상·새트랙 일반화(보고서 §8.2와 연결).

## 4. 운영에 필수인 기술 사실·규약 (IQL 라인에서 확립)
- **obs** = concat(min-pool lidar 128, state5)=**133D**. **action** = [steer, speed], 정규화 **[−1,1]** 2D. 평가 env V_MAX=20.
- **데이터 정렬(DreamerV3)**: action[t]/reward[t]는 obs[t]를 만든 행동·보상. 로더 s=obs[:-1],a=action[1:],r=reward[1:],s'=obs[1:],done=is_terminal[1:].
- **action 조화(harmonization)**: `a1'=(a1+1)(v_collect+5)/(v_common+5)−1`, v_common=20. (cap별 폴더명에서 v_collect 파싱: `f1tenth_data.py` 정규식 `cap_?(\d+)`.)
- **2-venv**: 학습=`/home/dlacksdn/f1tenth_IQL/.venv`(torch 2.4.1+cu124, GPU). 평가/관람/수집=`/home/dlacksdn/f1tenth_RL_project/.venv`(f110_gym 있음).
- **★GPU 학습은 반드시 run_in_background**(foreground+CUDA=exit144). 정지 전 디스크 ckpt 확인. 모델 저장 촘촘히(ckpt_freq 작게).
- **평가**: `eval_iql.py --ckpt ... --seed S --out ...`(결정론, seed=라이다 스캔 노이즈), `eval_sweep.py --run_dir ...`(전 ckpt 스윕). 완주 ckpt는 seed 0~3 스윕으로 견고성.
- **IQL D4 하이퍼(통제 기준)**: γ=0.999, collision_penalty=−50, β=3.0, iql_τ=0.7, common_v_max=20, lidar_n=128, hidden 256×2, lr 3e-4, batch 256, 600k step, ckpt_freq 20000.
- 코드: `train_iql.py`(--datasets <crash_data 하위폴더> --out_dir 등), `f1tenth_data.py`(npz→CORL+조화, DATA_ROOT env), `eval_iql.py`, `eval_sweep.py`, `vendor/CORL`.
- 데이터 위치: `f1tenth_RL_project/runs/crash_data/`(cap5/8/10/15/20, *_rc, 그리고 본 세션이 만든 `*_sub1lap` 심링크 폴더=<1랩 전용).
- **규약**: `_thinking` append-only·명시 지시 시에만 저장. git commit/push/pull·코드구현은 사용자 명시 지시 시에만(자율 push 금지). 로그·모델·데이터 폐기 금지. 산출물 덮어쓰기 금지(번호 증분 또는 신규 폴더). Dreamer/RL_project 학습·평가 독립성 훼손 금지. 한글로 답.
- repo: github.com/dlacksdn/f1tenth_IQL (branch main).

## 5. 미결정 (사용자 지시 대기)
1. 전문가 데모: 빠른 Dreamer(16.6s) 새 수집 vs cap10 느린 완주(즉시).
2. 범위: (A) 오프라인 보상학습→IQL 먼저 vs 곧장 (B) 온라인 AIRL.
3. 평가 축: Oschersleben 복원만 vs 새 트랙(map_easy3) 일반화까지.
