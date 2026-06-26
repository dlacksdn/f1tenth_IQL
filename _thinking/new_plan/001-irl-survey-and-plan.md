# 001 — IRL(Topic-2) 문헌 전수조사 · offline vs online 결론 · 실행 계획

> 2026-06-26. 역강화학습(IRL, 과제 Topic-2) 라인의 **문헌 조사 + 패러다임 결정 + 추천 스택** 정리.
> 선행 맥락: [[../additional_experiment/003-irl-plan]] (IRL 설계 공간) · 본 보고서 IQL 프로젝트(완료).
> 작성 원칙: 엄밀하되, 작성자(사용자)가 다시 읽어도 바로 이해되도록 쉽게. append-only.
> 근거 데이터: 멀티에이전트 웹조사 워크플로우(에이전트 58개, 웹툴 호출 732회, 논문 raw 200편 → 중복제거 188편, 채점 149편).
> 원시 산출물 보존: `scratchpad/irl_full.md`(149편 전체 reason·적용법), 워크플로우 결과 JSON `tasks/wd5l13krs.output`.

---

## 0. 한 줄 요약 (TL;DR)

- **"expert로 보상함수를 학습한다"는 건 맞게도 offline IRL의 핵심 영역**이다. 하지만 **순수 offline만으로는 폐루프(closed-loop) 발산을 못 잡는다.**
- **정답 패러다임 = offline-then-online 하이브리드**(offline로 reward를 학습해 무겁게 깔고, online은 "수리"용으로 짧게). 사용자의 직관("offline 하다 online 가는 게 제일 보편적")이 이 셋업에 정확히 맞다.
- **1차 추천**: **랭크 기반 offline 보상학습**(D-REX / TROFI) → EPIC로 offline 검증 → 기존 IQL로 정책 추출 → (폐루프 발산 시) OLLIE/CSIL로 짧게 online finetune.
- **중대한 주의**: 003에서 1차 후보로 적었던 **GAIL식 판별자 보상 `R=logD−log(1−D)`은 "occupancy(상태분포) 매칭"이라, 느린 데모를 *모방*하게 만들어 너희의 입증된 'exceed not imitate(데모를 초월)' 명제와 정면충돌**한다. → 1차 보상으로 쓰지 말 것. (자세한 이유 §4·§7)

---

## 1. 이 문서가 답하는 질문

1. expert 데모로 reward를 학습하는 게 정말 offline IRL인가? → **그렇다.** 하지만 offline에 갇힐 필요는 없다.
2. **offline이 좋은가 online이 좋은가?** → §3.
3. 고전부터 프론티어까지 **어떤 논문들이 있고, 이 프로젝트에 얼마나 맞는가?** → §5(전체 랭킹), §6(추가 발견), §7(역사 계보).
4. 그래서 **우리는 뭘 할 것인가?** → §4(추천 스택), §3 끝(권장 경로 9단계).

---

## 2. 채점 방법론 (점수의 의미부터 정확히)

- 8개 축을 각 0–10으로 매기고 가중합×10 = **0–100 종합점수**. 등급: **S ≥80 / A 68–79 / B 55–67 / C 42–54 / D <42**.
- **가중치**(이 프로젝트 기준으로 설계):

  | 축 | 가중치 | 뜻 |
  |---|---|---|
  | reward_recovery | 0.22 | reward를 실제로 *복원*하나? (Topic-2 핵심). explicit > implicit > policy-only |
  | regime_fit | 0.14 | 데이터/계산 현실에 맞나? (고정 데모 보유 + sim 있음. online도 이념적으로 안 깎음) |
  | integration_ease | 0.14 | 기존 IQL+CORL 파이프라인에 붙이기 쉬운가? |
  | continuous_highdim | 0.12 | 연속 2D action + 133D obs(딥넷) 처리 가능? |
  | transfer_generalization | 0.12 | 학습 reward가 새 트랙(map_easy3)으로 전이되나? |
  | code_repro | 0.10 | 공식·실행 가능한 코드가 있나? |
  | maturity_impact | 0.08 | 학문적 영향력/인용 |
  | paradigm_synergy | 0.08 | offline→online 하이브리드 적합도 |

- **⚠️ 점수 = "이 프로젝트 적합도"이지 "학문적 중요도"가 아니다.** GAIL·MaxEnt IRL 같은 고전이 C/D인 것은 *직접 갖다 쓰기 어렵다*는 뜻이지 안 중요하다는 게 아니다(maturity는 8%만 반영).
- **한계**: LLM 패널이 루브릭으로 매긴 점수다(웹서치로 grounding했으나 무오류는 아님). 중복제거(arXiv ID/약어 기준)는 완벽하지 않고, 39편은 제목 정규화 불일치로 채점에서 누락(§6).

---

## 3. 결론 — offline vs online vs 하이브리드

세 갈래를 이 셋업(133D obs, 연속 [steer, speed], **고정된 충돌위주 데모셋 + 느리지만 동작하는 f110 sim**, 목표 = 전이 가능한 학습 reward·새 트랙 일반화)에 대고 평가:

| Regime | 장점 | 치명적 단점 | 적합도 |
|---|---|---|---|
| **순수 Offline** (T-REX, CLARE, SMODICE …) | 환경 상호작용 0 → sim 샘플비용·불안정 0. 기존 IQL에 **그대로 드롭인**(reward만 교체). 결정론·재현성↑ | reward 외삽오차(데이터 support 밖에선 reward가 틀림). occupancy/거리 매칭류는 **느린 데모를 따라가게** 만듦. **폐루프 covariate shift를 offline로는 못 잡음**(Diffuser 라인이 이미 당함) | ★ 주력 |
| **순수 Online** (AIRL, GAIL, GCL) | **가장 전이 가능한 reward**(AIRL: `r=g(s,a)+h(s)`, 동역학과 분리). 폐루프를 실제로 교정하는 유일한 길 | adversarial min-max 불안정(spectral norm 필수). on-policy 샘플비용=킬러(데이터수집이 이미 cap당 수시간). **기존 offline 투자 전부 폐기** | △ 단독은 최악 |
| **Offline→Online 하이브리드** | offline로 거의 맞는 explicit reward+정책을 공짜로 얻고, **짧은 online finetune으로 폐루프 갭만 수리**. OLLIE/CSIL이 인계 전용. **IQL은 같은 목적함수로 online 연장 native 지원** | 2단계 복잡성. random discriminator가 사전학습 정책을 unlearn(OLLIE/CSIL로 방지). offline reward가 애초에 틀렸으면 online이 나쁜 신호를 더 쫓음 | ★★ 최적 |

> **용어 풀이**
> - *폐루프(closed-loop) covariate shift*: 모델이 한 스텝씩은 정확해도, 자기가 만든 약간 빗나간 상태로 계속 굴러가다 보면 학습 때 못 본 상태로 빠져 발산하는 현상. 너희 Diffuser 라인이 "스텝 예측은 정확한데 닫힌 루프에선 후진/스핀"으로 죽은 게 정확히 이거다. **offline 지표(EPIC 포함)로는 절대 못 잡는다 → 짧은 online 폐루프 점검이 필수.**
> - *occupancy(상태분포) 매칭*: "전문가가 자주 방문한 상태 분포를 똑같이 따라가라"는 목표. 전문가가 느리면 **느림까지 따라간다** → "더 빨리" 목표와 충돌.
> - *disentangled reward(AIRL)*: 보상을 동역학과 분리해 복원 → 트랙(동역학)이 바뀌어도 보상이 살아남음 = 전이성 최강.

### 권장 경로 (9단계 — 이게 실행 계획의 뼈대)

1. **action harmonization 먼저.** 검증된 V_MAX 재정규화 `a1'=(a1+1)(v_collect+5)/(v_common+5)−1`을 적용해 cap별 궤적의 return/랭킹을 비교 가능하게. (IQL 성공 때 load-bearing이었고, 랭크 기반 보상의 전제조건.)
2. **랭킹을 공짜로 구성.** 데이터에 이미 있는 구조 활용: cap5 < cap10 < cap15 < cap20 (속도/return 순), + 충돌로 잘린 깨끗한 1랩 조각(18~20s/lap)을 고품질 빠른 재료로. **D-REX**(BC + 행동노이즈 주입으로 자동 랭킹)면 사람 라벨 불필요.
3. **offline로 explicit reward 학습.** T-REX/D-REX의 Bradley-Terry 랭킹 손실로 `r_θ(obs)` 학습. **TROFI**가 바로 "랭크→relabel→offline RL(TD3+BC/IQL)"의 인스턴스 = 너희 스택과 동형. 이게 'exceed the demonstrator'를 직접 인코딩.
4. **정책 학습 전에 reward를 offline 검증.** **EPIC distance**로 sanity 기준(예: progress/곡률 proxy)과 비교, shaping/scaling 불변인지 확인. 트랙별 spurious feature에 reward가 붙으면 재랭킹.
5. **보수성 추가**(OOD 방어). 충돌위주 OOD 상태에서 reward가 폭주하지 않게 **CLARE식**(또는 생성 월드모델 bi-level MLE) 페널티, 최소한 IQL의 in-sample(expectile) 기계는 OOD action을 안 건드림.
6. **relabel 후 기존 IQL로 정책 추출.** 코드 그대로, 데이터의 보상만 학습 보상으로 교체(드롭인).
7. **f110에서 폐루프 평가.** 2랩 결정론 게이트(cap5 107.16s 등 베이스라인 대비). 스텝은 맞는데 폐루프가 발산하면(=알려진 Diffuser 실패) online 수리로.
8. **짧은 online finetune.** **OLLIE**식 정렬된 policy+discriminator로 초기화하거나 **CSIL** coherent reward로 변환 후, IQL의 같은 목적함수를 online 연장. 필요시 **off-policy-AIRL/OPIRL**로 replay 재사용해 sim 비용↓. adversarial 항은 spectral norm으로 안정화. **상호작용은 최소로.**
9. **새 트랙 전이 검증.** 데모 재수집 없이 map_easy3에서 EPIC + 폐루프 완주율 보고 → "전이 가능한 학습 reward" 주장을 *실측으로* 뒷받침. 필요시 causal-invariance 정규화 추가.

---

## 4. 추천 스택 상세 (왜·어떻게)

목표는 **"느린 데모를 모방하지 말고 초월"**(너희 IQL 성공의 입증된 명제). 그래서 1차는 **occupancy 매칭이 아니라 랭크 기반 보상**이다.

- **D-REX (2019, CoRL, arXiv:1907.03976) — 1차 실용 추천.**
  BC로 정책을 만든 뒤 행동에 노이즈를 점점 키워 "노이즈 많을수록 나쁨"이라는 **자동 랭킹**을 만들고, 그 랭킹으로 explicit reward를 학습. **사람 라벨 0.** 너희 데이터엔 cap 속도/충돌 랭킹이 이미 있어 더 쉽다. (노이즈 롤아웃 생성 한 스텝만 살짝 online인데 f110에선 저렴, 혹은 로그 행동 교란으로 offline 근사 가능.)
- **TROFI (2025, RLC, arXiv:2506.22008) — 우리 파이프라인과 정확히 동형인 템플릿.**
  "T-REX 보상 → 고정 데이터 relabel → offline RL(TD3+BC)"을 그대로 구현. 너흰 TD3+BC 자리에 IQL을 끼우면 됨. 단 공식 코드 미확인(논문 보고 재구현, T-REX 코드 출발점 존재).
- **T-REX (2019, ICML, arXiv:1904.06387) — 개념 코어.** 랭크된 궤적의 pairwise 손실로 explicit reward, **데모 초월 외삽** 설계. 공식 코드 있음(github.com/hiwonjoon/ICML2019-TREX).
- **CLARE (2023, ICLR, arXiv:2302.04782) — 보수성(OOD) 안전장치.** 완전 offline IRL + 동역학모델 + return-gap 경계. 충돌위주 데이터의 외삽오차를 잡는 용도(133D 동역학모델 적합이 주 엔지니어링 부담).
- **EPIC distance (2022, ICLR, arXiv:2006.13900) — 보상 검증 "도구"(IRL 방법 아님).** 정책 학습 없이, shaping/scaling 불변으로 두 보상을 비교. **다른 동역학에서도 regret 상한을 보장** → 새 트랙 전이를 *학습 전에* 예측하는 게이트.
- **(폐루프 수리용) OLLIE (2024, ICML, arXiv:2405.17477) / CSIL (2023, NeurIPS, arXiv:2305.16498).**
  - OLLIE: static 데모로 **정렬된 정책+판별자**를 공동 사전학습 → online IL이 사전학습 정책을 붕괴 안 시킴(=naive GAIL-after-IQL의 #1 실패 방지). 공식 코드 있음(github.com/HansenHua/OLLIE-ICML24).
  - CSIL: BC 정책을 `r=α(log q_BC − log p)`로 역산한 **일관된 shaped reward**. 별도 판별자 불안정 없이 online/offline 둘 다 finetune. 공식 코드 있음(github.com/google-deepmind/csil).
- **IQL (2021, arXiv:2110.06169) — 연결조직.** 너희 본 알고리즘이자, **offline→online finetune를 같은 목적함수로 native 지원** → 인계가 config 변경 수준.

> **003 계획에 대한 직접적 수정 제안**
> 003 §3의 (A)안은 "판별자 D로 보상 `R=logD−log(1−D)` 학습 → IQL"이었다. 이건 GAIL/occupancy 매칭이라 **느린 cap5/cap10 분포로 정책을 끌어당겨 "초월" 목표와 싸운다.** → (A)안의 *보상 형태*를 **랭크 기반(T-REX/D-REX/TROFI)으로 교체**할 것. 파이프라인 골격("offline 보상학습 → 기존 IQL relabel")은 그대로 유지하되, 보상의 *종류*만 바꾸면 된다. (B)안(online AIRL)은 폐루프 수리·전이검증 단계에서 off-policy-AIRL/OLLIE로 최소 도입.

---

## 5. 전체 랭킹 (고전 → 프론티어, 149편 채점)

집계: **S 3 · A 28 · B 40 · C 58 · D 20.** (regime 표기: off=offline, on=online, off→on=하이브리드)

### 🟣 S 티어 (3) — 지금 당장 1차 후보

| 방법 | 연도 | 점수 | regime | 핵심 / 이 프로젝트 적용 |
|---|---|---|---|---|
| **T-REX** | 2019 ICML | **85** | off→on | 랭크된 궤적 pairwise 손실로 explicit reward, **데모 초월 외삽**. lap-time/충돌로 자동 랭크 → reward MLP → 데이터 relabel → IQL. action-free라 노이즈 행동라벨 오염 없음 |
| **AIRL** | 2018 ICLR | **84** | off→on | `r=g(s,a)+h(s)`로 **동역학과 분리된 전이 reward를 증명적 복원** → 새 트랙 전이 최강. vanilla는 on-policy → offline 사전학습 후 online finetune. off-policy-AIRL로 비용↓ |
| **CLARE** | 2023 ICLR | **83** | off→on | **완전 offline IRL** + 보수성(OOD reward 비관화) + return-gap 경계. 동역학모델 적합이 부담이나 충돌위주 데이터의 외삽오차 방어에 최적 |

### 🔵 A 티어 (28) — 강력 후보 / 핵심 도구

상세(상위, 전체 reason은 `scratchpad/irl_full.md`):
- **CSIL (79)** — BC를 일관 shaped reward로 역산, online/offline finetune. discriminator 불안정 없음. 하이브리드 교과서.
- **D-REX (79)** — BC+노이즈로 자동 랭킹 → T-REX 보상, 사람 라벨 0. **너희엔 T-REX보다 실용적 1차.**
- **Generative World-Model MLE IRL (79, NeurIPS Oral, 2302.07457)** — bi-level 최대우도 offline IRL + 보수성 + non-asymptotic 보장. (코드 github.com/Cloud0723/Offline-MLIRL)
- **Preference Transformer (79, ICLR, 2303.00957)** — transformer로 **비-마르코프 보상**(랩 품질=구간 속성). offline 선호로 학습 후 relabel → IQL.
- **SMODICE (76)** — state-occupancy 매칭 offline IL. **단 occupancy 과적합 → 1차 보상으론 부적, 보조/베이스라인용.**
- **SSRR (76)** — reward-vs-노이즈 곡선 회귀로 외삽(GT와 ~0.95 상관). D-REX 상위 호환.
- **ORIL (75)** — PU 판별자 보상 → relabel → offline RL. 너희 템플릿과 동형(단 거리/모방 편향 주의).
- **TROFI (73)** — "랭크→relabel→offline RL"의 정확한 인스턴스. 코드 미확인.
- **OLLIE (73)** — 정렬 policy+discriminator 사전학습 → online IL 붕괴 방지. off→on #1 위험 해결.
- **EPIC distance (73)** — 정책 학습 없이 보상 품질 offline 검증, 전이 regret 상한. **측정 도구.**

나머지 A (압축):
```
OPIRL(72,off→on) PEMIRL(meta,72,off→on) Offline Pref-Reward benchmarks(72,off)
Guided Cost Learning(71,off→on) CLUE(71,off) EBIL(71,off→on) DRLHP(71,off→on·live teacher 필요)
Off-policy AIRL(71,off→on) IQL(70,off→on·본 알고리즘) Spectral Norm(70·안정화 도구)
PWIL(69,off→on) f-IRL(69,off→on·occupancy 주의) SEABO(69,off·모방편향) MAHALO(69,off)
PEBBLE(69,off→on·live feedback) Causally-Invariant Reward(69,off→on·전이 보강) Rethinking AIRL(68) Lipschitz off-policy GAIL(68)
```

### 🟢 B 티어 (40) — 유효하나 2차

```
f-div IL관점(67) LobsDICE(66) UNIQ(66) rank-game(66) Bayesian REX 안전IL(66) VAIL(65)
DRAIL(diffusion AIL,65) Inverse Preference Learning(65) Pragmatic Deep IL(65) VIPER(영상예측보상,65)
Dual RL(64) Sampling MaxEnt IRL(64) Deep Bayesian Reward(64) VILD(64) Sample-eff AIL(64)
ML-IRL 유한시간보장(63) IDQL(63) What Matters for AIL(63) Transferable Reward 앙상블(62)
SURF(61) Pessimism in IRL(60) Covariate-shift IL(60) DWBC(60) Diffusion Reward(60) DIFO(60) IRL-VLA(60)
MaxCausalEnt primer(59) DAC(59) AWAC(59) PU Reward Learning(59) Continuous IOC(57) CRR(57)
Active Preference Reward(57) RLPD(57) Diffusion Q-L(56) ValueDICE(56) SQIL(56) OPOLO(56)
Off2On balanced replay(56) Cal-QL(55)
```

### 🟡 C 티어 (58) — 고전 토대 / 주변부 (계보 참고용; 이름·점수만)

```
LEARCH(54) Relative-Entropy IRL(54) Revisiting MaxEnt IRL(54) CEIL(54) ODICE(54) PEX(54) Reward Transfer in AIRL(54)
Max Margin Planning(53) SAM(53) Uni-O4(53) Confidence imperfect-demo(53)
Deep MaxEnt IRL(DeepIRL,52) WAIL(52) GAIL-driver(52) SMILe-meta(52) DICE diverse-demo(52)
Apprenticeship Learning(Abbeel&Ng,51) Driving with Style(51)
MaxEnt IRL(Ziebart,50) GAIL(50) MARWIL(50) Kuderer driving-styles(50) GAIfO(50) CPL(50) Regularized IRL transfer(50) Relaxed offline IL(50) MaxEnt IRL of Diffusion(50)
Large-scale cost learning(49) MandRIL-meta(49) B-Pref(49) TD3+BC(48) Watch This(48) JSRL(48) Reward Model Ensembles(48)
Non-Adversarial IL(47) GPIRL(46) Partial Identifiability(46) RL-VLM-F(46)
Apprenticeship+gradient(45) MaxCausalEnt(45) OptiDICE(45) LP offline reward(45) DPO(45)
Ng&Russell IRL(2000,44) Continuous MaxEnt IRL(44) Off-policy AIL convergence(44) ODT(44) Reward(Mis)design driving(44)
Game-theoretic apprenticeship(43) BCO(43) Is IRL harder than RL(43) IRL scale to large state(43) Conditional gen modeling(43)
MMPBoost(42) InfoGAIL(42) Identifiability in IRL(42) Diffuser(42) GAN landscapes(42)
```

### 🔴 D 티어 (20) — 이론/주변/타도메인 (참고)

```
Bayesian IRL(41) MAP Bayesian IRL(41) DualDICE(41) Misspecification in IRL(41) Offline IRL 해법개념(41)
Hybrid RL Hy-Q(40) Transferable Rewards 학습보장(40) VLM zero-shot reward(40)
Apprenticeship LP(39) IRL+DQN driving(39) Reward Identification(39) InstructGPT/RLHF(39) Decision Transformer(39)
Transfer&IRL 샘플효율(38) IfO inverse-dynamics(38) AlgaeDICE(37) A Survey of IRL(37) Theoretical understanding of IRL(36)
Trajectory Transformer(32) DQfD(31)
```

---

## 6. 미채점 추가 발견 39편 — 일부는 **매우 중요** (제목 정규화 불일치로 채점 매칭만 누락)

특히 도메인/접근에 직결되는 것들(보고서 related-work에도 필수):

- **IQ-Learn (2021, NeurIPS)** — inverse soft-Q, 단일 Q로 reward+정책 동시. offline 변형 존재. **사실상 S/A급, 강력 후보 → 2차 채점 1순위.**
- **OTR: Optimal Transport for Offline IL (2023)** — OT로 expert까지 거리=reward → relabel. TROFI류 실용 라인.
- **Inverse RL without RL (2023)**, **SubIQ (2024, suboptimal offline IQ)**, **A Simple Solution for Offline IfO (2023)**, **Primal Wasserstein offline IfO (2024)** — offline IRL 최신 실전군.
- **레이싱 도메인 직결**(보고서 인용 필수): **GT Sophy: Outracing GT champions (2022, Nature)**, **Automated Reward Design for Gran Turismo (2025)**, **BeTAIL (2024, 인간 레이싱 AIL)**, **High-speed racing trajectory-aided DRL (2023)**, **F1tenth Offline RL (2024)**.
- 토대/보조: **CQL (2020)**, **AWR (2019)**, **Ng 1999 potential-based shaping**(EPIC 등가류의 근거), Ng&Russell 1998 확장초록.
- 운전 IRL: Naturalistic driving IRL(2021), Predictive planning IRL(2023). 서베이: Arora&Doshi(2021), Osa(2018).

> → 원하면 이 39편 **2차 채점 패스**로 점수까지 채워 넣겠다(IQ-Learn·OTR·레이싱군 포함).

---

## 7. 역사 계보 (옛날 → 프론티어, 한눈)

- **2000–2008 토대**: Ng&Russell(LP-IRL) → Abbeel&Ng(apprenticeship) → Ratliff(MMP) → **Ziebart MaxEnt IRL**(지금도 뿌리) · Bayesian IRL.
- **2011–2016 딥/연속화**: GPIRL·Relative-Entropy → **Deep MaxEnt IRL**(2015) → **GCL**(2016, GAN-IRL 연결) → **GAIL**(2016, IL을 분포매칭으로).
- **2017–2020 adversarial+선호+offline 태동**: **AIRL**(전이 reward)·InfoGAIL·VAIL → **DRLHP·T-REX·D-REX**(선호·랭크) → ValueDICE·SMODICE·ORIL(offline IL).
- **2021–2023 offline IRL 본격**: **IQ-Learn · CLARE · OPIRL · CSIL · Preference Transformer** · DICE 확장 · **IQL→online finetune** · Cal-QL/RLPD(off→on).
- **2024–2025 프론티어**: **TROFI · OLLIE · UNIQ · SubIQ** · diffusion-reward(DRAIL/Diffusion Reward) · causal-invariant reward · VLA/VLM reward · GT Sophy 자동 reward 설계.

---

## 8. 함정·주의 (실패를 미리 막는 체크리스트)

1. **보상 비식별성(non-identifiability)**: 보상은 potential shaping + 양수 스케일링까지만 복원됨. 학습 보상이 멀쩡해 보여도 다른 정책을 유도할 수 있다 → 비교는 **EPIC**(등가류 불변)으로, 보상 크기 직접 비교 금지.
2. **occupancy 과적합**: SMODICE·DemoDICE·f-IRL·SEABO·ORIL·CLUE는 느린 데모 분포를 모방하게 만든다 → **1차 보상으로 쓰지 말 것**(보조/베이스라인만).
3. **OOD 외삽오차**: 데이터가 충돌위주·완주 희소 → 빠른 정책이 가는 바로 그 상태에서 보상이 가장 안 믿음직. **보수성(CLARE/bi-level MLE) 또는 IQL expectile in-sample** 필수.
4. **폐루프 covariate shift**: 스텝 예측이 정확해도 닫힌 루프에선 발산(Diffuser 라인 전례) → **offline 지표로 못 잡음. 짧은 online 폐루프 점검 필수**, K-step/replan 지평 중요.
5. **online 불안정**: adversarial min-max는 spectral norm/gradient penalty 없으면 붕괴. 신규 online 판별자는 사전학습 정책을 unlearn → **OLLIE 정렬 초기화/CSIL 일관성** + off-policy 갱신 선호.
6. **sim 샘플비용**: 롤아웃 수집이 cap당 수시간 → online은 타이트하게, **replay 재사용(off-policy-AIRL/OPIRL)**으로 on-policy GCL/AIRL 회피.
7. **action harmonization**: V_MAX 재정규화를 **랭킹 전에**, 그리고 **offline·online 양쪽**에서 유지하지 않으면 cross-cap return과 데모-롤아웃 보상 비교가 조용히 깨진다.
8. **랭킹 미스캘리브레이션**: D-REX 노이즈랭크/TROFI return랭크는 "비교 가능한 궤적" 가정 → 랭킹이 쓰레기면 보상도 쓰레기. held-out 순서쌍으로 검증.
9. **전이성은 검증 전엔 주장일 뿐**: disentanglement 정리만 믿지 말고 map_easy3 실측(데모 재수집 없이)으로 입증.

---

## 9. 한계·재현 정보

- 근거: 멀티에이전트 워크플로우(`offline-irl-litreview`), 에이전트 58개, 웹툴 호출 732회, 약 23분, 토큰 ~1.76M.
- 수집 raw 200편 → 중복제거 188편(채점 149, 미채점 39). 점수는 **이 프로젝트 적합도** 기준(학문적 중요도 아님). LLM 패널 채점이라 무오류 아님.
- 산출물 보존: `scratchpad/irl_full.md`(149편 전체 reason·적용·subscore), `tasks/wd5l13krs.output`(원시 JSON).

---

## 10. 다음 액션 (제안)

- [ ] (선택) 미채점 39편 **2차 채점**으로 점수 보강 — 특히 **IQ-Learn, OTR, 레이싱군(GT Sophy/BeTAIL/F1tenth-offline-RL)**.
- [ ] **추천 스택 상세 설계 문서**(002): 데이터 랭킹 구성(cap별·1랩조각), reward MLP 아키텍처, `f1tenth_data.py` 보상 훅 변경점, EPIC 게이트, 평가 게이트(2랩 결정론).
- [ ] (구현은 사용자 명시 지시 후) D-REX/TROFI 보상학습 PoC → 기존 IQL relabel → 폐루프 평가.

> ※ 구현·수집·학습·commit/push는 모두 **사용자 명시 지시가 있을 때만**. 본 문서는 조사·계획 정리까지다.
