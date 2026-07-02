# 003 — IRL(Topic-2) 계획 개정·확정 (001 + 002 반영 + 독립검증)

> 2026-07-02. [[001-irl-survey-and-plan]](원안) 을 [[002-irl-survey-adversarial-review]](적대적 검수) 의 수용된 지적으로 개정한 **확정 계획**.
> 추가로, 개정에 넣을 기술 주장 9건을 **독립 웹검증**하고 미채점 핵심 5편(OTR·IQ-Learn·SubIQ·IRL-without-RL·ReCOIL)을 채점해 반영.
> 원칙: 엄밀하되 작성자가 다시 읽어도 이해되게. append-only. 한글.
> 원시 채점표(149편 전체 reason)는 옆에 영구 보존: `new_plan/_data-irl-full-scores.md` (002 M5 해소).

---

## 0. 이 문서가 하는 일 (한눈)

001은 "고전→프론티어 149편 채점 + offline vs online + 실행계획"이었다. 002가 적대적으로 깨보니 **문헌·인용은 견고(환각 0건)**했으나 **① 패러다임 프레이밍이 이 repo의 입증 결과와 모순, ② 검증된 offline 레버(rc) 누락, ③ 상위 랭킹 자기모순, ④ AIRL/EPIC 기술 과장**이 나왔다. 003은 그걸 다 고친 확정판이다. **핵심 변화: 결론이 "offline→online 하이브리드"에서 "offline-FIRST, online은 조건부 최후"로 이동**한다 — 이 프로젝트의 실측 증거가 그쪽을 가리키기 때문.

---

## 1. 패러다임 확정 — offline-FIRST (online은 조건부 최후 분기)

**바뀐 결론**: 001의 TL;DR("순수 offline은 폐루프를 못 잡는다 → online 필수")은 **틀린 일반화**였다. 근거:

- 이 repo의 **입증된 성과 자체가 순수 offline model-free IQL의 폐루프 2랩 완주(34.32s)** 다. 학습 중 환경 상호작용 0.
- 001이 근거로 든 Diffuser 실패는 **model-based world-model의 자기회귀 오차누적**이지 offline 학습 일반의 성질이 아니다. IQL은 model-free라 배포 때 학습모델을 롤아웃하지 않으므로 그 오차채널이 없다.
- 유일했던 폐루프 실패(lap2 covariate shift)의 **검증된 해법은 online이 아니라 offline rc(random-centerline) 상태커버리지**였다.

**진짜 위험(문서의 새 thesis)**: *학습된* 보상은 손수 설계한 보상과 달리 **빠른 정책이 가는 OOD 상태에서 오설정될 수 있고, IQL expectile는 OOD 행동만 막지 오설정 보상은 못 막는다.** → 처방은 online이 아니라 **(a) 보상 offline 재검증 + (b) 무료 폐루프 *평가* 게이트 + (c) 실패 시 offline 레버(rc/보상 재적합) 먼저**.

**중요 구분(001이 뭉갠 것)**: **폐루프 *평가*(frozen-policy 롤아웃, 공짜, 학습 없음) ≠ online *finetune*(비쌈, 학습 있음).** 폐루프를 "보려면" online 학습이 필요한 게 아니다.

> 결론 한 줄: **offline-FIRST가 기본값. online IRL(AIRL 등)은 패러다임이 아니라, offline 레버를 다 소진했을 때의 조건부 최후 분기.** (001이 online을 "필수"로 과대 프레이밍했다 — 사용자 직관 "offline→online이 보편적"은 학계 일반론으론 맞으나, *이 repo 증거*는 offline-first를 더 지지.)

---

## 2. 확정 실행계획 (개정 10단계)

굵게 = 001 대비 변경점.

1. **action harmonization 먼저** — `a1'=(a1+1)(v_collect+5)/(v_common+5)−1`, v_common=20. 랭킹·relabel의 전제(009에서 load-bearing 입증).
2. **랭킹 구성 — "공짜" 아님(정직화, 002-M2).** 순간속도순(cap20 최고) ≠ 궤적 return순(충돌 cap20이 완주 cap10보다 *나쁨*). **충돌 cap15/20 에피소드가 완주 cap10 위로 랭크되면 안 됨.** 랭킹은 harmonization + γ=0.999 + 충돌페널티 −50로 계산한 return에 묶고, **held-out 순서쌍 검증을 핵심 리스크로 승격.**
3. **offline 보상 학습 — 두 경로 중 택1(또는 병행):**
   - **(경로 A, 랭크기반)** T-REX/D-REX/TROFI 계열. ⚠️ **D-REX는 순수 offline 아님**(BC 정책을 sim에서 노이즈 롤아웃해 자동랭크 → hybrid). "로그행동 교란 offline 근사"는 **미검증**(단조성 근거는 pi_BC≫random interpolation이라 고정상태 교란으론 안 나옴). f110 롤아웃이 저렴하니 hybrid로 수용하되 offline이라 부르지 말 것.
   - **(경로 B, OT기반) OTR — 완전 offline, drop-in 후보(§5, 2차채점 S~81).** expert 데모와 unlabeled(rc/coverage) 궤적을 **optimal transport로 정렬 → 정렬비용을 per-step 스칼라 보상으로 → relabel**. 알고리즘 무관(IQL 백엔드 OK), 공식 JAX 코드. ⚠️ **선결난제: OT cost를 raw 133D에 그대로 걸면 lidar128이 speed/steer/pose(state5)를 수치적으로 압도** → cost를 정규화/저차원 사영(위치·진행률·속도) 위에서 정의해야 함(feature weighting이 성패 좌우, 004 설계). '드롭인'이라도 통합난도가 낮은 건 아님.
4. **보상 offline 검증 — EPIC은 "전이 게이트"가 아니라 "spurious-feature sanity 체크"(격하, 002-C4).** EPIC(ICLR 2021)은 두 보상 사이 pseudometric이고 regret 상한은 *참조보상* 대비만 유효 — IRL엔 정답보상이 없다. proxy와의 비교는 "내 proxy와 닮았다"만 증명. STARC(ICLR 2024)는 EPIC 상한이 L2-norm 정규화 때문에 vacuous해질 수 있음을 보임. → 보상이 트랙별 spurious feature에 붙었는지 거친 점검용으로만.
5. **보수성(OOD) — CLARE 아이디어만, world-model 요구는 지양(002-C3).** CLARE는 133D 학습 동역학모델 위에 서는데 *그게 Diffuser가 실패한 바로 그것*이다. → **동역학모델 대신 IQL expectile(in-sample)로 OOD 행동을 막고**, 보상 보수성은 unlabeled coverage로 support를 넓히는 방식으로 대체.
6. **IQL relabel(드롭인)** — 코드 그대로, 데이터 보상만 학습보상으로 교체. (검증된 파이프라인.)
7. **f110 폐루프 *평가* 게이트(무료, frozen-policy)** — 2랩 결정론. online 학습 아님. 이게 offline로 못 보는 걸 보는 핵심 단계.
8. **★게이트 실패 시 offline 진단 분기 먼저(신규, 002-C2).**
   **먼저 판정한다(두 실패모드는 증상이 겹칠 수 있음 — 판정 프로토콜 필수, 004 상세화):** 충돌지점 상태가 데이터 support **밖**인가? (009/`003-crash-diagnosis` 방식: 충돌위치·횡오프셋 %ile·커버리지 tail 계측) → **coverage gap**. support **안**인데 학습보상이 손설계 proxy 대비 **순위역전/EPIC 이상**(step4 재실행)을 보이는가? → **reward 오설정**.
   - (i) **coverage gap** → **rc 커버리지 추가(이 repo에서 검증된 해법).**
   - (ii) **reward 오설정** → 보상 offline 재적합/재검증(랭킹·OT cost 재조정).
   - (iii) **(i)·(ii)를 다 소진한 경우에만** → step9 online.
9. **(조건부 최후) 짧은 online finetune** — OLLIE식 정렬 init 또는 CSIL coherent reward로 사전학습 정책 붕괴 방지, off-policy-AIRL/OPIRL로 replay 재사용, adversarial 항은 spectral norm. **기본이 아니라 최후.** (m3: IQL의 online 연장은 *가치 목적함수*엔 native지만, 판별자 결합 시 적대 불안정 유입 — "config 변경 수준" 아님.)
10. **새 트랙 전이 검증(실측만)** — map_easy3에서 데모 재수집 없이 폐루프 완주율 보고. EPIC를 전이 게이트로 과신 금지. 필요시 causal-invariance 정규화.

---

## 3. 추천 스택 확정

**offline 보상 라벨링 → IQL relabel → 폐루프 평가 게이트 → (실패 시) offline 진단 → (최후) online.** 1차 보상은 **occupancy 매칭이 아니라** (a) 랭크기반 또는 (b) OT기반.

| 우선 | 방법 | regime | 왜 |
|---|---|---|---|
| **공동 1차** | **OTR** (ICLR 2023 Oral, 2303.13971) | **offline** | OT 정렬비용=보상 → relabel → IQL. **완전 offline, drop-in, 공식 JAX 코드.** 라벨/랭킹 불필요. ⚠️ **133D OT-cost 설계(lidar 지배 방지)가 선결난제 — 통합난도 '낮음' 아님**(§2 step3) |
| **공동 1차** | **D-REX**(CoRL 2019) / **TROFI**(RLVG 워크숍 2025) | off→on(hybrid) | 랭크로 explicit 보상, **데모 초월 외삽**. 라벨 불필요(D-REX 자동랭크). ⚠️ 롤아웃 의존 |
| 개념 코어 | **T-REX**(ICML 2019) | off→on | 랭크 pairwise 보상의 원조. 수동 랭크 필요 |
| 조건부/2차 | **AIRL**(ICLR 2018) | online | 전이 보상 잠재력 최강 **단 state-only g(s)에서만**(§4), online 비용·불안정. step9용 |
| 아이디어만 | **CLARE**(ICLR 2023) | offline | OOD 보수성 개념 차용, **133D world-model 요구는 지양**(§4·§5) |
| 대안 학습기 | **IQ-Learn**(B) / **ReCOIL**(B) | offline | 보상 제공자가 아니라 *IQL 대안*(§5) |
| 도구 | **EPIC**(sanity), **OLLIE/CSIL**(online 안정화) | — | §2 참조 |

> 001 대비 핵심 변화: **OTR 추가로 "완전 offline·드롭인" 경로가 생겼다** → 랭크기반의 hybrid 롤아웃 부담 없이 1차 가능. AIRL·CLARE는 "1차"에서 "조건부/아이디어"로 강등.

---

## 4. 기술 사실 정정표 (001의 오류 → 검증된 정정, 출처)

| # | 001의 서술 | 검증된 정정 | 출처 |
|---|---|---|---|
| T1 | AIRL `r=g(s,a)+h(s)`, "동역학 분리 전이 최강, 증명적" | **전이보장은 state-only `g(s)`에서만** 성립(행동의존 `g(s,a)`는 얽힘). 분해가능·(준)결정론 동역학 가정 필요. "증명적/최강" → "특정 가정 하 우수, 본 셋업 미검증" | Fu·Luo·Levine, ICLR 2018, 1710.11248 (※이 항목은 검증 에이전트 1건이 플레이스홀더 반환 → **저자 1차 논문 지식으로 직접 정정**, 신뢰도 표기) |
| T2 | (암묵) off-policy-AIRL로 비용↓ | **off-policy(SAC)는 전이보장을 약화**: SAC의 **entropy 항**이 복원 보상을 학습정책/동역학에 결합해 전이성을 깨는 방향(importance sampling 아님). ※정확한 결합 수식은 근거가 withdrawn preprint라 **미확정**(방향만 신뢰). 해법=PPO로 보상복원 + SAC로 정책만 | Zhang et al. 2403.14593 (2024, **withdrawn** → 시사적) + **동일저자그룹** 후속 2410.07643 (독립 후속 아님) |
| T3 | EPIC "2022 ICLR", "전이를 학습 전에 예측·regret 보장" | **ICLR 2021**. pseudometric, **참조보상 대비만 유효**. STARC(ICLR 2024)는 EPIC 상한이 **L2-norm 정규화**(coverage 상수 아님!) 때문에 vacuous·양측경계 실패 가능. → "sanity 체크"로 격하 | EPIC 2006.13900; STARC 2309.15257 |
| T4 | D-REX "offline 1차", "행동교란 offline 근사 가능" | **hybrid(sim 롤아웃 필요)**. offline 근사는 **미검증**(단조성 근거는 pi_BC≫random) | Brown et al. CoRL 2019, 1907.03976 (공식코드 확인) |
| T5 | TROFI "RLC" | **RLC 2025 *RLVG 워크숍*(main track 아님)**, 백엔드 **TD3+BC**(IQL 아님), 보상=랭크/선호 | 2506.22008 (arXiv comment 확인) |
| T6 | occupancy 매칭 "BC처럼 느림 따라감" | DICE는 BC 아님(**Fenchel dual + V + 밀도비 = 스티칭**, 단 최종추출은 weighted-BC). 천장=**단일 expert 정확매칭 한정**("any"는 과장 — 혼합/success-example occupancy면 깨짐). f-IRL은 explicit 전이 state-marginal 보상 | SMODICE 2202.02433; f-IRL 2011.04709; GAIL Prop 3.1(1606.03476); Syed 2008; Puterman Thm 6.9.1 |
| T7 | 003(A)="GAIL식 occupancy, 느린 데모 모방" | **GAIL 아님 → ORIL/PU 정적 판별자(occupancy 편향)**. 003은 *빠른 Dreamer*를 expert로 권장하므로 "느린 데모" 프레이밍 안 맞음. **단 바닥 결론(랭크기반>판별자, 초월엔)은 유효**(판별자 보상은 expert에서 최대라 초월 gradient 없음) | 003 본문; occupancy-ceiling 검증 |

---

## 5. 랭킹 갱신 (신규 채점 반영 + 정직화)

**2차 채점 5편** — 001의 **1차 149편과 별개 워크플로우**(003 verify-and-score)에서 같은 8축 루브릭·가중치로 채점, `_data-irl-full-scores.md` **부록(2차 채점)**에 보존. ⚠️ **두 채점 패스를 구분해 읽을 것**(1차 산출표엔 이 5편이 없다). LLM 패널 점수라 **밴드만 신뢰(±3~5)**:

| 방법 | 밴드 | regime | 판정 |
|---|---|---|---|
| **OTR** (Optimal Transport, ICLR 2023 Oral) | **S (~81)** | offline | OT보상→relabel→IQL, 완전 offline, 공식 JAX. **랭크기반과 동급 1차 후보**(최상위 단정 아님). ⚠️ 133D OT-cost 설계 선결(§2·§3) |
| IQ-Learn (offline 사용) | B | either | **M4의 "S/A급" 가정은 채점상 부정.** 보상이 Q에 암시적 → relabel용 보상 제공자 아님, *IQL 대안 학습기*. 중요하나 역할이 다름 |
| IRL-without-RL (ICML 2023) | B | online | explicit 보상이나 online RL 서브루틴 필요. "offline 라벨링 선호"의 이론적 근거로만 |
| ReCOIL (ICLR 2024 Spotlight) | B | offline | dual RL offline 모방, policy-only(보상 아님). IQL 대안 베이스라인 |
| SubIQ (ICML 2024) | C | offline | 열등데모용 offline inverse-soft-Q. 데이터에 명확한 열등군 섞을 때만 |

**S티어 정직화(002-M6)**: LLM 패널 점수는 ±3~5 불확실. **S티어는 순서 없는 집합으로 취급** — 85>84>83 정렬 폐기. 실무 S급 = **{OTR, T-REX/D-REX(랭크)}**, 배포 1순위 = OTR 또는 D-REX. **AIRL(원점수 84)·CLARE(83)는 원시표 점수가 그대로 남아 있으나, §3 caveat(AIRL=online 비용·전이 미검증 / CLARE=133D world-model)로 "조건부"로 강등했다 — 점수 *재산정*이 아니라 *배치 강등*으로 처리했음을 명시**(002-C3의 "실제 점수 차감"은 원시 워크플로우를 재실행하지 않는 한 불가하므로 배치·caveat로 대체). 001 §5 전체 149편 랭킹은 `_data-irl-full-scores.md`에 보존; "188편 중 149편 부분 랭킹, 미채점 명명 방법이 채점된 것 능가 가능"으로 격하해 읽을 것.

---

## 6. 002 지적 처리 대장

| 지적 | 처리 | 반영 위치 |
|---|---|---|
| C1 Diffuser 혼동 | **수용** | §1 (offline-first로 재프레이밍, Diffuser 인용 철회) |
| C2 rc 누락 | **수용** | §2 step8 진단분기(i) |
| C3 상위랭킹 자기모순 | **수용** | §3(AIRL/CLARE 강등), §5(S 무순서) |
| C4 EPIC 자기모순·오용 | **수용** | §2 step4, §4-T3 (sanity로 격하) |
| M1 D-REX offline 오표기 | **수용** | §2 step3, §4-T4 |
| M2 "랭킹 공짜" | **수용** | §2 step2 |
| M3 AIRL 공식·과확신 | **수용** | §4-T1·T2 |
| M4 IQ-Learn/OTR 미채점 | **수용+반전** | §5 (채점 결과: OTR=S, **IQ-Learn=B**로 M4 가정 일부 부정) |
| M5 원시산출물 부재 | **수용** | `_data-irl-full-scores.md`로 영구 보존 |
| M6 false precision | **수용** | §5 (S 무순서·불확실성 표기) |
| M7 occupancy 메커니즘 | **수용** | §4-T6 (프로즈 정정, 결론 유지) |
| C3.1 T-REX vs D-REX "치명" | **부분수용(심각도 하향)** | §3 (점수≠배포우선순위 명시, 배포1순위=D-REX/OTR) |
| §6c 003(A) 비판 | **수용(내 비판이 절반 틀림)** | §4-T7 |
| m1~m6 사실정정 | **수용** | §4 (EPIC 2021, TROFI 워크숍, SSRR 0.95 "추정") |

---

## 7. 남은 위험·한계 (정직 고지)

- **검증 실패 1건**: AIRL state-only(T1) 검증 에이전트가 플레이스홀더("test")를 반환 → 그 항목은 워크플로우가 아니라 **내 1차 논문 지식**으로 정정했다(신뢰도 높음이나 재확인 권장).
- **일부 인용은 시사적**: T2의 Zhang(2403.14593)은 **withdrawn preprint** — 단정 말고 후속(2410.07643)과 함께 시사적으로만. SSRR 0.95는 "추정(출처 미확인)".
- **미실측 예측**: "CLARE가 Diffuser처럼 실패한다", "D-REX 행동교란이 랭킹을 깬다", "OTR 133D에서 잘 된다"는 **이론·문헌 기반 예측**이지 실측 아님. 코드 실행 후 재확인 필요.
- **랭킹 정밀 순서**는 LLM 패널·부분채점(149/188)의 산물 → 티어/밴드로만 신뢰.
- **원시 채점표** 보존은 `_data-irl-full-scores.md`(이번 이관). 워크플로우 원시 JSON은 세션 scratchpad에 있어 휘발 가능.

---

## 8. 다음 액션

- [ ] (사용자 지시 시) **OTR PoC** — 완전 offline이나 **OT cost feature-space 설계(133D lidar 지배 방지) 선행 필요**: 정규화/저차원 사영 위 cost 정의 → expert 데모 vs rc coverage로 OT 보상 라벨 → 기존 IQL relabel → 폐루프 평가 게이트. (cost 정의는 OTR 코드 `ethanluoyc/optimal_transport_reward`에서 직접 확인)
- [ ] (병행 후보) **D-REX/TROFI 랭크기반** PoC — 단 harmonization+return랭크 검증(§2 step2) 선행.
- [ ] **004 상세 설계** — 데이터 랭킹/OT cost 구성, 보상 MLP·`f1tenth_data.py` 훅 변경점, EPIC sanity, 진단분기 로직, 평가 게이트.
- [ ] IRL-without-RL의 "expert-state reset"이 f110에서 가능한지 확인(online 최후분기 비용 절감용).

---

## 9. 003 자체 critic 재검수 반영 (자기승인 금지 규약)

003 초안을 별도 critic 에이전트로 재검수(002와 독립 lane)한 뒤 아래를 반영했다. 리뷰 루프 투명성을 위해 남긴다.

- **[반영] OTR 133D OT-cost 설계난제** — §2 step3·§3·§5·§8에 "raw 133D면 lidar128이 state5를 압도 → 정규화/저차원 사영 위 cost 정의 선결" 명시, integration '최적/드롭인' 과대표현 하향.
- **[반영] step8 판정기준 부재** — coverage gap vs reward 오설정을 구별하는 **판정 프로토콜**(support 계측 / 보상 sanity 재실행) 추가.
- **[반영] §4-T2 수식 과단정** — withdrawn 근거를 넘어 단정한 `γα·E[log π]` 수식을 "방향만 신뢰, 결합형태 미확정"으로 약화. 후속 2410.07643이 **동일저자그룹**(독립 아님)임 명시.
- **[반영] false-precision 재발** — 2차 채점 5편을 정수점수→**밴드(S/B/C)**로, "신규 최상위" 표현 삭제(무순서 원칙과 정합).
- **[반영] C3 잔여** — AIRL(84)·CLARE(83) 원점수는 재산정 불가(원시 워크플로우 재실행 필요)하므로, "점수 차감"이 아닌 **caveat·배치 강등**으로 처리했음을 §5에 명시.
- **[이견·정정] critic의 치명지적 "OTR S\|81은 원시표에 없는 사후조작"은 절반만 맞다.** OTR 점수는 *조작이 아니라* **2차 워크플로우(task wst7qh3bn)의 실제 8축 채점**(rew8 con8 reg9 int10 tra6 cod8 mat6 par9 → 가중합 81)이다. critic은 1차 산출(`_data-irl-full-scores.md`)만 봐서 그렇게 판단. **진짜 결함은 "그 2차 채점을 repo에 보존하지 않아 추적 불가"였던 것** → 데이터 파일에 **부록(2차 채점) 섹션을 append**해 감사 가능하게 해소하고, §5에서 "1차/2차 채점 패스 구분"을 명시했다.

> 검수 근거: critic 재검수(에이전트 1개, 웹툴 27회). 남은 열린질문: OTR OT-cost의 정확한 feature space는 1차 논문 PDF 미확인(신뢰도 MEDIUM) → 004에서 코드로 확정.

---

> ※ 본 문서는 계획 확정까지다. 코드 구현·수집·학습·commit/push/pull은 **사용자 명시 지시 시에만**. Dreamer/RL_project 독립성 유지, 로그·데이터 폐기 금지.
