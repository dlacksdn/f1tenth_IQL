# 002 — 001(IRL 전수조사·계획) 적대적 검수

> 2026-06-29. [[001-irl-survey-and-plan]]에 대한 **적대적 critic** 검수. append-only. 한글.
> 입장: 칭찬이 아니라 "문서를 깨는 것". 동의하는 부분도 먼저 공격한 뒤 "살아남았다"고만 적는다(steelman → stress-test).
> 근거: 멀티에이전트 검증 워크플로우(에이전트 11개, 웹툴 호출 117회, 논문 20여 편 + 기술주장 5건 + thesis 2건 교차검증)
> + 검수자 직접 확인(프로젝트 문서·디스크·공식 인용).

---

## 0. 한 줄 총평 + 최종 판정

**한 줄**: 문헌의 폭과 인용 진위는 우수하다(환각 0건). 그러나 **중심 논거("순수 offline은 폐루프 발산을 못 잡는다 → offline-then-online이 정답")가 이 프로젝트의 *입증된* 결과와 정면으로 모순**되고, **랭킹 상위 4개(T-REX·AIRL·CLARE·EPIC)가 문서 자신의 결론과 충돌**하며, **점수의 근거(원시 산출물)는 디스크에 부재해 검증 불가**다. 단, 실행계획의 *뼈대*(조화 → 랭크기반 보상 → IQL relabel → 폐루프 게이트)는 공격을 견디고 살아남았다.

**최종 판정: 조건부 채택 (조건부 채택).**
- 반려가 아닌 이유: 인용이 실재하고, 추천 스택의 골격이 *검증된 IQL 레시피와 동형*이라 폐기할 근거가 없다.
- 채택이 아닌 이유: TL;DR의 패러다임 단정, 상위 랭킹의 자기모순, 두어 건의 기술적 오류, **검증된 offline 해법(rc 커버리지)의 누락**은 의사결정을 오도할 수 있어 *반드시* 고쳐야 한다.
- 조건: §9의 구체 권고(특히 C1·C2·C3·C4 수정)를 반영하면 채택 가능.

---

## 1. 검증 방법 (무엇을 어떻게 확인했나)

**읽은 파일**: 001(검수 대상), 003-irl-plan(선행 설계), goal/001-goal, implementation/009-final-summary, report/004-report-draft(§4.2 Diffuser 실패 원문), additional_experiment/001-no-completion-stitching·002-rconly-ablation, CLAUDE.md.

**워크플로우 교차검증**(`irl-adversarial-verify`, 11 에이전트):
- 논문 진위(웹서치): TROFI/OLLIE/CSIL/D-REX/CLARE/MLIRL/PrefTransformer/AIRL/EPIC/T-REX/IQL/SSRR/ORIL/GT Sophy/BeTAIL/F1tenth-Offline-RL/High-speed-racing/Auto-Reward-GT/IQ-Learn/OTR — claimed vs actual(연도·venue·arXiv·저자·코드repo).
- 기술주장: EPIC의 실제 보장·요구조건, D-REX/SSRR의 offline성, DICE/occupancy 메커니즘, 003(A)=GAIL인가, AIRL의 off→on·전이보장.
- thesis 스트레스: offline-then-online 과확신 / 랭킹 내부일관성.

**검수자 직접 확인**: ① `scratchpad/`·`tasks/` 디렉토리 및 `irl_full.md`·`wd5l13krs.output` 전체검색 → **0건(부재)**. ② 001 line 57·109의 AIRL 공식 `r=g(s,a)+h(s)`를 "disentangled"로 명기. ③ 001 line 89의 EPIC "2022, ICLR" 표기. ④ 보고서 §4.2 "Diffuser 실패 = **병목은 데이터가 아니라 모델**" 원문.

---

## 2. 치명적 결함 (Critical)

### C1. 중심 논거가 자기네 입증 결과와 모순 — "순수 offline은 폐루프를 못 잡는다" 〔치명 · 신뢰도 높음〕

- **001의 주장(인용, TL;DR/line 13, §3/line 56·61)**: "순수 offline만으로는 폐루프(closed-loop) 발산을 못 잡는다. … offline 지표(EPIC 포함)로는 절대 못 잡는다 → 짧은 online 폐루프 점검이 필수. (Diffuser 라인이 이미 당함)"
- **반박 근거**:
  1. 이 프로젝트의 **입증된 성과 자체가 순수 offline IQL의 폐루프 2랩 완주**(34.32s, 009 §0·§1)다. 학습 중 환경 상호작용 0.
  2. 001이 근거로 드는 Diffuser 실패는 **world-model(생성형 궤적 예측) 실패**다. 보고서 §4.2 원문: *"데이터가 같고 모델만 다른데 결과가 갈렸다 → 병목은 데이터가 아니라 **모델**이다(느린 생성과 코너 정밀도의 한계)."* 즉 폐루프 발산은 **모델기반 자기회귀 롤아웃이 자기 오차를 누적**해 생기는 것이지, offline *학습* 일반의 성질이 아니다.
  3. IQL은 **model-free**다 — 배포 시 학습된 모델을 롤아웃하지 않고 실제 sim을 질의하므로 Diffuser식 오차누적 채널이 없다. 그래서 같은 offline인데도 폐루프를 통과했다.
- **영향**: 문서 전체의 패러다임 결론("online이 필수")이 *잘못된 일반화* 위에 서 있다. 한 모델클래스(Diffuser)의 실패를 offline 전체의 법칙으로 승격시켰다.
- **수정안**: TL;DR을 "Diffuser는 model-based world-model 실패이며 model-free IQL엔 전이되지 않는다. 순수 offline은 폐루프를 **잡을 수 있다**(우리 IQL이 증명). online이 정당화되는 유일한 경우는 *학습된 보상의 오설정*을 offline 지표로 검증 못 할 때다(C2·M의 진짜 알맹이)"로 교체. Diffuser를 offline-RL의 증거로 인용하지 말 것.

### C2. 폐루프 해법 오진 — 검증된 offline 레버(rc)를 9단계에서 누락 〔치명 · 신뢰도 높음〕

- **001의 주장(인용, §3 9단계 step7→8/line 73·74)**: "스텝은 맞는데 폐루프가 발산하면(=알려진 Diffuser 실패) **online 수리**로. … 짧은 online finetune."
- **반박 근거**: 이 프로젝트는 **이미 IQL 폐루프 실패(lap2 covariate shift)를 겪었고, 그 검증된 해법은 online이 아니라 offline이었다.** random-centerline(rc) 상태커버리지로 트랙 전역 진입상태를 덮어 폐루프 복구를 학습 → 3/4 seed 34.3~34.5s 2랩 완주(009 §2·§4, additional_experiment/001 §3·§5, 002 §5). 더구나 그 진단(횡방향 오프셋 98~99%ile 미커버 tail)도 **정적 데이터에서 offline로 측정**됐다. 즉 폐루프 발산의 검증된 레버는 **offline 데이터 커버리지**다. 001은 이 레버를 9단계 어디에도 넣지 않고 곧장 online을 처방한다 — 이미 가진(그리고 더 싼) 해법의 재발명.
- **영향**: 실행계획이 "online 수리"로 직행해, 이 코드베이스에서 *실제로 통한* offline 분기를 건너뛴다. online은 sim 비용(cap당 수시간)·적대 불안정을 새로 떠안는다.
- **수정안**: step7 게이트 실패 시 **진단 분기**를 step8 앞에 삽입 — (i) coverage gap이면 offline rc 커버리지 추가(검증됨), (ii) reward error면 보상 offline 재적합/검증, (iii) 둘 다 소진된 경우에만 online. online을 기본이 아니라 *최후*로 강등.

### C3. 랭킹 상위 3개가 문서 자신의 결론과 충돌 〔치명 · 신뢰도 높음〕

- **001의 주장(인용, §5 S티어)**: "T-REX | 85 | 지금 당장 1차 후보", "AIRL | 84 | off→on", "CLARE | 83 | 완전 offline IRL … 동역학모델 적합이 부담".
- **반박 근거**(세 건 모두 동일 문서 내부와 모순):
  1. **T-REX #1(85) vs 실제 1차 추천 D-REX(A, 79)**: §4·9단계는 D-REX를 "1차 실용 추천"이라 하면서 §5는 D-REX를 79(A티어)로 매기고 T-REX를 85(S티어, "1차 후보")로 올린다. **최고점 방법이 실제로 1순위로 안 쓰는 방법**이다. 랭킹이 의사결정을 안 끌면 랭킹이 아니다(6점·1티어 불일치).
  2. **AIRL #2(84, off→on) vs §3 "순수 online 단독 최악(△)"**: 같은 문서가 AIRL을 §3에선 "순수 Online … △ 단독은 최악(on-policy 비용=킬러, 기존 offline 투자 전부 폐기)"으로, §5에선 2등(84)·"off→on"으로 적는다. vanilla AIRL엔 offline 사전학습 단계가 없다 — "off→on" 태그는 *방법의 성질*이 아니라 001의 *희망적 배포안*일 뿐이라, CLARE/IQL의 진짜 off→on과 같은 열에 두면 사과(沙果)-오렌지 비교다.
  3. **CLARE #3(83)이 133D world-model 요구 = Diffuser가 실패한 바로 그것**: CLARE는 학습된 동역학모델 위에 선다. 이 프로젝트의 Diffuser(world-model)는 폐루프 실패했고 §5는 그 Diffuser를 42(C)로 매긴다. 그런데 *같은 133D world-model*을 요구하는 CLARE는 3등(83, S)이다. 위험을 산문(line 88 "부담")으로만 인정하고 **점수(continuous_highdim 0.12)에서 차감하지 않았다.**
- **영향**: §5는 "어떤 방법을 쓸지" 고르는 *결정 도구*인데, 상위 3개가 전부 문서 자신의 권고/증거와 어긋난다 → 랭킹의 결정 효용이 무너진다.
- **수정안**: (a) T-REX와 D-REX를 동급 S로 묶고 "deploy 1순위=D-REX(라벨 불필요·데이터 즉시랭크)"를 명시하거나 D-REX 가점. (b) AIRL에 §3의 regime_fit/integration_ease 페널티를 *실제로* 부과해 S티어에서 내릴 것(전이 잠재력은 transfer 축 점수로만 표기). (c) CLARE를 continuous_highdim·code_repro에서 강하게 감점(Diffuser 실패 증거를 점수 옆에 명기), OOD-비관화 *아이디어*만 차용하고 동역학모델 요구는 IQL expectile로 대체.

### C4. EPIC 자기모순 + 오용 — "전이를 학습 전에 예측"인데 동시에 "EPIC로도 절대 못 잡음" 〔치명 · 신뢰도 높음〕

- **001의 주장(인용)**: §3-step4/§4/§5(line 70·89·124) "EPIC distance로 sanity 기준(progress/곡률 proxy)과 비교 … **다른 동역학에서도 regret 상한을 보장** → 새 트랙 전이를 *학습 전에* 예측하는 게이트" ↔ §3/§8.4(line 61·206) "폐루프 covariate shift … **offline 지표(EPIC 포함)로는 절대 못 잡는다.**"
- **반박 근거**:
  1. **직접 모순**: 새 트랙 완주는 폐루프/covariate-shift 문제다. 한 문서가 "EPIC로 새트랙 전이를 학습 전에 예측"(§4)과 "EPIC 포함 폐루프 절대 못 잡음"(§8.4)을 동시에 말한다.
  2. **EPIC는 두 보상 사이의 거리**다(Gleave et al., ICLR **2021**, arXiv:2006.13900). regret 상한은 한쪽이 *ground-truth/reference reward*일 때만 의미가 있다(BAIR/DeepMind 설명: "reward 명세가 이미 풀린 과제에서 ground-truth reward를 줘야 성립"). IRL은 정의상 정답 보상이 없다 — "progress/곡률 proxy와 비교"는 *손수 만든 목적*을 다시 들여오는 것이고, 작은 EPIC는 "학습보상이 내 proxy와 닮았다"만 증명한다(proxy를 믿으면 IRL을 안 하면 되고, 안 믿으면 닮음은 검증이 아니다).
  3. STARC(Skalse/Jenner et al., ICLR 2024, arXiv:2309.15257)는 EPIC의 regret 상한이 coverage 상수 K(D) 때문에 *느슨/공허*해질 수 있고 worst-case regret의 하한도 못 준다고 보임 → "regret 상한을 보장"은 *거대 상수까지*만, 그것도 *참 보상* 대비에서만이다.
- **영향**: 9단계의 step4(정책 학습 전 보상 검증)와 step9(전이 게이트)가 과신된 도구 위에 선다.
- **수정안**: EPIC를 "전이 예측 게이트" → "**보상이 spurious feature에 안 붙었는지 거친 sanity 체크**"로 격하. reference가 ground-truth가 아닌 proxy임을 명시. §3·§4·§5를 §8.4("EPIC 포함 못 잡음")와 일치시키고, 전이는 §8.9대로 **map_easy3 폐루프 실측**으로만 주장.

---

## 3. 중대 결함 (Major)

### M1. D-REX "offline" 오표기 + "로그 행동 교란으로 offline 근사"는 검증 안 된 즉흥책 〔중대 · 신뢰도 높음〕
- **인용**(§0 "랭크 기반 **offline** 보상학습(D-REX/TROFI)"; §4/line 84 "노이즈 롤아웃 … **혹은 로그 행동 교란으로 offline 근사 가능**").
- **반박**: D-REX는 데이터 생성 단계가 **본질적으로 offline이 아니다** — BC 정책에 노이즈를 주입해 **환경에서 롤아웃**해 자동 랭킹을 만든다(노이즈↑ → 정책이 점점 나쁜 *상태분포*로 표류 → 단조적으로 낮은 return). "로그 행동 교란"은 이 단조성을 깬다: **고정된 로그 상태열**에 행동만 흔들면 상태분포 표류·누적이 없어(=001 스스로 §8.4에서 "offline로 못 잡는다"고 한 바로 그 폐루프 효과가 안 생김) 랭킹이 잘못 매겨진다 → §8.8 "랭킹이 쓰레기면 보상도 쓰레기". 출처 없는 즉흥 근사다. SSRR(arXiv:2010.11723, CoRL 2021)도 BC/AIRL 정책의 sim 롤아웃이 필수라 동일.
- **영향**: "offline 1차 추천"이라는 표기가 실제 hybrid 의존을 숨긴다. (단 §3/§5는 D-REX를 "off→on"으로 *옳게* 태그함 → 라벨 내부 불일치.)
- **수정안**: TL;DR/§4에서 D-REX를 "offline"이라 부르지 말 것(데이터 생성=짧은 sim 롤아웃, f110선 저렴하나 offline 아님). "행동 교란 offline 근사"는 삭제하거나 "별도 검증 필요(held-out 순서쌍)"로 강등.

### M2. "랭킹을 공짜로 구성"은 순간속도와 궤적 return을 혼동 〔중대 · 신뢰도 높음〕
- **인용**(§3 step2/line 68): "**랭킹을 공짜로 구성.** … cap5 < cap10 < cap15 < cap20 (속도/return 순)".
- **반박**: cap15/cap20는 **충돌 위주**(371/291 충돌, 003 §2). 충돌난 cap20 궤적은 직선 구간이 *빠르지만* 결과(return)는 완주한 cap10 lap보다 *나쁘다*. "속도순"(cap20 최고)과 "return순"(충돌 cap20이 최악)은 다른 순서인데 등치시켰다. 가장 어렵고 load-bearing한 단계(혼합 충돌/완주 데이터의 유효한 전순서)를 "공짜"라는 단어로 가린다.
- **영향**: 랭크기반 보상 전체가 이 전순서 위에 서므로, 틀리면 보상이 오염된다.
- **수정안**: "공짜" 삭제. 속도순과 return순을 분리 — 충돌 cap20 에피소드가 완주 cap10보다 위로 랭크되면 안 됨. cross-cap return을 비교 가능케 한 *검증된 레시피*(action harmonization + γ=0.999 + 충돌페널티 −50)에 랭킹을 묶고, held-out 순서쌍(§8.8)으로 검증을 *핵심 리스크*로 승격.

### M3. AIRL 전이보상 공식 오류 + "증명적 복원·전이 최강" 과확신 〔중대 · 신뢰도 높음〕
- **인용**(line 57·109): "AIRL: `r=g(s,a)+h(s)`, 동역학과 분리 … **동역학과 분리된 전이 reward를 증명적 복원** → 새 트랙 전이 최강".
- **반박**: AIRL의 disentangled(전이) 보장은 **state-only g(s)**에 대해서만 성립한다(판별자 f(s,s')=g(s)+γh(s')−h(s); "참 보상이 상태만의 함수이면 상수까지 복원"). `g(s,a)`(행동 의존)는 *얽힌(entangled)* 비전이 형태라 "동역학과 분리"라는 라벨과 모순이다. 게다가 보장엔 분해가능 동역학(rank(P−I)=|S|−1)·(준)결정론 가정이 붙고, Rethinking-AIRL(arXiv:2403.14593)은 이를 어기는 MDP, 그리고 **off-policy SAC로 바꾸면 entropy 항이 disentanglement를 깬다**고 보임 — 001이 비용절감용으로 기댄 off-policy-AIRL이 바로 그 위반이다. f1tenth(연속·행동의존 보상·off-policy finetune)에서 미검증인데 "증명적/최강"으로 단정. (001 §8.9가 "전이성은 검증 전엔 주장일 뿐"이라 스스로 헤지 → §3/§5의 무헤지 표현과 충돌.)
- **수정안**: 공식을 전이형 **g(s)**로 정정(또는 "행동의존이면 전이보장 깨짐" 명시). "증명적 복원/전이 최강"을 "특정 가정(state-only·분해가능·결정론) 하 전이 우수, 본 셋업에서 미검증"으로 약화.

### M4. IQ-Learn·OTR 미채점인데 "사실상 S/A급" → "149편 랭킹"의 권위 훼손 〔중대 · 신뢰도 높음〕
- **인용**(§6): "IQ-Learn … **사실상 S/A급, 강력 후보 → 2차 채점 1순위.**" (OTR도 "TROFI류 실용 라인".)
- **반박**: 랭킹의 권위는 "149편 전수"라는 데 있다. 그런데 저자 스스로 *한 번도 안 매긴* 방법이 현재 3편뿐인 S티어에 들 수 있다고 한다. 제목 정규화 불일치(§6)로 누락된 논문이 28편 A티어와 어쩌면 S티어를 능가한다면, 정밀 순서는 *어떤 논문이 우연히 매칭됐나*의 산물이다. IQ-Learn(NeurIPS 2021 Spotlight, **offline 변형 실재**)·OTR(ICLR 2023 Oral)은 둘 다 추천 파이프라인(랭크/OT→relabel→offline RL) **정중앙**이라 주변부 누락이 아니다.
- **수정안**: §5를 결정 근거로 쓰기 *전에* 최소 IQ-Learn·OTR·레이싱군 2차 채점을 돌리거나, §5를 "188편 중 149편 부분 랭킹"으로 명시 격하하고 §5 본문(§6 아님)에 "미채점 명명 방법이 채점된 것을 능가할 수 있음" 경고.

### M5. 점수 근거(원시 산출물)가 디스크에 부재 → 8축 subscore·149점 검증 불가 〔중대 · 신뢰도 높음〕
- **인용**(§0·§2·§9): "원시 산출물 보존: `scratchpad/irl_full.md`(149편 전체 reason·적용·subscore), `tasks/wd5l13krs.output`(원시 JSON)."
- **반박**: 검수자 직접 확인 — `scratchpad/`·`tasks/` 디렉토리 없음, `irl_full.md`·`wd5l13krs.output` 전체검색 0건. new_plan엔 001 본문뿐. 따라서 8축 분해·per-paper subscore·732 웹콜 provenance를 *전혀 들여다볼 수 없다* → 수치 랭킹은 현재 **검증 불가**다. (이 점은 검수 의뢰문도 사전 경고: "원시 산출물은 새 세션에서 접근 불가".)
- **영향**: 점수의 *타당성*을 "틀렸다"로 단정할 수는 없으나 "검증 불가"로 남는다. M6(false precision)과 결합해 §5의 정량 권위가 약해진다.
- **수정안**: 산출물을 001 옆에 복구/commit하거나 §0/§2/§9의 경로를 실제 위치로 갱신. 복구 전까지 §5를 "잠정 랭킹"으로 표기.

### M6. False precision — LLM 패널 점수인데 85/84/83 1점 차로 S티어를 순위매김 〔중대 · 신뢰도 중간〕
- **인용**: "T-REX 85 / AIRL 84 / CLARE 83", 79에 4편 동점(CSIL·D-REX·GenWM-MLE·PrefTransformer), S/A 경계 80/79.
- **반박**: §2·§9가 스스로 "LLM 패널이 루브릭으로 매긴 점수, 무오류 아님"이라 한다. LLM 루브릭 점수는 정수 해상도로 신뢰할 수 없는데 S티어 전체를 1점 간격으로 가른다. 21%(39/188) 미채점(저자 주장대로 일부가 80~85일 수 있음)이면 전순서는 ±1보다 훨씬 크게 불안정하다. 85 vs 84 vs 83을 의미 있는 순서로 제시하는 것은 false precision이고, 80/79 1점 선이 티어 색을 가른다.
- **수정안**: 점수를 5점 밴드 또는 티어만으로 제시, S티어를 *무순서 집합*으로. 명시적 ±3~5 불확실성 표기, 밴드 내 순위매김 중단("T-REX·AIRL은 동급 S, 선택은 regime로").

### M7. occupancy 비판 — 결론은 공정하나 메커니즘 서술이 부정확·과단순 〔중대 · 신뢰도 높음〕
- **인용**(§3 용어/§8.2): "occupancy(상태분포) 매칭 … 전문가가 느리면 느림까지 따라간다 … **BC처럼**". (SMODICE·DemoDICE·f-IRL·SEABO·ORIL·CLUE를 "1차 보상으론 부적".)
- **반박**(메커니즘): DICE류(SMODICE/LobsDICE/OptiDICE)는 BC가 아니다 — Fenchel 쌍대로 장기 할인 occupancy 매칭의 *가치함수*와 중요도가중(밀도비)을 배워 **스티칭**까지 한다(BC가 못 하는 것). "BC처럼"은 추정기 클래스를 오기술. 또 6개를 뭉뚱그리면 차이가 지워진다: f-IRL은 *명시적·전이 가능한 state-marginal 보상*을 복원, SMODICE는 *관측만(state-occupancy)* 매칭이라 느린 *행동*은 무시. **그래도 결론은 살아남는다**: 정책↔occupancy는 일대일이라 occupancy 매칭의 최적점은 *expert 정책 자신*이다 → expert가 느리면 천장이 expert 속도. 본 프로젝트 데모가 균일하게 느려(003 §2) 평균≈최선이므로 "초월 못함" 결론은 유효. (단 일반 명제 "occupancy는 반드시 expert 속도에 갇힌다"는 *분포 내 최선* 대비에서만 참 — mixed-quality 데모면 success-example occupancy로 평균은 능가 가능.)
- **수정안**: "BC처럼"을 "가치함수 기반 분포매칭이되 천장은 *주어진 expert 분포*"로 정정. 6개를 한 덩어리로 배제하지 말고 f-IRL(전이보상)·SMODICE(관측매칭)는 별도 취급. 결론(균일하게 느린 데모에선 1차 보상 부적)은 유지.

---

## 4. 경미 / 개선 (Minor)

- **m1. TROFI venue 약간 과대** 〔경미·높음〕: 001 "RLC(Reinforcement Learning Conference)" → 실제는 **RLC 2025의 *워크숍*(Reinforcement Learning and Video Games Workshop)**. main track 아님. → "RLC 2025 워크숍"으로 정정.
- **m2. EPIC 연도 오기** 〔경미·높음〕: 001 "2022, ICLR" → 실제 **ICLR 2021**(arXiv 2020). → "2021"로 정정.
- **m3. "IQL online 연장 = config 변경 수준" 과소평가** 〔경미·중간〕: §8.5가 스스로 반박 — 신규 online 판별자가 사전학습 정책 unlearn, OLLIE/CSIL init·spectral norm·off-policy replay 필요. IQL의 native 지원은 *가치 목적함수*에 한함이지, IRL이 요구하는 *판별자 공동학습*의 적대 불안정엔 해당 안 됨. → "value 목적함수는 같은 손실로 연장되나, 학습보상 결합 시 적대 불안정 유입(§8.5)"으로 한정.
- **m4. 무헤지 단정 톤(확증편향)** 〔경미·중간〕: "사용자 직관이 **정확히 맞다**"(line 14), "**정답** 패러다임", "단독은 **최악**", "**절대** 못 잡음". 결론을 증거보다 먼저 못박는다. 특히 사용자 직관을 top에서 "정확히 맞다"고 추인하면 하이브리드 결론으로 문서가 기운다 — 정작 이 repo의 증거는 offline-only로 충분했다고 말한다. → 중립 표현으로.
- **m5. SSRR "~0.95 corr with GT" 수치 미검증** 〔경미·중간〕: 논문(arXiv:2010.11723)·repo에서 정확한 0.95 출처 확인 불가. → "약 0.95(추정, 출처 미확인)"로 표기.
- **m6. 003(A) "GAIL" 명칭 부정확**(상세는 §6c): 정적 1-shot 판별자(E vs 고정 crash_data)는 GAIL의 적대 루프가 아니라 ORIL/PU 판별자다. → §6(c) 참조.

> **공격했으나 살아남은 것**(steelman 통과): ① 인용 진위(환각 0, arXiv ID·저자·코드repo까지 정확) — 오히려 강점. ② "exceed not imitate"로 occupancy 보상을 1차에서 배제한 *결론*(메커니즘 서술만 부정확). ③ 추천 골격(조화 → 랭크기반 보상 → IQL relabel)이 검증된 IQL 레시피와 동형. ④ action harmonization을 *랭킹 전에* 두라는 전제(009에서 load-bearing 입증). ⑤ non-identifiability 경고(보상 크기 직접비교 금지). ⑥ 9단계 *구조 자체*는 offline-first(online이 step8, 게이트 종속)라 TL;DR보다 정직. ⑦ 산술 무결(가중치 합 1.0; 티어 카운트 3+28+40+58+20=149, 188−149=39, A=10명명+18압축=28).

---

## 5. 논문 진위 스팟체크 표

| 논문 | 001 표기(연도/venue/arXiv) | 실제 확인 | 일치? | 비고 |
|---|---|---|---|---|
| TROFI | 2025, RLC, 2506.22008 | *Trajectory-Ranked Offline IRL*, arXiv:2506.22008, Sestini et al., **RLC 2025 워크숍** | ⚠️부분 | venue가 main이 아닌 워크숍. method(T-REX→relabel→TD3+BC) 정확 |
| OLLIE | 2024, ICML, 2405.17477 | 동일, *Offline Pretraining to Online Finetuning*, Hua et al., 코드 HansenHua/OLLIE-ICML24 실재 | ✅ | 완전일치 |
| CSIL | 2023, NeurIPS, 2305.16498 | 동일(NeurIPS 2023 **Spotlight**), Watson et al., google-deepmind/csil 실재 | ✅ | 완전일치 |
| D-REX | 2019, CoRL, 1907.03976 | 동일, Brown/Goo/Niekum, BC+노이즈 자동랭크 | ✅ | 완전일치 |
| CLARE | 2023, ICLR, 2302.04782 | 동일, Yue et al., 동역학모델 기반 보수적 offline IRL | ✅ | 일치(=world-model 의존 확인) |
| Gen.WM MLE IRL | NeurIPS Oral, 2302.07457 | *When Demos Meet Generative World Models*, **NeurIPS 2023 Oral**, Zeng et al., Cloud0723/Offline-MLIRL | ✅ | 일치(연도 2023 명시 권장) |
| Preference Transformer | ICLR, 2303.00957 | 동일(ICLR 2023), Kim et al., 비-Markovian 보상 | ✅ | 완전일치 |
| AIRL | 2018, ICLR, 1710.11248 | 동일, Fu/Luo/Levine, **on-policy(TRPO)** | ✅메타 / ⚠️기술 | 메타 정확하나 공식 g(s,a)·"off→on"은 오류(M3·C3) |
| EPIC | **2022**, ICLR, 2006.13900 | *Quantifying Differences in Reward Functions*, **ICLR 2021**, Gleave et al. | ⚠️연도 | 연도만 오기(2021). 나머지 정확 |
| T-REX | 2019, ICML, 1904.06387 | 동일, Brown et al., 랭크 pairwise→외삽 | ✅ | 완전일치 |
| IQL | 2021, 2110.06169 | 동일, Kostrikov/Nair/Levine, online finetune native | ✅ | 완전일치 |
| SSRR | (서술) noise-curve 회귀, ~0.95 corr | *Self-Supervised Reward Regression*, CoRL 2021, arXiv:2010.11723, Chen et al. | ✅메타 | 0.95 수치만 미확인(m5) |
| ORIL | (서술) PU 판별자→relabel→offline RL | *Offline Learning from Demos & Unlabeled Exp.*, arXiv:2011.13885, Zolna et al.(DeepMind) | ✅ | 정확 |
| GT Sophy | Outracing GT champions, 2022, Nature | 동일, Wurman et al., Nature 602:223-228 (2022) | ✅ | 완전일치 |
| BeTAIL | 2024, human racing AIL | arXiv:2402.14194, Weaver et al., IEEE RA-L 2024 | ✅ | 정확 |
| F1tenth Offline RL | 2024 | Koirala & Fleming, arXiv:2408.04198 (2024) | ✅ | 식별됨 |
| High-speed racing TAL | 2023 | Evans et al., arXiv:2306.07003, RA-L 2023 | ✅ | 식별됨 |
| Auto Reward Design GT | 2025 | arXiv:2511.02094, Ma et al.(Sony AI) | ✅ | 정확 |
| IQ-Learn | 2021, NeurIPS | arXiv:2106.12142, NeurIPS 2021 Spotlight, offline 변형 실재 | ✅ | 실재·주요(단 미채점=M4) |
| OTR | 2023 | *Optimal Transport for Offline IL*, ICLR 2023 Oral, arXiv:2303.13971 | ✅ | 정확(단 미채점=M4) |

**결론**: **환각·오귀속 0건.** 스팟체크 19편 중 17편 일치, 2편(EPIC 연도, TROFI 워크숍) 경미 오류뿐. 문헌 진위는 001의 *강점*이다 — "치명"으로 보고할 사안 없음.

---

## 6. 핵심 논제 재평가

### (a) "offline-then-online이 정답"이 맞나 → **부분적. offline-FIRST가 옳은 기본값, online은 조건부.** 〔신뢰도 높음〕
- 순수 offline steelman: 이 repo가 (i) offline IQL로 34.3s 2랩 완주, (ii) 유일한 폐루프 실패를 *offline rc 커버리지*로 해결, (iii) 폐루프 평가는 frozen-policy 무료 점검 — 이미 모두 보였다. 그래서 검소한 기본값은 "랭크보상 offline 학습 → relabel → IQL → 폐루프 *평가* 게이트 → 통과 시 정지". online은 패러다임이 아니라 *조건부 분기*.
- 순수 online AIRL steelman: Topic-2의 *명시 목표*가 "새 트랙 전이 보상"이고 001 스스로 AIRL을 "전이성 최강"이라 한다. 전이 보상이 진짜 deliverable이면 online AIRL이 그 성질로 가는 가장 곧은 길이고, 추천한 랭크보상(T-REX/D-REX)은 *전이 보장이 없다*(원천 트랙 궤적에 return 적합). → "단독은 최악"은 *비용* 논거이지 *품질* 논거가 아니다(off-policy-AIRL/OPIRL이 비용을 일부 완화). 001은 비용을 품질로 과장했다.
- 진짜 알맹이(001이 옳은 한 곳, line 58 "offline reward가 틀렸으면 online이 나쁜 신호를 더 쫓음"): **학습된(틀릴 수 있는) 보상은 폐루프 위험을 올린다** — 입증 IQL은 *손수 설계한* 보상(progress+−50+lap)을 썼고, 학습보상은 빠른 정책이 가는 OOD/off-line 상태(005/008의 98~99%ile tail)에서 틀릴 수 있는데 IQL expectile는 OOD *행동*만 막지 *오설정 보상*은 못 막는다. → 이게 문서의 *진짜* thesis가 돼야 한다(현재는 Diffuser 혼동에 묻힘). 그리고 그 처방은 online이 아니라 **offline 보상검증 + 폐루프 평가 게이트**.

### (b) occupancy 비판이 공정한가 → **결론 공정, 메커니즘 서술 부정확.** 〔신뢰도 높음〕
- SMODICE/DICE는 가치함수+중요도가중(스티칭 가능)이라 "BC처럼"은 틀린 기술. f-IRL은 전이 가능한 명시적 state 보상 복원이라 6개 뭉뚱그림은 과단순.
- 그러나 "occupancy 매칭의 천장 = 주어진 expert"라는 *결론*은 정책↔occupancy 일대일로 정당하고, 본 데모가 균일하게 느려 결론이 살아남는다. 랭크기반이 외삽으로 "초월"을 인코딩한다는 *선호*도 옳다(T-REX >2× best demo, D-REX 8/9 과제서 best demo 능가).

### (c) 003 (A) 비판이 정확한가 → **절반 부정확하나 바닥 결론은 옳음.** 〔신뢰도 높음〕
- 001의 "GAIL식 occupancy 매칭" 명칭은 **메커니즘상 틀림**: GAIL은 *현재 정책 롤아웃* 대비 판별자를 적대적으로 반복 갱신해 ρπ→ρE를 맞춘다. 003(A)는 정책 롤아웃·min-max·내부루프가 전혀 없이 E vs *고정* crash_data에 판별자를 한 번 학습 — 003 스스로 "1-shot 근사(정통 GAIL/AIRL은 온라인 반복)"라 부른다. 이 구성은 **ORIL/PU 판별자 offline IRL**이지 GAIL이 아니다. ("occupancy *매칭*"이 아니라 "occupancy *편향*"이 정확 — 001 §8.2는 "occupancy 과적합"으로 *옳게* 부른다.)
- "느린 cap5/cap10 분포로 끌어당긴다"는 비판은 **003의 권장안엔 안 맞는다**: 003 §2·§3·§5는 expert E로 *빠른 Dreamer(16.6s)* 새 수집을 권한다. E가 빠르면 판별자 보상은 *빠른* 행동으로 끌어당긴다. 001은 "crash_data를 positive로"라는 문자 그대로의 읽기만 공격하고 003이 *실제 권장한* 빠른-expert 안은 다루지 않는다.
- 그러나 **바닥 결론(랭크기반 > 판별자기반, exceed 목표엔)은 정확**하다 — 이유는 (b)·(c)와 무관하게 옳다: 판별자/occupancy 보상은 expert 분포에서 *최대*라 expert를 *초월*할 gradient가 없고, 랭크기반은 단조 보상을 적합해 best demo 너머로 외삽한다. → 001의 권고는 *잘못된 라벨·잘못된 슬로우-프레이밍에도 불구하고* 옳은 이유로 옳다.

---

## 7. 실행가능성 — 9단계별 구멍/숨은 전제

| 단계 | 판정 | 구멍/숨은 전제 |
|---|---|---|
| 1. action harmonization 먼저 | ✅ 견고 | 검증된 전제(009). cap별 v_collect 폴더명 파싱 유지 필요 |
| 2. 랭킹 "공짜" 구성 | ⚠️ **M2** | 순간속도≠궤적return. 충돌 cap20이 완주 cap10 위로 가면 보상 오염. "공짜"는 최난 단계를 숨김 |
| 3. offline reward 학습(T-REX/D-REX/TROFI) | ⚠️ **M1** | D-REX는 실제 sim-롤아웃 hybrid. "offline" 표기·"행동교란 근사" 미검증 |
| 4. EPIC로 보상 offline 검증 | ⚠️ **C4** | reference reward 없음(proxy와 비교=손수설계 재유입). §8.4와 모순 |
| 5. 보수성(CLARE/bi-level MLE) | ⚠️ **C3** | CLARE=133D world-model=Diffuser 실패 재현. IQL expectile로 충분할 수 있음 |
| 6. IQL relabel(드롭인) | ✅ 견고 | 검증된 파이프라인. 보상 훅만 교체 |
| 7. f110 폐루프 평가 게이트 | ✅ **핵심** | 이게 진짜 필요한 단계(frozen-policy, 무료). online과 혼동 금지 |
| 8. 짧은 online finetune | ⚠️ **C2** | 기본이 아니라 최후여야. 앞에 offline 진단분기(coverage→rc / reward→refit) 필요. m3(IQL online 비용 과소) |
| 9. 새 트랙 전이검증 | ✅ 방향 OK | 실측 강조 좋음. 단 EPIC를 전이 게이트로 과신 금지(C4) |

---

## 8. 누락 · 과대 · 과소 평가

- **누락**: ① **rc(offline 상태커버리지)** — 이 프로젝트의 검증된 폐루프 해법인데 9단계에 없음(C2). ② **IQ-Learn(offline)·OTR** — 추천 파이프라인 정중앙인데 미채점(M4). ③ AIRL 전이보장이 **행동의존 보상·off-policy에서 깨짐**(M3) 언급 없음. ④ 충돌 데이터의 **return 랭킹 구성·검증** 방법(M2).
- **과대**: "정답 패러다임", "전이 최강 증명적 복원", "절대 못 잡음", S티어 1점 정밀도(M6), EPIC "전이 예측 게이트"(C4), "랭킹 공짜"(M2), 사용자 직관 "정확히 맞다".
- **과소**: D-REX/SSRR의 online성(M1), "공짜 랭킹"의 실제 난이도, IQL online 연장 비용(m3), CLARE 133D world-model 비용(C3).

---

## 9. 그래서 계획을 바꿔야 하나 — 구체 권고

**뼈대는 유지한다**(조화 → 랭크기반 보상 → IQL relabel → 폐루프 *평가* 게이트). 아래만 고치면 채택 가능:

1. **TL;DR 재작성(C1)**: "Diffuser는 model-based world-model 실패이며 model-free IQL엔 전이 안 됨. 순수 offline은 폐루프를 잡을 수 있다(우리 IQL 증명). **offline-FIRST가 기본값, online은 조건부 최후 분기.** 진짜 새 위험은 *학습된 보상의 오설정*이다."
2. **9단계에 진단 분기 삽입(C2)**: step7 실패 → (i) coverage gap이면 **rc 추가(검증됨)**, (ii) reward error면 보상 offline 재적합, (iii) 둘 다 소진 시에만 step8 online.
3. **상위 랭킹 재산정(C3)**: AIRL/CLARE에 §3의 regime·integration 페널티를 *실제 점수에* 부과; Diffuser 증거를 CLARE continuous_highdim에 차감; S티어를 *무순서 집합*으로(85/84/83 순위 폐기, M6); D-REX를 deploy 1순위로 명시.
4. **EPIC 격하(C4)**: "전이 예측 게이트" → "보상 spurious-feature sanity 체크". reference가 proxy(ground-truth 아님)임 명시. §3/§4/§5를 §8.4와 일치. 연도 2021로 정정(m2).
5. **IQ-Learn·OTR 2차 채점 먼저(M4)**: §5를 결정근거로 쓰기 전에. 특히 IQ-Learn offline 변형·OTR(둘 다 추천 파이프라인 정중앙)을 점수까지 채워 비교.
6. **랭킹 구성 정직화(M2)**: "공짜" 삭제. 속도순≠return순 분리, 충돌 cap20이 완주 cap10 위로 안 가게. held-out 순서쌍 검증을 *핵심 리스크*로.
7. **D-REX 라벨 정정(M1)**: "offline" 표기 철회(데이터 생성=짧은 sim 롤아웃). "행동교란 offline 근사"는 삭제 또는 "검증 필요".
8. **AIRL 공식 정정(M3)**: 전이형 **g(s)**로; 전이보장 조건(state-only·분해가능·결정론) 명시; "증명적/최강"→"가정 하 우수, 미검증".
9. **원시 산출물 복구(M5)**: irl_full.md·JSON을 001 옆에 복구/commit하거나 §5를 "잠정 랭킹"으로 격하.
10. **메타데이터 정정**: EPIC 2021(m2), TROFI "RLC 2025 워크숍"(m1), SSRR 0.95 "추정"(m5).
11. **003(A) 비판 정정(§6c)**: "GAIL"→"ORIL/PU 정적 판별자(occupancy 편향)". "느린 데모 모방"은 003이 빠른 Dreamer를 expert로 쓰면 안 맞음을 인정. 단 바닥 결론(랭크기반 선호)은 유지.
12. **무헤지 표현 제거(m4)**: "정답"·"최강"·"절대"·"정확히 맞다"·"공짜"·"config 변경 수준".

> **요지**: 계획을 *버릴* 필요는 없다. 문헌은 실재하고 골격은 검증된 레시피와 같다. 고칠 것은 (i) 거짓 패러다임 프레이밍을 진짜 thesis(학습보상 오설정 위험)로 교체, (ii) 검증된 offline 레버(rc)를 폐루프 해법으로 복원, (iii) 자기모순 랭킹(T-REX/AIRL/CLARE/EPIC)을 정렬, (iv) 검증 불가·false-precision 수치를 정직화 — 네 가지다.

---

## 10. 검수 자체의 한계 (내가 확인 못 한 것)

- **원시 산출물(irl_full.md·wd5l13krs.output) 부재**로 8축 subscore·149편 *개별* 점수는 검증하지 못했다 → 점수의 타당성은 "틀렸다"가 아니라 **"검증 불가"**로만 판정. M6(false precision)·M3 가중적용 주장은 산출물 복구 후 재확인 필요.
- **미채점 39편 전체를 재채점하지 않았다** — IQ-Learn·OTR·레이싱군의 *실재/중요도*만 확인했지 8축 점수는 안 매겼다.
- **C/D 티어 다수 논문은 진위 스팟체크 비대상**(의뢰문 지정 + 추천 스택 직결 논문 위주 19편만 교차확인). 저티어에 환각이 숨어 있을 가능성은 배제 못 함.
- **코드 실행 금지**라 D-REX/CLARE/AIRL가 f1tenth(133D)에서 실제로 어떻게 동작하는지는 *이론·문헌 기반 예측*이다 — 실측 아님. M1의 "행동교란이 랭킹을 깬다", C3의 "CLARE가 Diffuser처럼 실패한다"는 강한 *예측*이되 미실행.
- **SSRR 0.95 등 일부 수치는 medium confidence**(출처 미확정).
- 검증 워크플로우의 웹서치 결과(특히 STARC 반례·Rethinking-AIRL의 off-policy 깨짐)는 에이전트 보고에 의존했다 — 1차 논문 전문 정독은 아님(신뢰도는 표기된 대로).

---

> ※ 본 문서는 검수(조사·비판)까지다. 코드 수정·실행·수집·학습·commit/push/pull은 **사용자 명시 지시가 있을 때만**.
