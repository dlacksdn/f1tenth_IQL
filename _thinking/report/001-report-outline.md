# 001 — 최종 보고서 목차 (구조 확정안)

> 2026-06-22. f1tenth Offline RL(IQL 스티칭) 개인과제 **최종 보고서**의 목차/구조 SSOT.
> 목적: 보고서 본문을 쓰기 전에 "어떤 순서로 무엇을 말할지"를 못박는다.
> 근거: IQL 성공 과정은 `f1tenth_IQL/_thinking/`(특히 implementation/009·010, goal/001),
> 시도-행적(LeWM·Diffuser 실패)은 `f1tenth_planning_with_diffusion/_thinking/`(53개 문서) 전량을 읽고 정리.
> 작성 원칙: 엄밀하게, 사용자도 이해할 수 있게. append-only(기존 문서 수정 금지).

---

## 0. 이 목차의 한 줄 요지
"느린 완주 데이터 + 빠른 **충돌** 데이터를, 완주 expert 시연 없이 IQL로 스티칭해
cap10 기록(56.14s)을 21.82s 경신(34.32s)했다"는 본론을, **왜 IQL이어야 했는가(LeWM·Diffuser 실패)**
→ **어떻게 했는가(핵심 레시피)** → **얼마나 됐는가(결과·ablation)** → **무엇이 한계인가(정직한 양면성)**
의 인과 흐름으로 전달한다.

---

## 본문 목차

### 1. 서론 — 문제정의·목표
- 1.1 과제·환경: Topic-1 Offline RL — ~100s급 정책으로 데이터 수집 후, **추가 환경 상호작용 없이** 더 빠른 정책 학습. 평가 = Oschersleben 2랩 lap time, eval env V_MAX=20.
- 1.2 난관: 폐루프 첫 코너 정밀도, BC가 모든 설정에서 완주 0%였던 벽.
- 1.3 목표 사다리: **G1** 폐루프 2랩 완주 / **G2** baseline cap5 107.16s 격파 / **G3** cap10 기록 56.14s 경신(진짜 목표).

### 2. 데이터셋
- 2.1 모은 전체 데이터: cap5~cap20, 완주/충돌 구성과 생성 정책(준-expert Dreamer).
- 2.2 실제 사용 데이터: cap10_full + cap15 + cap20 + cap15_rc + cap20_rc ≈ **1.02M transition / 1,684 ep**.
- 2.3 정렬 규약: DreamerV3 규약(action[t]/reward[t]는 obs[t]를 만든 행동·보상), 로더 s/a/r/s'/done 구성.

### 3. 시도한 행적들 (왜 IQL에 도달했나)
> 이 장은 `f1tenth_planning_with_diffusion/_thinking` 전량에 근거. "삽질"이 아니라
> IQL이 작동할 조건을 규명한 과정으로 프레이밍한다.

- 3.1 **LeWM (부적합 — 조기 이탈)**
  - reward-free goal-reaching 구조 → "빠르게"가 목적함수가 아님(연속 레이싱과 구조적 불일치).
  - 모달리티 불일치(픽셀/ViT vs 라이다 벡터), 2-프로세스 평가 한계(C-1: `_set_state`/goal-state 미지원, py3.10/py3.8 분리), 짧은 계획지평(lookahead 0.5s vs 랩 6~19s)·1프레임 속도 미관측.
- 3.2 **Planning with Diffuser (부적합 — 가장 길고 많이 배운 실패)**
  - 도입 동기와 계획 4회 반복(crash-only 수집 결정, action tier별 de-normalization 정립 = 후일 IQL **조화**의 원형).
  - P6 평가 = 완전 실패: 5/5 충돌, 완주 0, K{1..30}×scale{0..1}×value 32개 전부 실패(모델·normalizer·obs는 정상).
  - 진단된 구조적 부적합:
    - **D3(보상-가치 정렬 문제)**: dense progress는 value를 "충돌-fast"로 오도하고, 완주를 결정하는 lap bonus(+100)는 γ=0.99에서 ~1400 step 앞 → 0.0001로 소멸 → value가 완주를 못 봄. *문제는 보상 크기가 아니라 거리* → γ=0.999만이 순서를 역전. (※ "논문의 sparse 강점을 간과"가 아니라 **정량 진단으로 규명**된 문제로 서술.)
    - crash-poisoned prior(데이터 93% 충돌) / value가 나쁜 prior를 못 이김(`action[0,0]`만 실행).
    - closed-loop compounding error·covariate shift(튜닝으로 제거 불가능한 구조적 트레이드오프).
  - **결정적 대비**: 같은 cap8 완주 데이터 — dreamer RL 정책 9.5% 완주 vs Diffuser BC 0% → **병목은 모델**.
  - Diffuser 구조적 부적합 3축: 느린 추론(20-step denoising vs 50Hz)·open-loop 계획·생성 노이즈 vs margin-0 코너 정밀.
- 3.3 **Offline RL 모델 선정**
  - value-maximizing 채택(IRL/imitation은 천장=expert), TD3+BC vs IQL 검토.
  - **IQL 확정**: expectile τ 무차원·affine-invariant(비정규화 reward +100 spike에 강함), in-sample이라 OOD(v_max=20) 미조회, crash-dominated 데이터에 견고.

### 4. 왜 IQL인가
- 4.1 IQL의 특성: in-sample expectile value + advantage weighted regression(AWR) + stitching.
- 4.2 본 문제와의 부합: 3.2/3.3의 부적합 사유(보상-가치 정렬, 폐루프 정밀, OOD)를 어떻게 정확히 해소하는가.

### 5. 구현 — 핵심 레시피
- 5.1 IQL 파이프라인(CORL vendor, 상류 무수정).
- 5.2 **★ action 조화(harmonization)**: cap별 수집 v_max(10/15/20)를 공통 frame(v_common=20)으로 재정규화 `a1'=(a1+1)(v_collect+5)/(v_common+5)−1` → 고속 직선과 코너 감속을 한 frame에서 stitch 가능하게 함.
- 5.3 **★ random-centerline 수집**: 센터라인 랜덤 스폰으로 트랙 전역의 다양 상태를 덮어 covariate gap 해소(이게 없으면 lap2 완주 0).
- 5.4 하이퍼: γ=0.999, collision_penalty=−50, β=3.0, iql_τ=0.7, common_v_max=20.

### 6. 결과
- 6.1 최종 성능: 2랩 34.32s(lap [17.78, 16.54]), cap10 56.14s 대비 **+21.82s**, baseline 107.16s 대비 +72.84s.
- 6.2 **구성요소 ablation**(실험 여정 = 각 요소가 왜 필요했나): D2(cap10, 느림) → D3(+cap15/20, 고속 stitch 성공·lap2 충돌) → T1(보상 튜닝 실패) → D4(+random-centerline, 2랩 완주).
- 6.3 견고성/한계: 완주 ckpt 4/30, seed 3/4(0·2·3 완주, 1 충돌), 취약점=센서 노이즈 견고성(seed=라이다 스캔 노이즈).

### 7. 논의 — 데이터의 정직한 양면성
- 7.1 준-expert 근거: 충돌 라벨의 **99.84%가 정상 고속 주행** 전이, 충돌 ep의 27%가 1랩 완주 후 충돌, lap완료 보상 전이 511건 내포, 평균 12.9 m/s.
- 7.2 구조적 편향: 모든 전이가 충돌로 종단, **고속 2랩 완주 시연은 데이터에 0개**, expert 완주분(42%)을 의도적으로 폐기.
- 7.3 왜 "데이터 매칭이 아니라 능가"인가: 단일 궤적·폐기한 expert 완주보다 나은 정책 = 정통 offline RL stitching(IRL/BC와의 결정적 차이).

### 8. 향후 계획
- 8.1 완주 토대(cap10) 없이, **전부 1랩 미만 불완전 데이터만으로도** stitch가 완주를 합성할 수 있는가(cap10 제거 재학습으로 검증 가능한 가설).
- 8.2 여러 맵 주행 데이터로 학습 시 일반화 성능이 증가하는가(현재 단일맵 Oschersleben).

### 부록
- A. 재현: 2-venv 구성, eval/watch/sweep 명령, 코드맵(f1tenth_data.py·train_iql.py·eval_iql.py 등).
- B. best 정책: `runs/d4_iql_stitch_rc/checkpoint_600000.pt`.

---

## 주요 구조 결정 (원안 대비 변경점 — critic가 검증할 포인트)
1. **목표를 데이터셋에서 분리해 서론(1장)으로 독립** — 보고서의 척추.
2. **"시도한 행적들"(3장)을 LeWM/Diffuser/모델선정 3단으로** 확장, 모두 diffusion `_thinking` 문서 근거.
   특히 원안의 "sparse reward 강점을 간과" 표현 → **D3 정량 진단**으로 정정(문서에 "간과" 자책 없음).
3. **harmonization·random-centerline을 5장 "핵심 레시피"로 별도 부각**(단순 구현이 아니라 성패를 가른 기여).
4. **실험 여정을 6.2 ablation으로** 재배치(각 데이터/하이퍼 요소의 필요성을 증명).
5. **010의 정직한 양면성을 7장 독립 논의로**(준-expert vs 구조적 편향) — 보고서의 학술적 무게 중심.

## 검증이 필요한 사실(보고서 작성 전 재확인 권장)
- 사용 데이터 규모: 009/010은 ≈1.02M transition / 1,684~1,687 ep, D4 학습 시점 정확히 1,023,639 transition.
- 모든 lap time은 2랩 기준, eval은 결정론(한 seed면 100% 재현).
- 3장의 Diffuser 수치(γ 할인, cap8 9.5% vs 0% 대비 등)는 diffusion `_thinking`의 plan_new/019·020·021, implementation/013·014에 근거.
