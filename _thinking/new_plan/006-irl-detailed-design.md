# 006 — IRL 상세 설계 (005 확정계획의 구현 스펙)

> 2026-07-03. [[005-irl-plan-final]] §8이 예고한 상세 설계. 코드 실측([f1tenth_data.py](../../f1tenth_data.py), [train_iql.py](../../train_iql.py)) 위에 작성.
> 대화에서 확정된 추가 결정 반영: **경로 A의 기본 = TROFI식 기존-return 랭킹(완전 offline, 롤아웃 0)**,
> D-REX 노이즈 롤아웃은 **비상 보강으로 강등**. 파일럿은 **단계적 병행**(A 먼저, OTR 설계는 GPU 시간에 병렬).
> 원칙: 엄밀하되 작성자가 다시 읽어도 이해되게. append-only. 구현은 사용자 지시 후.

---

## 0. 전체 그림 (한눈)

```
[1] 에피소드 랭킹 구축(기존 데이터, 조화 return 아님 — 결과지표 기반)  ← 신규 스크립트
        ↓ (held-out 순서쌍 검증 게이트 G1)
[2A] T-REX 보상망 학습 (Bradley-Terry pairwise)     [2B] OTR: preprocessor + OT 정렬 보상
        ↓                                                ↓
[3] relabel — 에피소드별 사이드카 .npy 생성(원본 npz 불변)               ← 공통 인터페이스
        ↓ (보상 sanity 게이트 G2)
[4] 기존 IQL 그대로 (--reward_dir 하나만 추가)  → runs/irl_*
        ↓
[5] 폐루프 평가 게이트 G3 (eval_sweep, 기존 코드 불변)
        ↓ 실패 시
[6] 개입-소거 진단(rc 레버 먼저) → 재적합 → (최후) online
        ↓ 통과 시
[7] map_easy3 전이 — 대조군(D4 손설계 정책)과 정면 비교
```

핵심 설계 사상: **기존 코드는 거의 안 건드린다.** 신규 코드는 `irl/` 서브패키지에 격리하고, 기존 파이프라인 변경은 로더의 `reward_dir` 옵션 1개 + `train_iql.py` 인자 passthrough 1개뿐이다(기본값 None = 기존 동작과 완전 동일 → 하위호환 + 스모크로 검증).

---

## 1. 데이터 인벤토리와 학습 데이터 구성

디스크 실측(2026-07-03, `crash_data/`):

| 폴더 | ep | 성격 |
|---|---|---|
| cap5_full 31 / cap10_full 40 | 71 | **완주 포함**(느린 expert 후보) |
| cap15 371 / cap20 291 | 662 | 고속 충돌(단일 출발) |
| cap15_rc 517 / cap20_rc 468 | 985 | 고속 충돌(랜덤 센터라인 스폰 = 전트랙 커버리지) |
| cap8_jitter 300, cap10 21, cap5 13 등 | — | 보조 |
| *_sub1lap 4종 | 1,214 | <1랩 필터(심링크, 001 실험용) |

- **IQL 학습 데이터 = D4 구성 그대로**(cap10_full + cap15 + cap20 + cap15_rc + cap20_rc; 보고서의 통제 기준, ~1.02M transitions). **바뀌는 것은 보상 라벨뿐** — 그래야 기존 D4 결과(2랩 ~34.3s)가 그대로 손설계-보상 대조군이 된다(추가 계산 0).
- **OTR expert set** = cap10_full 중 완주 에피소드(2랩 56s, ~30 ep). ⚠️ 느린 expert라 근접 보상의 천장 위험(005 §5) — 파일럿 결과에 따라 빠른 Dreamer(16.6s) 데모 수집을 후속 옵션으로.

## 2. [1] 에피소드 랭킹 — Topic-2 정합이 핵심

**원칙: 랭킹 기준에 손설계 dense 보상을 쓰지 않는다.** progress+collision 합산 return으로 랭크하면 학습 보상이 손설계 보상의 증류가 되어 Topic-2("사람 설계 reward 없이")가 무너진다. 대신 **관측 가능한 결과지표(outcome)만** 쓴다 — 이것이 T-REX의 전제(궤적 수준의 서수적 약한 감독만 허용)와 정확히 일치한다.

- **에피소드 지표**(npz에서 직접 계산, 001 검증 방식 재사용):
  - `laps` = `log_reward_lap > 0` 횟수 (랩 통과 이벤트 수 — 보상값이 아니라 이벤트 카운트로만 사용)
  - `crashed` = `is_terminal.any()`
  - `dist` = Σ |state[:,0]×20| × 0.02s (실주행거리 — 스폰 위치에 견고, 001 실증)
  - `avg_v` = dist / (T×0.02)
- **전순서(사전등록)**: ① `laps` 내림차순 → ② 비충돌 > 충돌 → ③ 같은 계층 안에서 `dist` 내림차순(충돌군), 랩타임 오름차순(완주군). `avg_v`는 동점 타이브레이크.
  - 이유: "충돌한 고속 cap20이 완주한 cap10 위로 못 가게"(002-M2)를 ①②가 구조적으로 보장. 속도 선호는 완주군 내부 랩타임과, 충돌군 내부의 (같은 생존거리에서 더 빠름=상위) 비교에서 자연 유입.
- **페어 샘플링**: 랭크 차이가 명확한 쌍만(계층이 다르거나 dist 차 ≥ 트랙 10% = ~27.5m). 근소한 쌍은 라벨 노이즈라 제외.
- **G1 게이트(held-out 순서쌍 검증, 사전등록)**: 에피소드를 폴더별 층화 80/20 분할. 학습된 보상망의 세그먼트-합 예측이 held-out 쌍의 순서를 **≥80%** 맞추면 통과. 실패 시 → 세그먼트 길이/페어 기준 조정 1회 → 재실패 시 **비상 보강 = D-REX 노이즈 롤아웃**(rc 스폰으로 충돌 위치 분산, `collect_crash_data.py --start-jitter` 재사용) 투입.
- **정직한 한계(문서화)**: "멀리+무사고+빠름=좋음"이라는 서수 기준 자체는 사람이 정한 것이다. 다만 이는 궤적당 1비트 수준의 약한 감독이고, per-step dense 보상은 전부 학습된다 — T-REX 계열의 표준적 정당화. 보고서에 이대로 기술.

## 3. [2A] T-REX 보상망 (경로 A 기본)

- **입력**: obs 133D (state-only; action 미입력 — 노이즈 행동라벨 오염 회피 + T-REX 관례). 로더와 **동일한** min-pool·정규화 경로 사용(train/reward 정합).
- **아키텍처**: MLP 256×2 → scalar (프로젝트 관용 그대로). Adam 3e-4, batch 64쌍.
- **손실**: Bradley-Terry — 세그먼트 σ_A ≻ σ_B 쌍에 대해 `CE( softmax(Σr̂(σ_A), Σr̂(σ_B)), A )`. 세그먼트 길이 **L=50 스텝(1.0s)** 기본(부진 시 {25,100} ablation). 세그먼트는 각 에피소드에서 균일 샘플, 쌍의 라벨은 **에피소드 랭크**에서 상속.
- **정칙화**: weight decay 1e-4 + 출력 스케일 억제(l2 on Σr̂). 학습 ~30k iter, held-out 쌍 정확도로 early stop.
- **후처리(스케일)**: 전체 데이터셋에서 r̂를 z-score → 손설계 progress 보상의 per-step std에 맞춰 재스케일(γ=0.999·β=3.0 스케일 궁합 유지). terminals(충돌=True 흡수)는 **불변** — 충돌 페널티를 손으로 더하지 않는다(전부 학습 보상에 맡김; 근처 상태의 낮은 r̂가 대신해야 하고, 그게 안 되면 G2에서 걸린다).

## 4. [2B] OTR (경로 B — GPU 시간에 병렬로 설계만 선행)

- **preprocessor(선결난제, 004 경미-2 반영)**: 기본 cosine cost를 raw 133D에 걸면 성분 수 비대칭(128 vs 5)으로 lidar가 지배. **사전등록 변형 2개**:
  - V1(기본): 블록 균형 z-score — `[ zscore(lidar128)/√128 , zscore(state5)/√5 ]` → 두 블록이 내적에 동등 기여.
  - V2(대비): state5-only(+선택적으로 lidar 최소거리 1D 요약) — 기하 무시, 동역학 상태만.
  - V3(학습 인코더)는 파일럿에서 제외(과설계).
- **OT 정렬**: OTR 공식 코드(`ethanluoyc/optimal_transport_reward`)의 rewarder를 이식하되 cost feature만 preprocessor 출력으로 교체. 에피소드마다 expert set과 정렬해 per-step 보상(정렬비용의 음수 스케일) 산출, expert 여러 개면 max. Sinkhorn 파라미터는 코드 기본값에서 시작.
- **후처리**: 경로 A와 동일한 z-score+재스케일(비교 공정성).

## 5. [3] relabel — 사이드카 인터페이스 (공통, 원본 불변)

- **저장**: `f1tenth_RL_project/runs/irl_rewards/<tag>/<dataset>/<episode>.npy` — 원본 npz와 같은 길이 T의 per-step r̂ (규약 유지: r̂[0]=0 더미, r̂[t]=obs[t] 도착 보상 → 로더의 기존 `[1:]` 슬라이싱이 그대로 적용). `<tag>` 예: `trex_v1`, `otr_v1`. + `meta.json`(방법·보상망 ckpt 해시·스케일 통계·생성일).
  - 구현 정의: `r̂[t] = r̂_net(obs[t])` (도착 상태 기준) — 기존 규약 "reward[t]는 obs[t] 도착 보상"과 정확히 동형.
- **로더 변경(유일한 기존 코드 수정 1)**: `load_f1tenth_dataset(..., reward_dir=None)` — 지정 시 `_episode_reward()` 대신 사이드카 로드(파일 대응은 basename 매칭, 길이 불일치=즉시 에러). `collision_penalty`/`lap_bonus` 인자는 이 경로에서 무시됨을 docstring·로그에 명시. **기본값 None이면 바이트 단위로 기존 동작과 동일** → 스모크(기존 D2/D3 로드 결과 해시 비교)로 하위호환 검증.
- **train_iql.py 변경(유일한 기존 코드 수정 2)**: `--reward_dir` 인자 추가 → 로더로 passthrough. 그 외 불변.
- **G2 게이트(보상 sanity, 사전등록)**:
  - P1(경성): 완주 에피소드의 세그먼트 평균 r̂ **>** 충돌 직전 100스텝 윈도의 평균 r̂ (전 폴더에서 성립).
  - P2(연성·기록만): 안전 구간에서 r̂ ↔ 전진속도 양의 상관.
  - P3(spurious 점검): EPIC-lite — r̂ vs progress-proxy 거리 + pose(x,y) 성분 permutation importance가 지배적이지 않을 것. (005 §2-4: sanity 전용, 판별자 아님.)

## 6. [4]~[5] IQL 학습·평가 (기존 파이프라인, 통제 유지)

- **하이퍼 = D4 통제 그대로**: γ=0.999, β=3.0, iql_τ=0.7, common_v_max=20, lidar128, 256×2, lr 3e-4, batch 256, **600k step, ckpt_freq 20000, seed 0**, normalize=1.
- **실행 규약**: GPU 학습은 반드시 `run_in_background`(foreground+CUDA=exit144). 정지 전 디스크 ckpt(state_N>0) 확인. out_dir 신규(`runs/irl_e1_trex_v1` 등, 덮어쓰기 금지).
- **G3 게이트**: `eval_sweep.py --run_dir ...` 30-ckpt 스윕(seed 0) → 완주 ckpt에 seed 0~3 견고성. 판정 기준(사전등록): ① 2랩 완주 ckpt 존재 ② 최견고 ckpt seed ≥2/4 ③ (참고) 2랩 시간 vs D4 34.3s.

## 7. [6] 게이트 실패 시 — 개입-소거 진단 (005 step8 구체화)

관측 사전판별은 불가(두 실패모드가 같은 증상, 004 치명-2). 순서:
1. **rc 레버 먼저**(가장 싼 검증된 개입): 데이터 믹스에 `cap15_rc_sub1lap`·`cap20_rc_sub1lap`·`cap8_jitter` 추가(디스크에 이미 있음, 수집 0) → 같은 보상으로 재relabel→재학습→재평가. **해소되면 coverage gap이었던 것으로 사후 확정.**
2. **잔존 시 보상 재적합**: G1/G2 산출물 재검토 → 랭킹 기준(§2 ②③)·세그먼트 길이·스케일 조정 → 보상망 재학습. 충돌 직전 r̂ 분포를 사후 진단 재료로.
3. **둘 다 소진 시에만 online**(005 step9 — OLLIE/CSIL init, PPO로 보상 복원·SAC은 정책만, spectral norm).

## 8. [7] map_easy3 전이 — 대조군 정면 비교 (005 step10 구체화)

- **L1(파일럿 범위): 정책 zero-shot 전이.** 실험군 = 학습보상 IQL 최견고 ckpt / **대조군 = 기존 D4 손설계 ckpt(이미 디스크에 있음, 신규 학습 0)** — 둘 다 map_easy3에서 `eval_iql.py --map map_easy3`(맵 인자 지원 여부 확인, 없으면 소폭 추가) seed 0~3 완주율·시간.
- **사전등록 승리조건**: 학습보상 정책의 map_easy3 완주율 ≥ 손설계 정책. (Oschersleben 성능은 D4 34.3s 대비 열세여도 무방 — 주장은 "전이"에 건다.)
- **L2(후속 옵션): 보상 전이.** frozen r̂로 map_easy3 신규 수집 데이터를 relabel→IQL 재학습. map_easy3 데이터 수집이 선행되므로 파일럿 범위 밖(별도 지시 시).

## 9. 신규 코드 배치 (전부 `irl/` 격리)

| 파일 | 역할 | 규모 |
|---|---|---|
| `irl/rank_episodes.py` | §2 지표 계산·전순서·페어 생성·80/20 분할·G1 리포트(JSON) | ~150줄 |
| `irl/reward_trex.py` | §3 보상망 학습(+held-out 평가, early stop) | ~200줄 |
| `irl/reward_otr.py` | §4 preprocessor+OT 정렬 보상 | ~200줄 |
| `irl/relabel.py` | §5 사이드카 생성(공통 CLI: `--method {trex,otr} --tag ...`) | ~120줄 |
| `irl/sanity_g2.py` | §5 G2 프로브(P1~P3) 리포트 | ~100줄 |
| 기존 수정 | `f1tenth_data.py`(+reward_dir), `train_iql.py`(+--reward_dir) | 각 ~15줄 |

venv: 보상망 학습·relabel·IQL = `f1tenth_IQL/.venv`(GPU). (비상시) 노이즈 롤아웃·map_easy3 평가 = `f1tenth_RL_project/.venv`. 두 프로젝트 학습·평가 독립성 불변.

## 10. 파일럿 일정 (단계적 병행, 실측 기반 추정)

| 순서 | 작업 | 시간 | 게이트 |
|---|---|---|---|
| M0 | 공통 인프라: 로더 reward_dir + 스모크(하위호환), rank_episodes | 구현 반나절 | 스모크 통과 |
| M1 | 랭킹 구축 + held-out 검증 | 실행 분 단위 | **G1 ≥80%** |
| M2 | T-REX 보상망 학습 + relabel + G2 | GPU ~0.5h | **G2-P1** |
| M3 | IQL 600k (trex_v1) — *이 83분 동안 OTR preprocessor 설계·구현 병렬* | GPU ~1.4h | — |
| M4 | eval_sweep + seed 견고성 | ~0.5–1h | **G3** |
| M5a | G3 통과 → L1 전이 비교(대조군은 기존 ckpt) → 종료 | ~0.5h | 승리조건 |
| M5b | G3 실패/아슬 → OTR relabel+IQL(+1.4h)로 정면 비교 → §7 진단 | +2–3h | — |
| — | (비상) G1 실패 → D-REX 노이즈 롤아웃 보강 | +1–2h 수집 | — |

**요약: 계산시간은 최선 ~4h/최악 ~9h(하룻밤), 지배 비용은 구현(M0+M2+OTR ≈ 1.5~2.5일).** 경로 A가 완전 offline이 되어(TROFI식) 벽시계·리스크 모두 이전 추정보다 감소.

## 11. 사전등록 요약 (사후 변경 금지 목록)

G1 임계 80% / G2-P1 경성 / G3 판정 ①②③ / L1 승리조건 / 랭킹 전순서(§2) / OTR 변형 V1·V2 / 세그먼트 L=50 기본. 이 값들을 바꾸면 "조정했음"을 결과와 함께 기록한다(체리피킹 방지).

## 12. 열린 항목 (구현 착수 시 확인)

- `eval_iql.py`의 map 인자 지원 여부(§8) — 없으면 소폭 추가(하위호환 유지).
- OTR 공식 rewarder의 정확한 보상 변환식(exp 스케일 등) — 코드에서 확정 후 §4에 추기.
- cap5_full(31ep 완주)을 랭킹 하위 계층으로 포함할지 — 완주군 내 속도 스펙트럼이 넓어져 G1에 유리할 가능성(포함 기본, M1에서 확인).

---

> ※ 구현·학습·수집은 사용자 명시 지시 후 착수. GPU 학습 run_in_background 필수. 산출물 덮어쓰기 금지(tag/번호 증분). 로그·모델·데이터 폐기 금지. commit/push는 지시 시에만.

---

## 부록 A — 사후 확정사항 (2026-07-03 append; 본문은 불변, 아래가 §8·§12 일부를 갱신)

사용자 결정으로 확정·변경된 내용을 본문 수정 없이 여기에 추가한다(append-only).

### A1. 방법 최종 확정 (본문 §0~§4의 우선순위를 결론으로 고정)

- **실행안 = 경로 A: T-REX 계열(TROFI 방식).** 기존 crash_data를 결과지표로 랭킹 → Bradley-Terry 보상망(MLP 256×2) → D4 데이터 relabel → 기존 IQL. 완전 offline, 롤아웃 0.
- OTR = **2번 카드**(G3 실패/아슬 시에만 IQL까지 실행). D-REX 노이즈 롤아웃 = **비상 보강**(G1 실패 시에만, rc 스폰). online(AIRL 등) = **최후 분기**(offline 레버 소진 시에만).

### A2. 전이 대상 변경: map_easy3 → **SOCHI** (§8 갱신)

- 근거: 사용자 지정(2026-07-03). **디스크 확인 완료** — `f1tenth_RL_project/f1tenth_gym_ros/maps/SOCHI.{yaml,png}` 및 `pkg/src/pkg/maps/SOCHI.{yaml,png}` 실재. 수집·설치 불요.
- 기본 = **plain SOCHI**. `SOCHI_OBS`(장애물 변형)는 보너스 실험 후보로만.
- §8의 "map_easy3"는 전부 SOCHI로 읽는다. L1 프로토콜·대조군(기존 D4 손설계 ckpt)·승리조건("학습보상 정책 완주율 ≥ 손설계 정책")은 불변.

### A3. L1 판정지표 보강 (§8 승리조건의 공허 판정 방지)

- SOCHI는 실제 서킷 기반이라 map_easy3보다 난도가 높아 **양쪽 정책 모두 zero-shot 완주 0**일 가능성이 있다. 그 경우 승리조건이 0 ≥ 0으로 공허해지므로, 판정지표를 계층화한다(사전등록):
  1. **1순위: 2랩 완주율(seed 0~3)** — 기존 승리조건.
  2. **2순위(완주 0 동률 시): 충돌 전 최장 주행거리**(Σ|state[:,0]×20|×0.02, §2와 동일 계측) 평균 비교.
  3. 기록용: 랩 이벤트 수, 충돌까지 시간.

### A4. §12 열린 항목 갱신

- `eval_iql.py`의 map 인자 확인 대상이 map_easy3 → SOCHI로 바뀜(확인 방법 동일). SOCHI yaml의 스폰 포즈·센터라인 웨이포인트 유무도 함께 확인(2순위 지표의 진행거리 계측엔 불필요하나, 향후 L2/수집 시 필요).

> 이 부록 이후의 결정도 본문 수정 없이 부록 B, C, …로 이어 붙인다.
