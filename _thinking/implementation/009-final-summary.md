# 009 — 프로젝트 최종 종합 (마무리)

> 2026-06-22. f1tenth offline RL(IQL 스티칭) 개인과제 종료 시점의 캡스톤 문서. 전 과정·결과·재현법·견고성을
> 한 곳에 종합. 상세는 001~008 참조. 목표 [[001-goal]].

---

## 0. 결론 (한 줄)
**충돌(불완전) 데이터만 IQL로 스티칭해 Oschersleben 2랩을 34.32s에 완주 — cap10 기록 56.14s를 21.82s,
baseline 107.16s를 72.84s 경신. 목표 G3 달성.** 완주(expert) 데이터 없이 데이터를 *능가*한 offline RL 정통 결과.

## 1. 최종 성능
| 지표 | 값 | 기준 | 판정 |
|---|---|---|---|
| 2랩 완주 | ✓ (best 600k) | G1: 완주 | **달성** |
| 2랩 시간 | **34.32s** (lap [17.78,16.54]) | G2: <107.16s | **+72.84s** |
| 2랩 시간 | **34.32s** | G3: <56.14s(cap10) | **+21.82s 경신** |
| 단일 best lap | 16.54s | Dreamer expert 16.6s | 동급 |
- best 정책: `runs/d4_iql_stitch_rc/checkpoint_600000.pt`

## 2. 무엇이 성공시켰나 (핵심 레시피)
1. **알고리즘**: CORL IQL vendor(in-sample expectile value + AWR). [[001-iql-pipeline-and-d2]]
2. **데이터 정렬**: DreamerV3 규약(action[t]→obs[t]) 실측 확정 — corr 0.99. 틀렸으면 전부 무음 파손.
3. **★ action 조화(harmonization)**: 저장 action은 정규화 [-1,1]인데 수집 v_max가 cap별로 달라(10/15/20)
   물리속도가 다름 → 공통 frame(v_common=20)으로 재정규화 `a1'=(a1+1)(v_collect+5)/(v_common+5)−1`.
   이게 직선 고속(cap20)+코너 감속(cap10)을 한 frame서 stitch 가능하게 한 전제. [[002-d3-stitching-result]]
4. **★ random-centerline 수집**(사용자 아이디어, 결정적): 기존 데이터는 단일 출발점·깨끗한 라인뿐 →
   정책이 죽는 다양 상태 미커버. 센터라인 랜덤 지점 스폰(v=0)으로 트랙 전역을 다양 상태로 덮음
   → lap2 견고성 회복. 이게 없으면 2랩 완주 0(D2/D3/T1 전부 lap2 충돌). [[006-random-centerline-collection]]
5. **보상/하이퍼**: γ=0.999(lap +100 bootstrap), collision_penalty −50, β=3.0, iql_τ=0.7. 큰 데이터(1.02M)로
   과적합 회피 → 최종 ckpt가 best.

## 3. 실험 여정 (요약)
| 실험 | 데이터 | 결과 | 교훈 |
|---|---|---|---|
| D2 | cap10 | lap1 28s@9.5m/s, lap2 충돌 | 첫코너 병목 극복, 느림 |
| D3 | cap10+15+20 | lap1 **18s@16m/s**, lap2 충돌 | 속도 stitch 성공, 견고성 부족 |
| T1 | +cp−100 | 충돌 위치만 이동 | reward 튜닝으론 못 품 [[004-t1-collision-penalty]] |
| 진단 | — | lap2서 off-line(~0.7m) 충돌, 데이터 미커버 | 병목=covariate gap [[003-crash-diagnosis]][[005-data-lever-coverage]] |
| **D4** | **+random-centerline(rc)** | **2랩 완주 34.32s** | **목표 달성** [[007-breakthrough-d4-record]][[008-d4-600k-robustness]] |

## 4. 견고성 (정직한 한계)
- **완주 체크포인트 4/30**(300k·360k·420k·600k) — 학습 전반 재현 = 우연 아님. best=600k 34.32s.
- **seed-robustness 3/4**(seed 0·2·3 완주, seed 1 충돌). ★ seed는 **출발 위치가 아니라 f110 라이다 스캔
  노이즈를 바꾼다**(실측: 출발 pose 전 seed 동일, lidar[:5] seed별 상이, 동역학은 결정론). 즉 취약점은
  **센서 노이즈 견고성** — 한 seed 안에선 100% 재현, 노이즈 시퀀스 따라 경계선서 완주/충돌 갈림.
- 완전 견고(대다수 ckpt·seed 완주) 미달 — 추가 옵션은 §6.

## 5. 산출물 / 재현
**코드**(f1tenth_IQL): `f1tenth_data.py`(loader+조화) · `train_iql.py` · `eval_iql.py` · `eval_sweep.py` ·
`diagnose_crash.py` · `collect_random_centerline.py` · `watch_iql.py` · `vendor/CORL`.
**2-venv**: 학습=`f1tenth_IQL/.venv`(torch 2.4.1+cu124), 평가/관람/수집=`f1tenth_RL_project/.venv`(f110_gym).
```bash
# 평가(텍스트): 2랩 완주·시간 확인
/home/dlacksdn/f1tenth_RL_project/.venv/bin/python eval_iql.py \
  --ckpt runs/d4_iql_stitch_rc/checkpoint_600000.pt --episodes 1
# 관람(실시간 창 + steer/속도 출력)
/home/dlacksdn/f1tenth_RL_project/.venv/bin/python watch_iql.py \
  --ckpt runs/d4_iql_stitch_rc/checkpoint_600000.pt --seed 0
# 전 체크포인트 스윕
/home/dlacksdn/f1tenth_RL_project/.venv/bin/python eval_sweep.py --run_dir runs/d4_iql_stitch_rc
```
데이터(RL_project): 원본 `crash_data/cap{10_full,15,20}` + 신규 `crash_data/cap{15,20}_rc`(930 ep). 정책 ckpt
출처는 [[006-random-centerline-collection]] §1.

## 6. 더 밀 경우 (선택, 과제 목표는 이미 충족)
- **센서 노이즈 robustness**: 학습 obs에 lidar 노이즈 주입(augmentation) → seed robustness↑.
- **off-line 커버리지 더**: 센터라인 랜덤 + lateral offset 스폰(005 §6) — 단 "jitter OFF" 지시와 충돌, 재논의 필요.
- **표현**: frame-stacking(시간맥락) / lidar 해상도↑.
- seed-앙상블/조기종료 ckpt 선택 정책.

## 7. 메모리
`~/.claude/.../memory/f1tenth-iql-success.md` 기록(다음 세션 인수인계).
