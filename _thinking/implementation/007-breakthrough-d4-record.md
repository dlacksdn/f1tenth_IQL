# 007 — ★★ 돌파: D4 2랩 완주 34.88s (G3 기록 경신)

> 2026-06-22 야간. random-centerline 데이터를 더한 D4 재스티칭이 **2랩 완주 + cap10 기록 경신**.
> 실측·재현 검증. 선행: [[006-random-centerline-collection]] · 목표 [[001-goal]]

---

## 0. 한 줄 요약
**D4(cap10+15+20+cap15_rc+cap20_rc, cp−50, 300k step)가 Oschersleben 2랩을 34.88s에 완주**
(lap [18.06, 16.82]). 3 에피소드 전부 동일 완주(결정론). **G1(완주)✓ · G2(<107.16s)✓ +72.28s ·
G3(<56.14s)✓✓ +21.26s — cap10 기록 경신.** 충돌(불완전) 데이터만 stitch해 데이터를 능가한
offline RL 정통 결과. **사용자의 random-centerline 수집 아이디어가 결정적이었다.**

## 1. 결과 (스윕 + 정밀 재검증)
- 스윕 15 ckpt 중 **step 300k = 1/15 완주**(나머지 충돌). best 2랩 34.88s.
- 재검증(eval_iql, 3 ep): 전부 `lap_complete`, laps=[18.06, 16.82], 2랩=34.88s, len=1744.
  - 단일 lap best=16.82s(=Dreamer expert 16.6s 수준), median 17.44s.

| 지표 | 값 | 기준 | 판정 |
|---|---|---|---|
| 2랩 완주 | ✓ (완주율 1.0) | G1: >0 | **달성** |
| 2랩 시간 | 34.88s | G2: <107.16 | **+72.28s** |
| 2랩 시간 | 34.88s | G3: <56.14(cap10) | **+21.26s 경신** |

## 2. 왜 됐나 (이전 분석 정정 포함)
- **off-line 복구 예시 절대량 2배**(s15 ≥0.6m: 70→168 등) → IQL value/policy가 라인 벗어난 상태서
  복구를 학습 → lap2의 off-line 충돌 극복.
- **★ 정정**: 006에서 "off-line *비율* 불변(~1%)이라 효과 의문"이라 했으나, 실제 이득은 **3개 하드코너
  국소 측정이 못 잡은 "트랙 전역 다양 상태 커버리지"**였다. random-centerline 스폰이 모든 구간을 다양한
  진입상태로 덮어 **전반적 견고성**을 올렸다 — **사용자의 원래 직관("트랙을 다양하게 덮자")이 내 좁은
  3코너 metric보다 옳았다.**
- **과적합 회피**: 데이터 1.02M(원본 591k +73%) → 300k step=~77 epoch(D2의 750 epoch 대비 적음) →
  D2/D3와 달리 **최종(300k) 체크포인트가 best**. 큰 데이터가 늦은 step까지 퇴화 안 시킴.

## 3. 의미
cap10=저속완주(56s), cap15/20=고속충돌, 완주(expert) 데이터는 **배제**. 그런데 IQL이 이들의 좋은 조각을
**stitch**해 **34.88s 완주 정책**을 만들었다 = 어떤 단일 데이터(또는 cap10 expert)도 못 한 성능. **"데이터를
매칭이 아니라 능가"하는 offline RL 취지를 정확히 실증.** (IRL이면 천장이 expert였을 것.)

## 4. 한계 / 주의
- **1/15 체크포인트만 완주** = 성공이 threshold 경계. 견고성(여러 ckpt·seed 완주) 확인 필요.
  → **D4 600k까지 resume 학습 중**(bb6ar8a2v): 더 학습이 robustness/속도 개선하는지 검증. 결과 → 008.
- 평가 결정론(1 ep=대표)이라 단일 ckpt 완주는 실재하나, seed·트랙 변동 robustness는 추가 확인 가치.

## 5. 다음
- 600k 결과로 robustness 판단. 부족하면 round2 rc 수집(절대량↑) 또는 더 학습.
- 현 상태로 **G3 목표는 이미 달성**. best ckpt: `runs/d4_iql_stitch_rc/checkpoint_300000.pt`.
- 평가 JSON: `runs/d4_iql_stitch_rc/eval_iql_f1tenth_Oschersleben_checkpoint_300000.json`.
```bash
/home/dlacksdn/f1tenth_RL_project/.venv/bin/python eval_iql.py \
  --ckpt runs/d4_iql_stitch_rc/checkpoint_300000.pt --episodes 3
```
