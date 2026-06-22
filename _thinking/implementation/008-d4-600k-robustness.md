# 008 — D4 600k 연장 학습: 견고성 + best 갱신

> 2026-06-22 야간. D4를 300k→600k resume 학습해 robustness 확인. 실측. 선행: [[007-breakthrough-d4-record]]

---

## 0. 한 줄 요약
D4를 600k까지 학습하니 **완주 체크포인트 1/15 → 4/30 (300k·360k·420k·600k)**으로 늘고, **best가 34.32s
(step 600k)**로 갱신(300k 34.88s보다 빠름). G3 +21.82s 경신 재확인. 완주가 학습 전반에 재현 = 우연 아닌
실능력. 단 여전히 간헐적(대부분 ckpt 충돌)=경계선.

## 1. 스윕 결과 (30 ckpt, 결정론 1ep)
완주 체크포인트:
| step | 2랩 시간 | lap 분해 |
|---|---|---|
| 300k | 34.88s | [18.06, 16.82] |
| 360k | 35.06s | — |
| 420k | 35.70s | — |
| **600k** | **34.32s** ★ | [17.78, 16.54] (2ep 일관) |

나머지 26/30 충돌. best=600k 34.32s (G2 +72.84s, **G3 +21.82s 경신**).

## 2. 해석
- **robustness↑**: 완주 ckpt가 4개로 늘고 학습 후반(300~600k)에 분포 → 능력이 안정적으로 형성됨(단일
  운빨 아님). 더 학습이 D2/D3처럼 퇴화시키지 않음(큰 데이터 1.02M 덕 과적합 회피).
- **여전히 경계선**: 26/30은 충돌. 완주 ckpt와 충돌 ckpt가 번갈아 = 정책이 데이터 한계 근처에서
  미세차로 완주/충돌이 갈림. 완전 견고(대다수 ckpt 완주)는 아님.
- eval 결정론 → 각 ckpt는 항상 완주 or 항상 충돌(seed 변동 robustness는 별개 확인 가치).

## 3. 현 상태 / 다음(선택)
- **G3 목표 달성·재확인.** best 정책 `runs/d4_iql_stitch_rc/checkpoint_600000.pt` (34.32s).
- 완전 견고성을 원하면(대다수 ckpt 완주):
  - round2 rc 수집으로 off-line 절대량 더↑ (006의 lateral-spawn 옵션 포함 — 사용자 확인 후),
  - 또는 표현 업그레이드(frame-stacking),
  - 또는 seed 앙상블/조기종료 ckpt 선택 정책.
- 현 시점 **과제 목표는 충족** — 충돌 데이터 stitch만으로 cap10 expert를 21초 능가.

## 4. 파일
- best: `runs/d4_iql_stitch_rc/checkpoint_600000.pt` · eval JSON: `eval_iql_..._checkpoint_600000.json`
- 학습로그: `runs/d4_iql_stitch_rc.log`(0~300k) + `_ext.log`(300~600k) · 스윕: `sweep_f1tenth_Oschersleben.json`
