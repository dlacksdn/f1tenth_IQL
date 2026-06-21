# 001 — IQL 실행 계획 (코드 근거 기반)

> 2026-06-22. 목표(goal/001) = **cap10 완주 토대 + cap15/20 고속충돌 재료로 IQL stitching → cap10 기록(56s) 격파**.
> CORL `iql.py` 실제 코드를 라인 단위로 읽어 통합 지점을 확정하고, 데이터로더·보상·실험·평가를 설계 후
> 적대적으로 검증한 계획. append-only. 엄밀(파일/라인/하이퍼파라미터) + 쉽게(표·근거). 구현 미실행(기록만).

---

## 0. 목표 = 성공 사다리 (낮은 단부터)
| 게이트 | 기준(20ep 결정론 2랩 eval, v_max=20) | 의미 |
|---|---|---|
| **G1 (전제)** | 완주율 > 0 (≥1/20), 강하게는 ≥0.5 | **BC가 못 넘은 0% 벽 돌파** = IQL의 in-sample value가 covariate shift를 넘었다는 증거 |
| **G2 (최소 승리)** | best 2랩 < **107.16s** + 완주율 ≥0.5 | 베이스라인 cap5 격파 (완주만 하면 거의 자동, cap10 라인=56s) |
| **G3 (★ 진짜 목표)** | best 2랩 < **56.14s**(stretch <53.66s) | 고속충돌을 완주 라인에 stitching = 진짜 offline 개선 |

→ **G1이 모든 것의 분기점.** G1 통과 못 하면 병목은 알고리즘이 아니라 데이터/정밀도(§9).

## 1. 큰 그림 — 4개 부품
```
[A] 환경: f1tenth_IQL 전용 venv + vendor/CORL(복사) + shim(d4rl/gym/wandb import 차단)
[B] 데이터로더 f1tenth_data.py: npz → CORL 5-key dict {obs, act, rew, next_obs, done}
[C] 학습 train_iql.py: vendored iql.py의 클래스 재사용 + 교체점 5개 + 촘촘 ckpt + 로깅
[D] 평가 eval_iql.py: RL_project .venv에서 별도 프로세스로 f110 2랩 (학습과 분리)
```
핵심: **학습은 순수 offline(env 무접촉), 평가만 f110_gym(RL_project venv)에서 별도 실행.** 둘의 유일한 결합점은
`actor.act(obs133)->[-1,1]` 인터페이스다.

## 2. [A] 환경 셋업 (venv + vendor)
- **신규 전용 venv** `/home/dlacksdn/f1tenth_IQL/.venv` (python3.10 또는 3.8). 설치는 **최소**:
  `torch==2.4.1+cu124`(이 박스 4060 Ti·CUDA12.4에서 검증됨), `numpy<2`, `tqdm`, `pyrallis`, `tensorboard`.
  → **CORL의 d4rl/mujoco-py/gym0.23/jax/wandb는 일부러 제외**(cu124에서 빌드 불가·불필요).
- **vendor**: `/home/dlacksdn/CORL/algorithms/offline/{iql.py, any_percent_bc.py, td3_bc.py}`를
  `/home/dlacksdn/f1tenth_IQL/vendor/CORL/`로 **복사**(LICENSE·출처 헤더 보존). **상류 `/home/dlacksdn/CORL`은 무수정.**
- **shim 재사용 방식**: vendored `iql.py`는 상단에서 `import d4rl, gym, wandb`라 그대로 실행 불가. → `train_iql.py`가
  그 3개를 **stub**한 뒤 iql.py의 클래스(`ImplicitQLearning, TwinQ, ValueFunction, GaussianPolicy,
  DeterministicPolicy, MLP, ReplayBuffer, soft_update, asymmetric_l2_loss, compute_mean_std`)를 **그대로 import**.
  → CORL의 IQL 수식을 **검증된 채로 재사용**(우리가 다시 짜지 않음).
- 규칙: **GPU 학습은 run_in_background 전용**(foreground+CUDA=exit144), kill은 단독 명령.

## 3. [B] 데이터로더 `f1tenth_data.py` (npz → CORL 5-key dict)
CORL `ReplayBuffer.load_d4rl_dataset(dict)`(iql.py L154-170)는 **transition 단위**라 정확히 5키만 필요:
`observations(N,133)·actions(N,2)·rewards(N,)·next_observations(N,133)·terminals(N,)`. d4rl 경로는 **완전 우회**.

- **obs = `concat(min-pool(lidar,128), state5)` = 133D.** (full 1085D는 replay prealloc ~19GB→불가; 133D ~0.3GB.)
  min-pool 다운샘플은 prior의 `_downsample_lidar`(`vendor/diffuser/diffuser/datasets/f1tenth.py:80`, 섹터 내 최근접
  빔 보존=레이싱 안전)를 **그대로** 재사용. **평가도 반드시 동일 함수**로 obs 생성.
- **action**: npz `action`이 **이미 [-1,1]** → 그대로 저장(`max_action=1.0`). 역정규화는 **평가 때만**(§5).
- **transition 생성**: **파일별·에피소드별로** `is_terminal`/`is_last`로 경계를 끊고, 각 ep 내 `t∈[0,T-2]`에서
  `(obs_t, act_t, rew_t, obs_{t+1}, done_t)`. **next_obs가 다른 에피소드로 넘어가면 안 됨.** 마지막 행 drop
  (next 없음 + NaN action 방어; 단 cap10/15/20 702파일 NaN 0개 검증됨).
- **★ terminals = `is_terminal` 정확히**: 충돌=True(흡수상태), **완주=False**(→ value가 +100 lap 보상을 뒤로
  bootstrap). 잘못 표시하면 value 부트스트랩이 깨진다. 로더에서 assert + D0에서 단위검증.
- **보상 설계**(modify_reward는 d4rl 이름 게이트라 우리에겐 no-op → **로더에서 직접 reshape**):
  - 기본(D2): raw 그대로 `progress(~0.27) + lap(+100) + collision(−10)`, **γ=0.999**(config.discount).
  - D3 튜닝 노브: **충돌 −10은 약함**(RTG 부호역전이 ≤−50에서, −200=100%) → `{−10, −50, −200}` 스윕.
    **+100 lap 스파이크는 평소의 ~370배** → q_mean 폭주 시 클립/정규화.
- **정규화**: 133D obs를 `compute_mean_std(eps=1e-3)`로 z-정규화, **mean/std를 `obs_norm.npz`로 ckpt 디렉터리에
  영속화**(평가가 동일 정규화 재사용 — 불일치 시 스폰 즉시 충돌). next_obs에도 동일 적용.
- **데이터 믹스**: D2=cap10_full만 / D3=cap10_full(완주 토대 30) + cap15(371충돌, 18m/s) + cap20(291충돌, 20m/s)
  ≈591k transition. (D3b 옵션: cap8_jitter 26완주 = 코너 커버리지.)

## 4. [C] 학습 `train_iql.py` — iql.py 재사용 + 교체점 5개
iql.py `train(config)`(L540-662)에서 **교체할 곳만**:
1. **데이터/차원**(L542-547): `gym.make`/`d4rl.qlearning_dataset` 삭제 → `state_dim=133, action_dim=2,
   dataset=load_f110_dataset(...)`.
2. **max_action**(L572): `env.action_space.high[0]` → **`1.0`**(이미 정규화).
3. **eval**(L639-648): `eval_actor`/`env.get_normalized_score` 삭제 → 인라인 eval 제거(평가는 §5 별도 프로세스).
4. **dense ckpt**(L655-659): eval_freq에 묶인 저장을 분리 → `ckpt_freq`(예 2e4)마다 **무조건** 저장(§아래).
5. **d4rl/gym/wandb 결합 제거**: import stub + `WANDB_MODE=offline`(또는 wandb 제거), TensorBoard+JSONL 로깅.
- **그대로 두는 핵심 수식**(검증 앵커): `_update_q` TD타깃 `rewards + (1-terminals)*discount*next_v`(L454) —
  terminals 정확성 필수. `_update_v`=expectile `asymmetric_l2_loss(adv, iql_tau)`. `_update_policy`=
  **advantage-weighted BC** `exp(beta*adv).clamp(max=100)*bc_loss`(L472) → **명시적 BC prior 없이 고-advantage
  transition만 증폭** = 우리가 노리는 stitching 메커니즘.
- **촘촘 체크포인트**(CLAUDE.md 규칙 + verify-before-kill): `trainer.state_dict()`(qf·vf·actor·옵티마이저·
  actor_lr_schedule·total_it 포함=resume 충분)를 `ckpt_freq`마다 **번호 붙여 보존**(`checkpoint_<step>.pt`,
  덮어쓰기 금지) + `latest.pt`만 롤링 갱신. 첫 저장 시 `obs_norm.npz`+`config.yaml` 같이. resume=`--resume latest.pt`
  → 끊겨도 최대 ckpt_freq step만 손실. (ckpt <10MB, 디스크 884GB 여유.) **kill 전 디스크 state>0 확인.**
- **로깅(run_logs/ 영구보존)**: train JSONL(1000step마다 value/q/actor loss, adv/q/v mean, lr) + TensorBoard +
  eval_history.jsonl + stdout tee. 이름 충돌 시 `-1/-2` 증분(덮어쓰기 금지).

## 5. [D] 평가 `eval_iql.py` — RL_project venv, 별도 프로세스
- 실행: `cd /home/dlacksdn/f1tenth_RL_project && .venv/bin/python /home/dlacksdn/f1tenth_IQL/eval_iql.py
  --ckpt <iql.pt> --obs_norm <obs_norm.npz> --task f1tenth_Oschersleben --episodes 20 --v_max 20`.
- `eval_gate.py` 패턴 재사용: `dreamer.make_env(config,'eval',0)`(wrapped env: dict obs{lidar1080,state5},
  NormalizeActions, 2랩, done=terminated|truncated). **device=cpu·precision=32**(백그라운드 GPU 학습과 비경합).
- 매 step: `obs133 = normalize(concat(_downsample_lidar(lidar,128), state5), mean, std)` → `a = actor.act(obs133)`
  ∈[-1,1] → **`env.step(a)` 그대로**(★ wrapped env의 NormalizeActions가 내부에서 [S,v_max]로 역정규화 = 수집 때와
  동일 → **수동 역정규화 금지**, v_max=20 헤드룸에서 a1=+1→20m/s·a1=0→7.5m/s, ~10m/s 명령 시 ~56s 주행).
- **결정론 metric**(GaussianPolicy `dist.mean`): ① 완주율(`info['cause']=='lap_complete'`) ② 2랩 시간
  (`log_lap_time_s>0` 합) ③ **첫 코너 생존 step**(종료 길이+원인 분포 — 완주율 0일 때 "정밀도 vs 알고리즘" 진단).
  ckpt당 JSON + `run_logs/eval_history.jsonl`. 최고 ckpt에서 stochastic(sample) 1회로 결정론 갭 보고.

## 6. 실험 사다리 (싸게-먼저, 병목부터 가른다)
| 단계 | 셋업 | 결정 규칙(무엇을 알게 되나) |
|---|---|---|
| **D0 스모크**(~5분) | IQL@cap10, 2000 step, eval 3ep | 파이프라인 무결성. NaN/차원오류→인프라 고침. 스폰 즉사(len 1~5)→정규화/부호 버그. 차가 움직임(len>50)→D1. |
| **D1 BC@cap10** | vendored `any_percent_bc.py`(shim), 동일 obs/eval, 1e5 step | **BC 0% 벽 재확인**(대조군). 예상 0%·~66step 첫코너사. >0%면 새 스택이 prior와 다름→재기준. |
| **D2 IQL@cap10**(핵심 베팅) | γ=0.999, iql_tau=0.7, beta=3, GaussianPolicy, 5e5 step, 20ep | **G1 게이트.** BC=0%인데 **IQL 완주>0 → ★in-sample value가 벽을 넘음**(중심가설 확증). IQL도 0%·같은 66step사→**병목=정밀도/데이터**(알고리즘 아님)→D3/데이터레버. 더 오래 버팀(150 vs 66)→value 효과 있으나 부족→γ/beta/충돌 튜닝. 완주하나 느림(~100s)→D3로 고속 재료. |
| **D3 IQL@cap10+15+20**(★ 진짜 목표) | D2 + cap15·cap20 고속충돌(~591k) | **G2/G3.** 완주+D2보다 빠름(→<56s)→**★★ stitching 성공**(진짜 개선). 완주하나 안 빨라짐→충돌페널티/beta/iql_tau/보상정규화 튜닝. **추가가 완주를 악화**(충돌오염 지배)→D2 + 고속 prefix만(cap15/20을 lap-1 직전까지 잘라 사용)으로 폴백. D2·D3 둘 다 같은 코너서 실패→offline RL 단독 한계(정직 보고). |

## 7. 하이퍼파라미터 (시작값 + D3 튜닝 노브)
| 항목 | 시작값 | 비고 |
|---|---|---|
| discount(γ) | **0.999** | ★검증필수(0.99=완주 안 보임). |
| iql_tau(expectile) | 0.7 | D3에서 0.8~0.9↑=낙관적 value=고속 선호. |
| beta(actor inv-temp) | 3.0 | D3에서 6~10↑=Q-greedy stitching 강제. |
| tau(Polyak)·batch·lr | 0.005·256·3e-4 | 기본. |
| actor | GaussianPolicy | eval은 dist.mean. 과적합 시 dropout 0.1. |
| max_timesteps | 5e5 | actor CosineLR 지평. |
| normalize(state) | True | mean/std 영속화 필수. |
| normalize_reward | False | 보상 reshape는 로더에서. |
| collision penalty | −10→{−50,−200} | D3 스윕(RTG 부호역전 임계 ≈−50). |

## 8. 성공 게이트 → §0 표 (G1 완주>0 / G2 <107.16s / G3 <56.14s). 완주0이면 **첫코너 생존 step**으로 정밀도-vs-알고리즘 진단.

## 9. ★ 적대적 검증 — 정직한 전망·리스크·폴백
**(a) G3(<56s)는 진짜 목표지만 가장 불확실하다.** 데이터에 **안전한 고속(v_max≥15) 완주가 0개**다. cap15/20은
전부 충돌이라, IQL이 그 고속 조각을 cap10 라인에 이어붙여 **닫힌 루프에서 안전하게** 실행해야 하는데, value는
**"속도 선호(완주 아님)" 편향**이 있어 고속을 물려다 첫 코너서 충돌할 위험이 크다(D3가 D2보다 나빠질 수 있음).
→ **현실적 1차 산출물 = G1+G2(완주 + 107s 격파) + "stitching이 되는가"의 정량 분석.** G3는 야심 목표로 두되 실패
가능성을 인정하고, 실패 시 그 자체가 강한 보고 자료(왜 offline stitching이 어려운가)다.

**(b) 최대 리스크 = G1조차 못 넘을 가능성.** 진짜 병목은 **폐루프 첫 코너 정밀도**(알고리즘 무관). BC가 모든 설정서
0/10이었다. IQL의 in-sample이 이를 넘는다는 보장은 **없다(미검증)**. → **D2가 프로젝트의 운명을 가른다.** D2가 0%·
같은 66step이면 답은 알고리즘이 아니라 **데이터**(v_max≤10 코너 커버리지 추가수집, cap8_jitter류) — 이때 데이터 레버로
전환. D2가 더 오래 버티면(부분 작동) γ/beta/충돌페널티 튜닝.

**(c) 기타 리스크/완화**:
- **충돌오염**: done=is_terminal 오표시→부트스트랩 붕괴. → 로더 assert + D0 검증(완주=False, 충돌=True 검증됨).
- **train/eval obs·action 불일치**: 동일 `_downsample_lidar` + `obs_norm.npz` 재사용, **action 이중역정규화 금지**
  (wrapped env가 처리). D0가 "차 움직임(len>50)"으로 조기 검출.
- **γ=0.999 불안정**: +100 스파이크로 Q 폭주 가능 → q_mean/value_loss 모니터, 폭주 시 lap 보상 클립, 촘촘 ckpt resume.
- **venv 의존성 지옥**: CORL이 torch1.11+cu113+d4rl 핀 → 신규 minimal venv(torch2.4.1+cu124) + 3개 import stub.

## 10. 절대 규칙 체크리스트
- [ ] 순수 offline(학습 중 env 무접촉) · Dreamer/f110_gym 코어·물리·reward·맵 무변경 · 상류 CORL 무수정.
- [ ] GPU=run_in_background, kill 단독, **kill 전 디스크 ckpt(state>0) 확인**, 촘촘 저장(번호 보존+latest 롤링).
- [ ] 평가=RL_project `.venv`(cpu), V_MAX=20·2랩·Oschersleben. 로그·모델 폐기금지(run_logs/, 번호증분).
- [ ] _thinking append-only(목표 바뀌면 goal/002). 코드구현·commit·push는 사용자 지시 시만.

## 11. 참조
- 현황: [[001-status-synthesis]] (analysis) / 목표: [[001-goal]] (goal)
- CORL: `vendor/CORL/iql.py`(상류 `/home/dlacksdn/CORL/algorithms/offline/iql.py`). 재사용 클래스=`ImplicitQLearning`
  외. 교체점 L542-547·572·639-648·655-659 + import stub.
- 재사용: 로더 `_downsample_lidar`(`...diffuser/datasets/f1tenth.py:80`) · 평가 `f1tenth_RL_project/scripts/eval_gate.py`
- 데이터/정책: `f1tenth_RL_project/runs/crash_data/` · `runs/*_oschersleben/`
- 산출 파일(예정): `f1tenth_IQL/{train_iql.py, eval_iql.py, f1tenth_data.py, vendor/CORL/, .venv, run_logs/}`
