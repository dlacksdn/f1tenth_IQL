#!/usr/bin/env python3
"""
eval_iql.py — 학습된 IQL 정책을 f110_gym(Oschersleben 2랩)에서 평가.

★ 반드시 RL_project venv 로 실행(거기에 f110_gym + dreamerv3-torch 가 있음):
    /home/dlacksdn/f1tenth_RL_project/.venv/bin/python eval_iql.py --ckpt <run_dir 또는 .pt>

설계(정합성):
  - env: eval_gate.py 의 build_config + dreamer.make_env 를 그대로 재사용(데이터수집·baseline
    107.16s 와 동일 env). v_max 는 학습 시 조화 공통 frame(config.json.common_v_max, 기본 20)
    으로 강제 → 정책의 공통-frame action 이 올바른 물리속도로 환원됨.
  - obs: 학습과 동일한 build_obs(min-pool lidar 128 + state 5 = 133) 후 obs_norm.npz 로 정규화.
  - action: 정책 출력 [-1,1] 을 그대로 env.step({"action": a}) 에 전달(NormalizeActions 가 denorm).
  - 완주 판정: info['cause']=='lap_complete'(2랩). per-lap 시간 obs['log_lap_time_s']>0.

CORL 정책 클래스는 vendored iql.py 에서 import(d4rl/wandb/pyrallis 만 stub, gym 은 RL_project
real gym 유지 — f110_gym 이 의존하므로 gym 은 절대 stub 안 함).
"""
import argparse
import json
import os
import pathlib
import sys
from unittest.mock import MagicMock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RL_ROOT = "/home/dlacksdn/f1tenth_RL_project"
BASELINE_2LAP = 107.16   # cap-5 deterministic 2랩(plan_f1tenth.py) — G2 기준
CAP10_RECORD = 56.14     # cap10 완주 2랩 기록 — G3(stitching 으로 깨야 할 목표)
DT_WRAP = 0.02           # SIM_TIMESTEP(0.01) × action_repeat(2): env step 당 sim 시간

# ── import 경로 ──
#  ① vendored CORL(iql 정책 클래스) — d4rl/wandb/pyrallis 만 stub, gym 은 real 유지
sys.path.insert(0, os.path.join(HERE, "vendor", "CORL"))
sys.path.insert(0, HERE)  # f1tenth_data
for _m in ("d4rl", "wandb", "pyrallis"):
    sys.modules.setdefault(_m, MagicMock())
import iql as corl                         # noqa: E402
from f1tenth_data import build_obs, OBS_DIM  # noqa: E402
#  ② RL_project eval 인프라(build_config + make_env + 집계 함수)
sys.path.insert(0, os.path.join(RL_ROOT, "scripts"))
sys.path.insert(0, os.path.join(RL_ROOT, "vendor", "dreamerv3-torch"))
from eval_gate import build_config, aggregate_episodes, is_completed  # noqa: E402


def load_policy(run_dir, ckpt_path, device="cpu"):
    """config.json 으로 actor 아키텍처 복원 + ckpt['actor'] 로드. obs_norm(mean,std)·common_v_max 반환."""
    import torch
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    det = bool(cfg.get("iql_deterministic", False))
    hid, nh = int(cfg.get("hidden_dim", 256)), int(cfg.get("n_hidden", 2))
    common_v_max = float(cfg.get("common_v_max", 20.0))
    lidar_n = int(cfg.get("lidar_n", 128))
    sd = int(cfg.get("state_dim", OBS_DIM))
    if det:
        actor = corl.DeterministicPolicy(sd, 2, 1.0, hidden_dim=hid, n_hidden=nh)
    else:
        actor = corl.GaussianPolicy(sd, 2, 1.0, hidden_dim=hid, n_hidden=nh)
    ck = torch.load(str(ckpt_path), map_location=device)
    state = ck["actor"] if "actor" in ck else ck
    actor.load_state_dict(state)
    actor.to(device).eval()   # eval() → GaussianPolicy.act 가 dist.mean(결정적) 사용
    # obs 정규화 통계
    nz = np.load(os.path.join(run_dir, "obs_norm.npz"))
    mean, std = nz["mean"].astype(np.float32), nz["std"].astype(np.float32)
    return actor, mean, std, common_v_max, lidar_n, det


def run_episode(actor, env, mean, std, lidar_n, device="cpu", max_steps=100000):
    """단일 episode rollout(eval_gate.run_episode 의 IQL 버전). 학습과 동일 build_obs+정규화."""
    obs = env.reset()
    lap_times = []
    ep_return, length, cause = 0.0, 0, None
    while length < max_steps:
        o = build_obs({"lidar": obs["lidar"][None], "state": obs["state"][None]}, lidar_n)[0]
        o = ((o - mean) / std).astype(np.float32)
        a = actor.act(o, device=device)                 # [-1,1] (2,)
        obs, reward, done, info = env.step({"action": np.asarray(a, dtype=np.float32)})
        ep_return += float(reward)
        length += 1
        lt = float(obs.get("log_lap_time_s", 0.0))
        if lt > 0.0:
            lap_times.append(lt)
        if done:
            cause = info.get("cause")
            break
    return {"cause": cause, "lap_times": lap_times, "length": length, "return": ep_return}


def main():
    ap = argparse.ArgumentParser(description="IQL 정책 f110 평가(Oschersleben 2랩)")
    ap.add_argument("--ckpt", required=True,
                    help="체크포인트 .pt 또는 run 디렉터리(디렉터리면 latest.pt 사용)")
    ap.add_argument("--task", default="f1tenth_Oschersleben",
                    choices=["f1tenth_map_easy3", "f1tenth_Oschersleben"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--v_max", type=float, default=None,
                    help="env action-space 속도상한. 미지정=config.common_v_max(권장; 학습 조화 frame 과 일치)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="결과 JSON 경로(기본 run_dir/eval_iql_<task>_<tag>.json)")
    args = ap.parse_args()

    import torch
    ckpt = pathlib.Path(args.ckpt)
    if ckpt.is_dir():
        run_dir = str(ckpt)
        ckpt = ckpt / "latest.pt"
    else:
        run_dir = str(ckpt.parent)
    if not ckpt.exists():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt}")

    actor, mean, std, common_v_max, lidar_n, det = load_policy(run_dir, ckpt, args.device)
    v_max = args.v_max if args.v_max is not None else common_v_max

    # env: 골드스탠다드 경로 재사용 + v_max 강제(조화 frame 일치)
    config = build_config(args.task)
    config.v_max = v_max
    config.seed = args.seed
    from dreamer import make_env
    env = make_env(config, "eval", 0)

    print(f"[eval_iql] task={args.task} ckpt={ckpt} actor={'det' if det else 'gauss'} "
          f"v_max={v_max} lidar_n={lidar_n} episodes={args.episodes}", flush=True)

    episodes = []
    try:
        for i in range(args.episodes):
            res = run_episode(actor, env, mean, std, lidar_n, args.device)
            two_lap = sum(res["lap_times"]) if res["lap_times"] else None
            res["total_time_s"] = round(res["length"] * DT_WRAP, 3)
            res["two_lap_sum_s"] = round(two_lap, 3) if two_lap else None
            episodes.append(res)
            comp = "✓완주" if is_completed(res["cause"]) else " "
            print(f"[eval_iql] ep {i+1}/{args.episodes}: cause={res['cause']:<12} {comp} "
                  f"laps={[round(t,2) for t in res['lap_times']]} "
                  f"2lap={res['two_lap_sum_s']} total={res['total_time_s']}s len={res['length']}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    agg = aggregate_episodes(episodes)
    completed = [ep for ep in episodes if is_completed(ep.get("cause"))]
    two_laps = [ep["two_lap_sum_s"] for ep in completed if ep["two_lap_sum_s"]]
    best_2lap = min(two_laps) if two_laps else None
    med_2lap = float(np.median(two_laps)) if two_laps else None

    print("\n========== eval_iql 결과 ==========", flush=True)
    print(f"episodes        : {agg['n_episodes']}", flush=True)
    print(f"완주(2랩)       : {agg['n_completed']}  (완주율 {agg['completion_rate']:.3f})  [G1: >0]", flush=True)
    print(f"cause 분포      : {agg['cause_counts']}", flush=True)
    lm, lb = agg["lap_median"], agg["lap_best"]
    print(f"단일 lap median : {lm:.3f}s" if lm is not None else "단일 lap median : -", flush=True)
    print(f"단일 lap best   : {lb:.3f}s" if lb is not None else "단일 lap best   : -", flush=True)
    if best_2lap is not None:
        print(f"2랩 best        : {best_2lap:.3f}s   (median {med_2lap:.3f}s)", flush=True)
        print(f"  vs baseline {BASELINE_2LAP}s [G2]: {'PASS ✓' if best_2lap < BASELINE_2LAP else 'FAIL'} "
              f"({BASELINE_2LAP - best_2lap:+.2f}s)", flush=True)
        print(f"  vs cap10 기록 {CAP10_RECORD}s [G3]: {'PASS ✓✓ (기록 경신!)' if best_2lap < CAP10_RECORD else 'FAIL'} "
              f"({CAP10_RECORD - best_2lap:+.2f}s)", flush=True)
    else:
        print("2랩 완주 0건 — 시간 비교 불가(G1 미달).", flush=True)
    print("===================================\n", flush=True)

    tag = ckpt.stem
    out = pathlib.Path(args.out) if args.out else pathlib.Path(run_dir) / f"eval_iql_{args.task}_{tag}.json"
    payload = {
        "task": args.task, "ckpt": str(ckpt), "v_max": v_max, "episodes": args.episodes,
        "aggregate": agg, "best_2lap_s": best_2lap, "median_2lap_s": med_2lap,
        "baseline_2lap_s": BASELINE_2LAP, "cap10_record_s": CAP10_RECORD,
        "per_episode": episodes,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[eval_iql] JSON 저장: {out}", flush=True)


if __name__ == "__main__":
    main()
