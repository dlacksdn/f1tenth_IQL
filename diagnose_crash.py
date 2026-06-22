#!/usr/bin/env python3
"""
diagnose_crash.py — IQL 정책이 트랙 '어디서' 충돌하는지 진단(plan/002 ①, 무료/학습0).

★ RL_project venv 로 실행. env 1회 빌드 후 여러 체크포인트를 굴리며:
  - 매 step 위치(log_pose_x/y)·속도(vel_x)·arclength(W._total_arclen) 기록
  - 충돌 시: 몇 번째 lap, lap 내 어느 지점(s/L_track, %), 좌표, 충돌속도
  - D2 vs D3 가 같은 코너서 깨지면 → 표현/특정코너 문제(H2), 제각각이면 → 튜닝여지(H3)
  - 트랙 센터라인 + 각 궤적 + 충돌지점(X) 플롯 저장(덮어쓰기 금지, -1/-2 증분)
"""
import os
import sys
from unittest.mock import MagicMock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor", "CORL"))
sys.path.insert(0, HERE)
for _m in ("d4rl", "wandb", "pyrallis"):
    sys.modules.setdefault(_m, MagicMock())
from eval_iql import load_policy, build_config  # noqa: E402
from f1tenth_data import build_obs  # noqa: E402

CKPTS = [
    ("D2-50k",  "runs/d2_iql_cap10/checkpoint_50000.pt"),
    ("D3-280k", "runs/d3_iql_stitch/checkpoint_280000.pt"),
    ("D3-300k", "runs/d3_iql_stitch/checkpoint_300000.pt"),
]


def find_inner(env, attr="_total_arclen"):
    """env 래퍼 그래프를 훑어 attr 을 가진 내부 객체(F110GymnasiumWrapper) 반환."""
    seen, stack = set(), [env]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if hasattr(o, attr):
            return o
        cands = [getattr(o, a, None) for a in ("env", "unwrapped", "_env", "_environment")]
        try:
            cands += list(vars(o).values())
        except TypeError:
            pass
        for v in cands:
            if v is not None and hasattr(v, "__dict__") and not isinstance(v, type):
                stack.append(v)
    return None


def main():
    sys.path.insert(0, "/home/dlacksdn/f1tenth_RL_project/scripts")
    sys.path.insert(0, "/home/dlacksdn/f1tenth_RL_project/vendor/dreamerv3-torch")
    from dreamer import make_env

    rd0 = os.path.dirname(CKPTS[0][1])
    _, _, _, cvm, ln0, _ = load_policy(rd0, CKPTS[0][1], "cpu")
    cfg = build_config("f1tenth_Oschersleben")
    cfg.v_max = cvm
    cfg.seed = 0
    env = make_env(cfg, "eval", 0)
    W = find_inner(env)
    assert W is not None, "F110GymnasiumWrapper(_total_arclen) 못 찾음"
    L = float(W.L_track)
    centerline = np.asarray(W._centerline_xy, dtype=np.float32)
    print(f"[diag] L_track={L:.1f}m  centerline_pts={len(centerline)}  v_max={cvm}", flush=True)

    results = {}
    for name, ck in CKPTS:
        run_dir = os.path.dirname(ck)
        actor, mean, std, _, ln, _ = load_policy(run_dir, ck, "cpu")
        obs = env.reset()
        xs, ys, spd = [], [], []
        lap_times = []
        while True:
            o = build_obs({"lidar": obs["lidar"][None], "state": obs["state"][None]}, ln)[0]
            o = ((o - mean) / std).astype(np.float32)
            a = actor.act(o, device="cpu")
            obs, r, done, info = env.step({"action": np.asarray(a, np.float32)})
            xs.append(float(obs.get("log_pose_x", W._raw_obs["poses_x"][0])))
            ys.append(float(obs.get("log_pose_y", W._raw_obs["poses_y"][0])))
            spd.append(float(W._raw_obs["linear_vels_x"][0]))
            lt = float(obs.get("log_lap_time_s", 0.0))
            if lt > 0:
                lap_times.append(round(lt, 2))
            if done:
                break
        arclen = float(W._total_arclen)
        lap = int(arclen // L)
        s_in = arclen % L
        pct = s_in / L * 100.0
        pre = float(np.mean(spd[-10:])) if len(spd) >= 10 else float(np.mean(spd))
        results[name] = dict(cause=info.get("cause"), steps=len(xs), n_laps=len(lap_times),
                             lap_at_crash=lap, s_in_lap=round(s_in, 1), pct=round(pct, 1),
                             crash_xy=(round(xs[-1], 2), round(ys[-1], 2)),
                             v_crash=round(spd[-1], 2), v_pre=round(pre, 2),
                             xs=xs, ys=ys, spd=spd)
        print(f"[{name}] cause={info.get('cause')} steps={len(xs)} lap완주={len(lap_times)}  "
              f"→ 충돌: lap{lap+1}, s={s_in:.0f}/{L:.0f}m ({pct:.0f}%), "
              f"xy=({xs[-1]:.1f},{ys[-1]:.1f}), v_직전10={pre:.1f}m/s", flush=True)
    env.close()

    print("\n=== 충돌 지점 비교 ===", flush=True)
    for n, r in results.items():
        print(f"  {n}: lap{r['lap_at_crash']+1}의 {r['pct']:.0f}% 지점(s={r['s_in_lap']:.0f}m), "
              f"직전속도 {r['v_pre']:.1f}m/s", flush=True)
    pcts = {n: r["pct"] + r["lap_at_crash"] * 100 for n, r in results.items()}  # lap 포함 누적%
    print(f"  → lap 내 위치(%) 차이로 '같은 코너인지' 판단: { {n: round(r['pct']) for n,r in results.items()} }",
          flush=True)

    # ── 플롯(덮어쓰기 금지) ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(centerline[:, 0], centerline[:, 1], "k--", lw=0.8, alpha=0.4, label="centerline")
    colors = {"D2-50k": "tab:blue", "D3-280k": "tab:red", "D3-300k": "tab:orange"}
    for n, r in results.items():
        c = colors.get(n, "gray")
        ax.plot(r["xs"], r["ys"], color=c, lw=1.6, alpha=0.85,
                label=f"{n}: {r['cause']} lap{r['lap_at_crash']+1} {r['pct']:.0f}% (v≈{r['v_pre']:.0f})")
        ax.plot(r["xs"][-1], r["ys"][-1], marker="X", color=c, ms=18, mec="k", mew=1.5)
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9)
    ax.set_title("Oschersleben — IQL 정책 주행 궤적 + 충돌 지점(X)")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    out = "runs/diag_crash_map.png"
    k = 1
    while os.path.exists(out):
        out = f"runs/diag_crash_map-{k}.png"
        k += 1
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n[diag] 플롯 저장: {out}", flush=True)

    npz = "runs/diag_trajectories.npz"
    k = 1
    while os.path.exists(npz):
        npz = f"runs/diag_trajectories-{k}.npz"
        k += 1
    np.savez(npz, centerline=centerline,
             **{f"{n}_{key}": np.asarray(results[n][key]) for n in results for key in ("xs", "ys", "spd")})
    print(f"[diag] 궤적 저장: {npz}", flush=True)


if __name__ == "__main__":
    main()
