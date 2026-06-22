#!/usr/bin/env python3
"""
watch_iql.py — 학습된 IQL 정책의 주행을 **실시간 창**으로 관람(2랩 완주 시연).

★ RL_project venv 로 실행(f110_gym + dreamerv3-torch + 렌더):
    cd /home/dlacksdn/f1tenth_IQL
    /home/dlacksdn/f1tenth_RL_project/.venv/bin/python watch_iql.py \
        --ckpt runs/d4_iql_stitch_rc/checkpoint_600000.pt --seed 0

env/물리/맵 무수정 — make_env 체인을 그대로 쓰고 내부 F110Env.render()만 호출(watch_drive.py 방식).
정책·obs·평가 규약은 eval_iql.py 와 100% 동일(load_policy/ build_obs/ [-1,1] action). 렌더만 추가.

표시(WSL2): 렌더 창은 디스플레이(WSLg 또는 X 서버)가 필요. 창이 안 뜨면 WSLg 활성/`echo $DISPLAY` 확인.
seed 0/2/3 은 best(600k)가 2랩 완주, seed 1 은 lap2 충돌 — 완주를 보려면 0(기본)/2/3 사용.
"""
import argparse
import os
import sys
import pathlib
from unittest.mock import MagicMock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RL_ROOT = "/home/dlacksdn/f1tenth_RL_project"
sys.path.insert(0, os.path.join(HERE, "vendor", "CORL"))
sys.path.insert(0, HERE)
for _m in ("d4rl", "wandb", "pyrallis"):
    sys.modules.setdefault(_m, MagicMock())
from eval_iql import load_policy  # noqa: E402  (정책 로드·정규화 통계, eval과 동일)
from f1tenth_data import build_obs  # noqa: E402
sys.path.insert(0, os.path.join(RL_ROOT, "scripts"))
sys.path.insert(0, os.path.join(RL_ROOT, "vendor", "dreamerv3-torch"))
from eval_gate import build_config  # noqa: E402
from f110_gym.envs.f110_env import F110Env  # noqa: E402


def find_f110(env):
    """make_env 체인을 .env/._env 로 내려가 내부 F110Env(render 대상)를 찾는다."""
    e, seen = env, set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, F110Env):
            return e
        e = getattr(e, "env", None) or getattr(e, "_env", None)
    raise RuntimeError("체인에서 F110Env 를 못 찾음")


def main():
    ap = argparse.ArgumentParser(description="IQL 정책 주행 실시간 관람")
    ap.add_argument("--ckpt", default="runs/d4_iql_stitch_rc/checkpoint_600000.pt")
    ap.add_argument("--task", default="f1tenth_Oschersleben")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="0/2/3=완주, 1=lap2 충돌(best 600k 기준)")
    ap.add_argument("--mode", default="human", choices=["human", "human_fast"])
    ap.add_argument("--v_max", type=float, default=None, help="미지정=config.common_v_max(권장)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = pathlib.Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = pathlib.Path(HERE) / ckpt
    run_dir = str(ckpt.parent)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    actor, mean, std, common_v_max, lidar_n, det = load_policy(run_dir, ckpt, args.device)
    v_max = args.v_max if args.v_max is not None else common_v_max

    config = build_config(args.task)
    config.v_max = v_max
    config.seed = args.seed
    from dreamer import make_env
    env = make_env(config, "eval", 0)
    f110 = find_f110(env)

    print(f"[watch] ckpt={ckpt.name} task={args.task} seed={args.seed} v_max={v_max} mode={args.mode}", flush=True)
    print("[watch] 실시간 창을 띄웁니다. (창이 안 뜨면 WSLg/X 디스플레이 확인)", flush=True)

    DT = 0.02
    for ep in range(args.episodes):
        obs = env.reset()
        lap_times, steps = [], 0
        while True:
            o = build_obs({"lidar": obs["lidar"][None], "state": obs["state"][None]}, lidar_n)[0]
            o = ((o - mean) / std).astype(np.float32)
            a = actor.act(o, device=args.device)
            # 실제 종방향 속도(직전 obs의 state[0]=vel_x/20) — 명령과 비교용
            vel_x = float(obs["state"][0]) * 20.0
            obs, r, done, info = env.step({"action": np.asarray(a, np.float32)})
            # 정규화 action [-1,1] → 물리값(NormalizeActions 역매핑, watch_drive.py 방식)
            steer = float(a[0]) * 0.4189                              # [-1,1]→[S_MIN,S_MAX] rad
            speed_cmd = (float(a[1]) + 1.0) / 2.0 * (v_max + 5.0) - 5.0  # [-1,1]→[V_MIN,v_max] m/s
            print(f"[drive] steer={steer:+.3f} rad ({np.degrees(steer):+5.1f}°)  "
                  f"속도명령={speed_cmd:5.1f} m/s  실제={vel_x:5.1f} m/s  "
                  f"(norm a=[{a[0]:+.2f},{a[1]:+.2f}])", flush=True)
            try:
                f110.render(mode=args.mode)
            except Exception as exc:
                print(f"[watch] 렌더 종료/실패: {exc}\n"
                      f"        창을 닫았거나 디스플레이(WSLg/X)가 없을 수 있습니다.", flush=True)
                env.close()
                return
            steps += 1
            lt = float(obs.get("log_lap_time_s", 0.0))
            if lt > 0.0:
                lap_times.append(round(lt, 2))
                print(f"[watch]   ✓ lap {len(lap_times)} 완주: {lt:.2f}s", flush=True)
            if done:
                c = info.get("cause")
                tot = sum(lap_times)
                ok = "🏁 2랩 완주!" if c == "lap_complete" else "충돌"
                print(f"[watch] 에피소드 {ep+1}: {ok}  cause={c}  laps={lap_times}  "
                      f"2랩={tot:.2f}s  주행시간={steps*DT:.1f}s", flush=True)
                break
    env.close()
    print("[watch] 관람 종료.", flush=True)


if __name__ == "__main__":
    main()
