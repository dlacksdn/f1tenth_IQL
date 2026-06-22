#!/usr/bin/env python3
"""
collect_random_centerline.py — 센터라인 랜덤 지점 스폰 충돌 데이터 수집.

목적(implementation/005): 기존 cap15/20은 '한 출발점 + 깨끗한 라인'만 담아 우리 정책이 죽는
off-line 상태(라인서 ~0.7m 벗어남)를 안 덮는다. 매 ep를 **센터라인 위 랜덤 s에 v=0으로 스폰**해
트랙 전역을 다양한 진입상태로 덮어 그 갭을 메운다(사용자 지시: jitter는 안 켜고 '스폰 위치'만 랜덤).

env 물리/리워드 무수정 — reset(options={'pose':...}) 경로만 사용(검증됨: 임의 pose 스폰 오차 0).
검증된 collect_crash_data.collect_episode(정렬·npz 형식 100% 동일) 재사용. RL_project venv로 실행.

내장 품질 self-check: 저장된 충돌 ep의 하드코너(우리 충돌 s=7/15/49m ±5m) **lateral offset 분포**를
누적 → 주기적 로그. random-centerline이 실제로 ±0.7m off-line 을 덮는지 실시간 확인.
"""
import argparse
import os
import sys
import pathlib

import numpy as np

RL_ROOT = "/home/dlacksdn/f1tenth_RL_project"
sys.path.insert(0, os.path.join(RL_ROOT, "scripts"))
sys.path.insert(0, os.path.join(RL_ROOT, "vendor", "dreamerv3-torch"))

HARD_CORNERS = {"s7": 7.0, "s15": 15.0, "s49": 49.0}   # 우리 정책 충돌 지점(센터라인 s)
CRASH_LATERAL = 0.7    # 우리 정책이 충돌한 off-line 거리(목표: 데이터가 이만큼 덮어야)


def load_centerline():
    cl = np.loadtxt(os.path.join(RL_ROOT, "maps", "Oschersleben_centerline.csv"),
                    delimiter=",", skiprows=1)  # s,x,y,tx,ty
    return cl[:, 0], cl[:, 1:3], cl[:, 3:5], float(cl[-1, 0])


def main():
    ap = argparse.ArgumentParser(description="센터라인 랜덤 스폰 충돌 수집")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--v_max", type=float, required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="스폰 RNG seed(병렬수집 시 프로세스마다 다르게)")
    ap.add_argument("--lateral", type=float, default=0.0,
                    help="스폰 시 좌우 offset 범위(m). 0=센터라인 위(기본, 사용자 지시). >0이면 ±U.")
    ap.add_argument("--quality_every", type=int, default=40, help="N 충돌마다 lateral 커버리지 로그")
    ap.add_argument("--max_env_steps", type=int, default=None, help="배회 ep truncate(env step)")
    args = ap.parse_args()

    import tools
    from eval_gate import build_config, load_agent
    from collect_crash_data import collect_episode
    from scipy.spatial import cKDTree

    ckpt = pathlib.Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = pathlib.Path(RL_ROOT) / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    s_arr, xy, tang, L = load_centerline()
    tree = cKDTree(xy)
    rng = np.random.default_rng(args.seed)

    # config: stochastic rollout (collect_crash_data와 동일 규약)
    config = build_config("f1tenth_Oschersleben")
    config.eval_state_mean = False
    config.v_max = args.v_max
    config.device = "cpu"
    config.envs = 1
    config.parallel = False
    if args.max_env_steps is not None:
        config.time_limit = args.max_env_steps

    print(f"[rc-collect] ckpt={ckpt.name} v_max={args.v_max} episodes={args.episodes} "
          f"seed={args.seed} lateral={args.lateral} out={out_dir}", flush=True)
    agent, env = load_agent(config, ckpt)

    def collect_policy(obs, reset, state=None):
        return agent._policy(obs, state, training=True)

    def spawn_pose():
        s = float(rng.uniform(0.0, L))
        i = int(np.argmin(np.abs(s_arr - s)))
        x, y = float(xy[i, 0]), float(xy[i, 1])
        head = float(np.arctan2(tang[i, 1], tang[i, 0]))
        if args.lateral > 0.0:
            off = float(rng.uniform(-args.lateral, args.lateral))
            nx, ny = -np.sin(head), np.cos(head)
            x += off * nx
            y += off * ny
        return np.array([x, y, head], dtype=np.float32)

    def lateral_of(p):
        d, i = tree.query(p[:2])
        n = np.array([-tang[i, 1], tang[i, 0]])
        return float((p[:2] - xy[i]) @ n), float(s_arr[i])

    # 하드코너 lateral 누적(품질 self-check)
    lat_acc = {k: [] for k in HARD_CORNERS}
    n_collision = n_complete = 0
    n_other = {}

    try:
        for ep in range(args.episodes):
            pose = spawn_pose()
            cache, ep_id, cause, length = collect_episode(
                collect_policy, env, args.v_max, reset_options={"pose": pose})
            saved = False
            if cause == "collision":
                tools.save_episodes(out_dir, {ep_id: cache[ep_id]})
                n_collision += 1
                saved = True
                # 품질: 이 ep 전이들의 하드코너 lateral 누적(서브샘플)
                poses = np.asarray(cache[ep_id]["pose"])
                for j in range(0, len(poses), 2):
                    lat, sj = lateral_of(poses[j])
                    for k, sc in HARD_CORNERS.items():
                        if abs(((sj - sc + L / 2) % L) - L / 2) < 5.0:
                            lat_acc[k].append(lat)
            elif cause == "lap_complete":
                n_complete += 1
            else:
                n_other[str(cause)] = n_other.get(str(cause), 0) + 1
            del cache

            if (ep + 1) % 10 == 0 or saved:
                print(f"[rc-collect] ep {ep+1}/{args.episodes}: cause={cause} len={length} "
                      f"saved={saved} (collision={n_collision} complete={n_complete})", flush=True)

            # 주기적 품질 self-check
            if n_collision > 0 and n_collision % args.quality_every == 0 and saved:
                msg = []
                for k in HARD_CORNERS:
                    a = np.array(lat_acc[k])
                    if len(a):
                        cov07 = (np.abs(a) >= CRASH_LATERAL).mean() * 100
                        msg.append(f"{k}:n={len(a)} max|lat|={np.abs(a).max():.2f} "
                                   f"std={a.std():.2f} %≥0.7={cov07:.0f}")
                    else:
                        msg.append(f"{k}:n=0")
                print(f"[rc-quality] 충돌{n_collision}개 시점 하드코너 lateral 커버리지: "
                      + " | ".join(msg), flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    print("\n========== rc-collect 결과 ==========", flush=True)
    print(f"시도={args.episodes} 충돌저장={n_collision} 완주폐기={n_complete} 기타={n_other}", flush=True)
    print(f"수집률={n_collision/args.episodes:.3f}  out={out_dir}", flush=True)
    print("최종 하드코너 lateral 커버리지(목표: ±0.7m 덮기):", flush=True)
    for k, sc in HARD_CORNERS.items():
        a = np.array(lat_acc[k])
        if len(a):
            print(f"  {k}(s={sc:.0f}): n={len(a)} 범위[{a.min():+.2f},{a.max():+.2f}] "
                  f"std={a.std():.2f} %≥0.7={np.mean(np.abs(a)>=CRASH_LATERAL)*100:.0f}", flush=True)
    print("=====================================", flush=True)


if __name__ == "__main__":
    main()
