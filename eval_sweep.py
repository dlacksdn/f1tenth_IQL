#!/usr/bin/env python3
"""
eval_sweep.py — 한 run 의 여러 체크포인트를 f110 에서 일괄 평가(env 1회 빌드 재사용).

★ RL_project venv 로 실행:
    /home/dlacksdn/f1tenth_RL_project/.venv/bin/python eval_sweep.py --run_dir runs/d2_iql_cap10

eval 이 결정론적(정책 mean + env reset 고정)이라 기본 1 에피소드. env 는 한 번만 빌드해
체크포인트별 actor 만 교체(체크포인트 스윕 비용 ↓). 결과: step 오름차순 요약표 + best 선정.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_iql import (load_policy, run_episode, build_config,  # noqa: E402
                      BASELINE_2LAP, CAP10_RECORD, DT_WRAP)
from eval_gate import is_completed  # noqa: E402


def _step_of(path):
    m = re.search(r"checkpoint_(\d+)\.pt", os.path.basename(path))
    return int(m.group(1)) if m else (10**12 if path.endswith("latest.pt") else -1)


def main():
    ap = argparse.ArgumentParser(description="IQL 체크포인트 스윕 평가")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", default="f1tenth_Oschersleben")
    ap.add_argument("--episodes", type=int, default=1, help="결정론이라 기본 1")
    ap.add_argument("--v_max", type=float, default=None, help="미지정=config.common_v_max")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpts", default="numbered",
                    help="'numbered'(checkpoint_*.pt) | 'all'(+latest) | glob 패턴")
    args = ap.parse_args()

    run_dir = args.run_dir
    if args.ckpts == "numbered":
        files = glob.glob(os.path.join(run_dir, "checkpoint_*.pt"))
    elif args.ckpts == "all":
        files = glob.glob(os.path.join(run_dir, "*.pt"))
    else:
        files = glob.glob(args.ckpts)
    files = sorted(files, key=_step_of)
    if not files:
        raise FileNotFoundError(f"체크포인트 없음: {run_dir}")

    # 정규화/아키텍처는 run 공통 — 첫 체크포인트로 메타 확보 후 env 1회 빌드
    _, mean, std, common_v_max, lidar_n, det = load_policy(run_dir, files[0], args.device)
    v_max = args.v_max if args.v_max is not None else common_v_max
    config = build_config(args.task)
    config.v_max = v_max
    config.seed = 0
    from dreamer import make_env
    env = make_env(config, "eval", 0)
    print(f"[sweep] run={run_dir} task={args.task} v_max={v_max} actor={'det' if det else 'gauss'} "
          f"ckpts={len(files)} eps={args.episodes}", flush=True)

    rows = []
    try:
        for f in files:
            actor, mean, std, _, lidar_n, _ = load_policy(run_dir, f, args.device)
            eps = []
            for _ in range(args.episodes):
                res = run_episode(actor, env, mean, std, lidar_n, args.device)
                res["two_lap_s"] = round(sum(res["lap_times"]), 3) if res["lap_times"] else None
                eps.append(res)
            comp = [e for e in eps if is_completed(e["cause"])]
            crate = len(comp) / len(eps)
            two = [e["two_lap_s"] for e in comp if e["two_lap_s"]]
            best2 = min(two) if two else None
            # 대표 cause(최빈)
            causes = {}
            for e in eps:
                causes[str(e["cause"])] = causes.get(str(e["cause"]), 0) + 1
            cause0 = max(causes, key=causes.get)
            len0 = int(np.median([e["length"] for e in eps]))
            rows.append({"step": _step_of(f), "file": os.path.basename(f),
                         "completion": crate, "best_2lap": best2,
                         "cause": cause0, "len": len0})
            bs = f"{best2:.2f}s" if best2 else "-"
            print(f"  step={_step_of(f):>7}  완주율={crate:.2f}  best2lap={bs:>9}  "
                  f"cause={cause0:<10} len={len0}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    # ── 요약 ──
    completers = [r for r in rows if r["completion"] > 0 and r["best_2lap"]]
    print("\n========== 스윕 요약 ==========", flush=True)
    if completers:
        best = min(completers, key=lambda r: r["best_2lap"])
        print(f"완주한 체크포인트: {len(completers)}/{len(rows)}", flush=True)
        print(f"★ best: step={best['step']}  2랩={best['best_2lap']:.2f}s  완주율={best['completion']:.2f}", flush=True)
        print(f"  vs baseline {BASELINE_2LAP}s [G2]: "
              f"{'PASS ✓' if best['best_2lap'] < BASELINE_2LAP else 'FAIL'} ({BASELINE_2LAP-best['best_2lap']:+.2f}s)", flush=True)
        print(f"  vs cap10 {CAP10_RECORD}s [G3]: "
              f"{'PASS ✓✓ 기록경신!' if best['best_2lap'] < CAP10_RECORD else 'FAIL'} ({CAP10_RECORD-best['best_2lap']:+.2f}s)", flush=True)
    else:
        print(f"완주한 체크포인트 0/{len(rows)} — G1 미달.", flush=True)
        # 가장 멀리 간 체크포인트(진행 길이) 참고
        far = max(rows, key=lambda r: r["len"])
        print(f"가장 오래 버틴 체크포인트: step={far['step']} len={far['len']} cause={far['cause']}", flush=True)
    print("===============================", flush=True)

    out = os.path.join(run_dir, f"sweep_{args.task}.json")
    with open(out, "w") as fp:
        json.dump({"run_dir": run_dir, "task": args.task, "v_max": v_max,
                   "episodes": args.episodes, "rows": rows}, fp, indent=2, ensure_ascii=False)
    print(f"[sweep] JSON 저장: {out}", flush=True)


if __name__ == "__main__":
    main()
