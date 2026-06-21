"""
train_iql.py — vendored CORL IQL 클래스를 재사용한 F1TENTH offline IQL 학습기.

설계(plan/001 §4 교체점 5개):
  ① 데이터: gym.make+d4rl.qlearning_dataset 대신 f1tenth_data.load_f1tenth_dataset (npz, obs133/act2).
  ② import: vendored iql.py 최상단의 `import d4rl, gym, wandb` 를 stub(아래) — 클래스만 재사용.
  ③ eval: 인라인 eval/env 제거 — 평가는 별도 프로세스(eval_iql.py, RL_project venv, f110_gym).
  ④ ckpt: eval과 분리한 '촘촘' 체크포인트(번호본 보존 + latest.pt 롤링) + obs_norm.npz + config.json.
  ⑤ 로깅: wandb 대신 JSONL(train_log.jsonl) + TensorBoard.

규칙 준수:
  - GPU 실행은 반드시 run_in_background (foreground+CUDA=exit144). CPU 스모크만 foreground 허용.
  - 촘촘 ckpt: 중간에 끊겨도 날아가지 않게. --resume 로 latest.pt 이어서 학습.
  - γ(discount) 기본 0.999: +100 lap 보상이 value 까지 bootstrap 되도록(0.99는 lap 보상에 사실상 blind).
"""
import argparse
import json
import os
import sys
import time
import types
from unittest.mock import MagicMock

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor", "CORL")
for _p in (HERE, VENDOR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 교체점 ②: vendored iql.py 가 import 하는 d4rl/gym/wandb 를 stub ───────────────
#   gym.Env 는 함수 시그니처 주석(typing) 평가에 쓰이므로 '진짜 타입'(object)으로 둔다.
#   gym.wrappers / gym.make / d4rl / wandb 는 우리가 호출하지 않으므로 MagicMock 으로 충분.
_gym = types.ModuleType("gym")
_gym.Env = object
_gym.wrappers = MagicMock()
_gym.make = MagicMock()
sys.modules.setdefault("gym", _gym)
sys.modules.setdefault("d4rl", MagicMock())
sys.modules.setdefault("wandb", MagicMock())

import iql as corl  # noqa: E402  (vendored CORL — 클래스 재사용)
from f1tenth_data import load_f1tenth_dataset, OBS_DIM, ACT_DIM  # noqa: E402


def build_trainer(args, state_dim, action_dim, max_action):
    """CORL 클래스로 IQL trainer 조립(상류 train() 의 599~621행과 동일 구성)."""
    dev = args.device
    q_network = corl.TwinQ(state_dim, action_dim, hidden_dim=args.hidden_dim,
                           n_hidden=args.n_hidden).to(dev)
    v_network = corl.ValueFunction(state_dim, hidden_dim=args.hidden_dim,
                                   n_hidden=args.n_hidden).to(dev)
    if args.iql_deterministic:
        actor = corl.DeterministicPolicy(state_dim, action_dim, max_action,
                                         hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
                                         dropout=args.actor_dropout).to(dev)
    else:
        actor = corl.GaussianPolicy(state_dim, action_dim, max_action,
                                    hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
                                    dropout=args.actor_dropout).to(dev)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=args.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=args.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    trainer = corl.ImplicitQLearning(
        max_action=max_action,
        actor=actor, actor_optimizer=actor_optimizer,
        q_network=q_network, q_optimizer=q_optimizer,
        v_network=v_network, v_optimizer=v_optimizer,
        iql_tau=args.iql_tau, beta=args.beta,
        max_steps=args.max_timesteps,
        discount=args.discount, tau=args.tau, device=dev,
    )
    return trainer


def main():
    p = argparse.ArgumentParser(description="F1TENTH offline IQL (CORL classes)")
    # ── 데이터 / 보상 셰이핑 ──
    p.add_argument("--datasets", nargs="+", default=["cap10_full"],
                   help="crash_data 하위폴더들. 예: cap10_full / cap10_full cap15 cap20")
    p.add_argument("--collision_penalty", type=float, default=-10.0,
                   help="충돌 step 보상(-10 기본; 더 강한 회피엔 -50 등)")
    p.add_argument("--lap_bonus", type=float, default=None,
                   help="lap 완주 보상 재스케일(미지정=+100 원본 유지)")
    p.add_argument("--lidar_n", type=int, default=128)
    p.add_argument("--common_v_max", type=float, default=20.0,
                   help="★ speed action 조화 공통 frame. eval env v_max 와 반드시 일치(기본 20)")
    # ── IQL 하이퍼파라미터 ──
    p.add_argument("--discount", type=float, default=0.999,
                   help="γ. lap(+100) bootstrap 위해 0.999 권장(0.99는 lap 보상에 blind)")
    p.add_argument("--iql_tau", type=float, default=0.7, help="expectile τ")
    p.add_argument("--beta", type=float, default=3.0,
                   help="AWR inverse temperature(작으면 BC쪽, 크면 Q최대화쪽)")
    p.add_argument("--iql_deterministic", action="store_true",
                   help="결정론적 actor(기본=Gaussian)")
    p.add_argument("--actor_dropout", type=float, default=None)
    p.add_argument("--tau", type=float, default=0.005, help="target Q Polyak")
    p.add_argument("--vf_lr", type=float, default=3e-4)
    p.add_argument("--qf_lr", type=float, default=3e-4)
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_hidden", type=int, default=2)
    p.add_argument("--normalize", type=int, default=1, help="1=state z-score 정규화")
    # ── 학습 루프 ──
    p.add_argument("--max_timesteps", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", required=True, help="ckpt/로그/정규화 통계 저장 폴더")
    p.add_argument("--ckpt_freq", type=int, default=5000, help="번호 ckpt 저장 주기(촘촘하게)")
    p.add_argument("--latest_freq", type=int, default=1000, help="latest.pt 롤링 저장 주기")
    p.add_argument("--log_freq", type=int, default=1000, help="JSONL/TB 로깅 주기")
    p.add_argument("--resume", action="store_true", help="out_dir/latest.pt 에서 이어서 학습")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[fatal] --device cuda 인데 CUDA 사용 불가. CPU 스모크는 --device cpu.", flush=True)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    corl.set_seed(args.seed)

    # ── 교체점 ①: npz → CORL 5-key dict ──
    dataset = load_f1tenth_dataset(
        tuple(args.datasets), lidar_n=args.lidar_n,
        collision_penalty=args.collision_penalty, lap_bonus=args.lap_bonus,
        common_v_max=args.common_v_max,
    )
    state_dim = dataset["observations"].shape[1]
    action_dim = dataset["actions"].shape[1]
    assert state_dim == OBS_DIM and action_dim == ACT_DIM, (state_dim, action_dim)
    n_transitions = dataset["observations"].shape[0]
    max_action = 1.0  # action 은 [-1,1] (NormalizeActions 입력 규약)

    # ── state 정규화: 통계 저장(eval 이 동일 mean/std 로 obs 정규화해야 함) ──
    norm_path = os.path.join(args.out_dir, "obs_norm.npz")
    if args.normalize:
        if args.resume and os.path.exists(norm_path):
            z = np.load(norm_path)
            state_mean, state_std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
            print(f"[norm] resume: {norm_path} 에서 mean/std 로드", flush=True)
        else:
            state_mean, state_std = corl.compute_mean_std(dataset["observations"], eps=1e-3)
            state_mean = state_mean.astype(np.float32)
            state_std = state_std.astype(np.float32)
            np.savez(norm_path, mean=state_mean, std=state_std)
            print(f"[norm] mean/std 계산·저장 → {norm_path}", flush=True)
        dataset["observations"] = corl.normalize_states(dataset["observations"], state_mean, state_std)
        dataset["next_observations"] = corl.normalize_states(dataset["next_observations"], state_mean, state_std)
    else:
        # eval 정합 위해 항등 통계도 기록
        np.savez(norm_path, mean=np.zeros(state_dim, np.float32), std=np.ones(state_dim, np.float32))

    # ── ReplayBuffer (정확히 N 크기 → VRAM 최소) ──
    replay_buffer = corl.ReplayBuffer(state_dim, action_dim, n_transitions, args.device)
    replay_buffer.load_d4rl_dataset(dataset)

    trainer = build_trainer(args, state_dim, action_dim, max_action)

    # ── config 기록(재현·인수인계용) ──
    cfg = vars(args).copy()
    cfg.update(dict(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                    n_transitions=int(n_transitions),
                    actor="deterministic" if args.iql_deterministic else "gaussian"))
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ── resume ──
    start_it = 0
    latest_path = os.path.join(args.out_dir, "latest.pt")
    if args.resume and os.path.exists(latest_path):
        sd = torch.load(latest_path, map_location=args.device)
        trainer.load_state_dict(sd)
        start_it = trainer.total_it
        print(f"[resume] {latest_path} (total_it={start_it}) 에서 이어서 학습", flush=True)

    # ── 교체점 ⑤: JSONL + TensorBoard ──
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log_f = open(log_path, "a")
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.out_dir, "tb"))
    except Exception as e:  # TB 실패해도 학습은 계속(JSONL만)
        print(f"[warn] TensorBoard 비활성: {e}", flush=True)

    def save_ckpt(tag):
        path = os.path.join(args.out_dir, f"checkpoint_{tag}.pt")
        torch.save(trainer.state_dict(), path)
        return path

    print(f"--- IQL 학습 시작: datasets={args.datasets} N={n_transitions} "
          f"obs={state_dim} act={action_dim} γ={args.discount} β={args.beta} τ={args.iql_tau} "
          f"actor={'det' if args.iql_deterministic else 'gauss'} dev={args.device} "
          f"steps[{start_it}→{args.max_timesteps}] ---", flush=True)

    t0 = time.time()
    last_t = t0
    for it in range(start_it, args.max_timesteps):
        batch = replay_buffer.sample(args.batch_size)  # 이미 device 텐서
        log_dict = trainer.train(batch)
        step = trainer.total_it  # = it+1

        # NaN 가드(조용히 망가지지 않게 즉시 중단 + latest 보존)
        if any((v != v) for v in log_dict.values()):  # NaN
            save_ckpt(f"NANABORT_{step}")
            torch.save(trainer.state_dict(), latest_path)
            print(f"[fatal] NaN loss @step {step}: {log_dict}. latest 저장 후 중단.", flush=True)
            sys.exit(2)

        if step % args.log_freq == 0 or step == args.max_timesteps:
            now = time.time()
            its = args.log_freq / max(now - last_t, 1e-9)
            last_t = now
            rec = {"step": step, "elapsed_s": round(now - t0, 1), "it_per_s": round(its, 1)}
            rec.update({k: round(float(v), 6) for k, v in log_dict.items()})
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            if writer:
                for k, v in log_dict.items():
                    writer.add_scalar(f"loss/{k}", float(v), step)
                writer.add_scalar("perf/it_per_s", its, step)
            print(f"[step {step:>8}] " + "  ".join(f"{k}={v:.4f}" for k, v in log_dict.items())
                  + f"  ({its:.0f} it/s)", flush=True)

        # latest.pt 롤링(중단 대비)
        if step % args.latest_freq == 0 or step == args.max_timesteps:
            torch.save(trainer.state_dict(), latest_path)

        # 번호 ckpt(촘촘 보존 — 폐기 금지)
        if step % args.ckpt_freq == 0 or step == args.max_timesteps:
            save_ckpt(str(step))

    torch.save(trainer.state_dict(), latest_path)
    log_f.close()
    if writer:
        writer.close()
    print(f"--- 학습 완료: {args.max_timesteps} steps, {round(time.time()-t0,1)}s, out={args.out_dir} ---",
          flush=True)


if __name__ == "__main__":
    main()
