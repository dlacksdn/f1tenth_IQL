"""
f1tenth_data.py — crash_data npz → CORL IQL transition dict (5 keys).

VERIFIED data convention (DreamerV3 replay, 2026-06-22 직접 측정):
  action[t] / reward[t] / log_reward_*[t] 는 obs[t]를 **만든** 행동(이전 행동)·obs[t] **도착** 보상이다.
  action[0]은 reset 더미(=0), reward[0]=0. 근거: corr(action[t].speed, state[t][4])=0.99 ≫ 0.48.
  => 표준 RL transition (s_t, a_t, r_t, s_{t+1}, done_t):
       s_t      = obs[t]
       a_t      = action[t+1]        # s_t 에서 취한 행동
       r_t      = reward[t+1]        # s_{t+1} 도착 보상
       s_{t+1}  = obs[t+1]
       done_t   = is_terminal[t+1]   # 다음 상태가 terminal (충돌=True, 완주=False)
     for t in [0, T-2]  → 배열로는 obs[:-1], action[1:], reward[1:], obs[1:], is_terminal[1:].

obs = concat(min-pool(lidar,128), state5) = 133D. action 은 [-1,1] 그대로(NormalizeActions 입력).
terminals: 충돌=True(흡수), 완주=False(value 가 +100 lap 보상을 뒤로 bootstrap).
"""
import os
import re
import glob
import numpy as np

DATA_ROOT = os.environ.get("F1TENTH_DATA_ROOT", "/home/dlacksdn/f1tenth_RL_project/runs/crash_data")
LIDAR_DOWNSAMPLE = 128
STATE_DIM = 5
OBS_DIM = LIDAR_DOWNSAMPLE + STATE_DIM   # 133
ACT_DIM = 2

# NormalizeActions 상수.
S_MIN, S_MAX = -0.4189, 0.4189     # steer 물리범위 — 모든 cap 정책 공통(불변)
V_MIN = -5.0                       # 후진 속도 하한 — 공통(불변)
COMMON_V_MAX = 20.0                # ★ action 조화(harmonize) 공통 frame. eval env v_max 와 동일해야 함.

# ──────────────────────────────────────────────────────────────────────────────
# ★ action 조화(harmonization) — 정합성 핵심
#   저장된 action 은 정규화 [-1,1]. 수집 시 NormalizeActions 가 [-1,1]→[V_MIN, v_max_collect]
#   로 매핑하므로 같은 정규화 speed 값이 데이터셋마다 다른 물리속도를 의미한다
#   (실측: action[1]=+1 → cap10=10 m/s, cap15=15, cap20=20 m/s, 분산 0).
#   섞으면 (state,action)→(next,reward) 가 모순 → Bellman 타깃 ill-posed.
#   해법: speed action 을 단일 공통 frame(COMMON_V_MAX)으로 재정규화.
#     a1_common = (a1+1)·(v_collect+5)/(v_common+5) − 1     (steer 는 불변: S_MIN/S_MAX 공통)
#   eval 은 env v_max = COMMON_V_MAX 로 굴려 정책의 공통-frame action 을 올바른 물리속도로 환원.
def dataset_vmax(ds_name):
    """폴더명 'capN...' → 수집 forward-speed 상한 N (실측 검증: cap10/15/20=10/15/20)."""
    m = re.match(r"cap_?(\d+)", os.path.basename(ds_name.rstrip("/")))
    if not m:
        raise ValueError(f"수집 v_max 를 폴더명에서 못 읽음(capN 형식 필요): {ds_name!r}")
    return float(m.group(1))


def harmonize_speed(a1, v_collect, v_common=COMMON_V_MAX):
    """정규화 speed action[1] 을 v_collect frame → v_common frame 으로 재정규화.
    유도: phys=(a1+1)/2·(v_collect−V_MIN)+V_MIN ; a1'=2·(phys−V_MIN)/(v_common−V_MIN)−1
         ⇒ a1' = (a1+1)·(v_collect−V_MIN)/(v_common−V_MIN) − 1."""
    return (a1 + 1.0) * (v_collect - V_MIN) / (v_common - V_MIN) - 1.0


def downsample_lidar(lidar, n=LIDAR_DOWNSAMPLE):
    """(T,1080)->(T,n) 섹터별 min-pool(최근접 장애물 보존=레이싱 안전). 옛 로더와 동일 식.
    train/eval 이 **반드시 동일 함수**를 써야 함(정합)."""
    lidar = np.asarray(lidar, dtype=np.float32)
    if n is None or n >= lidar.shape[-1]:
        return lidar
    B = lidar.shape[-1]
    edges = np.linspace(0, B, n + 1).round().astype(int)
    out = np.empty((lidar.shape[0], n), dtype=np.float32)
    for j in range(n):
        out[:, j] = lidar[:, edges[j]:edges[j + 1]].min(axis=1)
    return out


def build_obs(z, n=LIDAR_DOWNSAMPLE):
    """npz(dict-like) -> (T, n+5) obs = [min-pool lidar, state5]."""
    lidar = downsample_lidar(z["lidar"], n)
    state = np.asarray(z["state"], dtype=np.float32)
    return np.concatenate([lidar, state], axis=1)


def _episode_reward(z, collision_penalty=-10.0, lap_bonus=None):
    """성분 합으로 per-step 보상 재구성(+ 충돌/lap 재스케일 옵션).
    collision_penalty != -10 이면 충돌 step(-10)을 그 값으로 치환. lap_bonus 지정 시 +100 을 그 값으로."""
    prog = np.asarray(z["log_reward_progress"], dtype=np.float32)
    coll = np.asarray(z["log_reward_collision"], dtype=np.float32)   # -10 at crash else 0
    lap = np.asarray(z["log_reward_lap"], dtype=np.float32)          # +100 at lap else 0
    rev = np.asarray(z["log_reward_reverse"], dtype=np.float32)
    div = np.asarray(z["log_reward_diverged"], dtype=np.float32)
    if collision_penalty != -10.0:
        coll = np.where(coll < 0, np.float32(collision_penalty), np.float32(0.0))
    if lap_bonus is not None:
        lap = np.where(lap > 0, np.float32(lap_bonus), np.float32(0.0))
    return (prog + coll + lap + rev + div).astype(np.float32)


def load_f1tenth_dataset(datasets, data_root=DATA_ROOT, lidar_n=LIDAR_DOWNSAMPLE,
                         collision_penalty=-10.0, lap_bonus=None,
                         common_v_max=COMMON_V_MAX, verbose=True):
    """하나 이상의 crash_data 하위폴더 → CORL 5-key transition dict.

    datasets: 폴더명 시퀀스, 예 ('cap10_full',) 또는 ('cap10_full','cap15','cap20').
    common_v_max: speed action 을 이 공통 frame 으로 조화(None=조화 끄기, 원본 frame 유지).
                  eval env v_max 와 반드시 일치시켜야 함(기본 20).
    반환: {observations, actions, rewards, next_observations, terminals}.
    """
    if isinstance(datasets, str):
        datasets = (datasets,)
    O, A, R, NO, D = [], [], [], [], []
    n_ep = n_comp = n_crash = 0
    vmax_used = {}
    for ds in datasets:
        v_collect = dataset_vmax(ds) if common_v_max is not None else None
        if common_v_max is not None:
            vmax_used[ds] = v_collect
        files = sorted(glob.glob(os.path.join(data_root, ds, "*.npz")))
        if not files and verbose:
            print(f"[warn] npz 없음: {ds}")
        for f in files:
            z = np.load(f)
            T = int(z["action"].shape[0])
            if T < 2:
                continue
            # 한 파일 = 한 에피소드 가정(검증됨: is_first 가 0에만)
            isfirst = np.asarray(z["is_first"], dtype=bool)
            if isfirst[0] != True or isfirst[1:].sum() != 0:
                raise ValueError(f"다중 에피소드 파일(미지원): {f} is_first idx={np.where(isfirst)[0].tolist()}")
            obs = build_obs(z, lidar_n)                         # (T, OBS)
            rew = _episode_reward(z, collision_penalty, lap_bonus)  # (T,)
            act = np.asarray(z["action"], dtype=np.float32).copy()  # (T,2) action[t]->obs[t]
            # ★ speed action 조화: v_collect → common_v_max (steer[:,0] 불변)
            if common_v_max is not None and v_collect != common_v_max:
                act[:, 1] = harmonize_speed(act[:, 1], v_collect, common_v_max)
            isterm = np.asarray(z["is_terminal"], dtype=bool)   # (T,)
            # 표준 transition (위 규약): a_t=action[t+1], r_t=reward[t+1], done=is_terminal[t+1]
            O.append(obs[:-1])
            A.append(act[1:])
            R.append(rew[1:])
            NO.append(obs[1:])
            D.append(isterm[1:])
            n_ep += 1
            if isterm.any():
                n_crash += 1
            else:
                n_comp += 1
    if not O:
        raise ValueError(f"데이터 없음: {datasets} @ {data_root}")
    dataset = {
        "observations": np.concatenate(O).astype(np.float32),
        "actions": np.concatenate(A).astype(np.float32),
        "rewards": np.concatenate(R).astype(np.float32),
        "next_observations": np.concatenate(NO).astype(np.float32),
        "terminals": np.concatenate(D).astype(bool),
    }
    for k in ("observations", "actions", "rewards", "next_observations"):
        if not np.isfinite(dataset[k]).all():
            raise ValueError(f"비유한 값(NaN/Inf) in {k}")
    if verbose:
        d = dataset
        harm = (f"harmonize→v_common={common_v_max} (수집 v_max={vmax_used})"
                if common_v_max is not None else "harmonize=OFF(원본 frame)")
        print(f"[f1tenth] datasets={list(datasets)}  ep={n_ep}(완주 {n_comp}/충돌 {n_crash})  "
              f"transitions={d['observations'].shape[0]}  obs_dim={d['observations'].shape[1]}  "
              f"act_dim={d['actions'].shape[1]}  terminals={int(d['terminals'].sum())}")
        print(f"           {harm}")
        print(f"           steer[min,max]=[{d['actions'][:,0].min():.3f},{d['actions'][:,0].max():.3f}]  "
              f"speed[min,max]=[{d['actions'][:,1].min():.3f},{d['actions'][:,1].max():.3f}]  "
              f"reward[min,mean,max]=[{d['rewards'].min():.2f},{d['rewards'].mean():.3f},{d['rewards'].max():.2f}]")
    return dataset


if __name__ == "__main__":
    # 스모크: D2(cap10_full) 와 D3(cap10_full+cap15+cap20) 구성 확인
    print("=== D2: cap10_full ===")
    load_f1tenth_dataset(("cap10_full",))
    print("=== D3: cap10_full + cap15 + cap20 ===")
    load_f1tenth_dataset(("cap10_full", "cap15", "cap20"))
