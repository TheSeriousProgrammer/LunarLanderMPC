import argparse
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from train import LunarModelLightning, minimal_obs_encode


CHECKPOINT = "./checkpoints_hard_negative_angv/checkpoint_epoch=0_vl=5.7813.ckpt"

MAX_STEPS = 2000
SEED = 42
HORIZON = 4
POPULATION = 256
ELITES = 32
CEM_ITERS = 8
INITIAL_STD = 0.8
MIN_STD = 0.05
MODEL_BREAK_THRESHOLD = 0.01


DIGITS = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "001", "001", "001"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
}


def draw_digit(frame, digit, x, y, scale=4):
    color = np.array([255, 255, 255], dtype=np.uint8)
    for row, bits in enumerate(DIGITS[digit]):
        for col, bit in enumerate(bits):
            if bit == "1":
                y0 = y + row * scale
                x0 = x + col * scale
                frame[y0 : y0 + scale, x0 : x0 + scale] = color


def draw_frame_counter(frame, count):
    text = str(count).zfill(4)
    x = 8
    y = 8
    scale = 4
    width = len(text) * 4 * scale + 8
    height = 5 * scale + 8
    frame[y - 4 : y - 4 + height, x - 4 : x - 4 + width] = 0
    for digit in text:
        draw_digit(frame, digit, x, y, scale)
        x += 4 * scale
    return frame


def draw_model_break_flag(frame):
    size = 28
    margin = 12
    y0 = frame.shape[0] - size - margin
    x0 = frame.shape[1] - size - margin
    frame[y0 : y0 + size, x0 : x0 + size] = np.array([255, 0, 0], dtype=np.uint8)
    return frame


class FrameCounterWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.frame_count = 0
        self.predicted_next_latent = None
        self.model_break = False
        self.model_break_count = 0
        self.max_model_l2_error = 0.0

    def reset(self, **kwargs):
        self.frame_count = 0
        self.predicted_next_latent = None
        self.model_break = False
        self.model_break_count = 0
        self.max_model_l2_error = 0.0
        return self.env.reset(**kwargs)

    def set_predicted_next_latent(self, latent):
        self.predicted_next_latent = latent.detach().cpu().numpy()[0]

    def step(self, action):
        self.frame_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.predicted_next_latent is not None:
            observed_latent = minimal_obs_encode(np.array(obs))[:-2]
            error = np.linalg.norm(self.predicted_next_latent - observed_latent)
            self.model_break = error > MODEL_BREAK_THRESHOLD
            self.model_break_count += int(self.model_break)
            self.max_model_l2_error = max(self.max_model_l2_error, float(error))
            info = dict(info)
            info["model_l2_error"] = float(error)
            info["model_break"] = self.model_break
        return obs, reward, terminated, truncated, info

    def render(self):
        frame = self.env.render()
        if frame is None:
            return None
        frame = draw_frame_counter(frame.copy(), self.frame_count)
        if self.model_break:
            frame = draw_model_break_flag(frame)
        return frame


def encode_obs(obs, device):
    return (
        torch.from_numpy(minimal_obs_encode(np.array(obs))[:-2])
        .view(1, 6)
        .to(device=device, dtype=torch.float32)
    )


def load_model(checkpoint_path, device):
    print("checkpoint", checkpoint_path)
    model = LunarModelLightning.load_from_checkpoint(Path(checkpoint_path)).model
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@torch.no_grad()
def rollout(model, state, actions):
    state = state.expand(actions.shape[0], -1)
    states = []
    for t in range(actions.shape[1]):
        state = model(state, actions[:, t])
        states.append(state)
    return torch.stack(states, dim=1)


def cost_fn(states, actions):
    target = torch.tensor([0.0, 0.01, 0.0, -0.01, 0.0, 0.0], device=states.device)
    weights = torch.tensor([1.75, 1.0, 1.0, 15.0, 20.0, 20.0], device=states.device)

    scaling = torch.linspace(
        0.8, 2.0, steps=states.shape[1], device=states.device
    ).unsqueeze(0)

    diff = states - target.view(1, 1, 6)
    diff[:, -1] *= 2
    rollout_cost = ((diff.square() * weights.view(1, 1, 6)).sum(dim=2) * scaling).mean(
        dim=1
    )
    action_cost = 0.001 * actions.square().mean(dim=(1, 2))
    return rollout_cost + action_cost


@torch.no_grad()
def cem(model, state, previous_plan, device):
    if previous_plan is None:
        mean = torch.zeros(HORIZON, 2, device=device)
    else:
        mean = previous_plan.clone()
    std = torch.full_like(mean, INITIAL_STD)

    best = mean
    best_cost = float("inf")

    for _ in range(CEM_ITERS):
        actions = mean + std * torch.randn(POPULATION, HORIZON, 2, device=device)
        actions = actions.clamp(-1.0, 1.0)
        if previous_plan is not None:
            actions[0] = previous_plan

        states = rollout(model, state, actions)
        costs = cost_fn(states, actions)
        elite_idx = torch.topk(costs, ELITES, largest=False).indices
        elites = actions[elite_idx]

        mean = elites.mean(dim=0)
        std = elites.std(dim=0, unbiased=False).clamp_min(MIN_STD)

        if costs[elite_idx[0]].item() < best_cost:
            best_cost = costs[elite_idx[0]].item()
            best = actions[elite_idx[0]].clone()

    return best, best_cost


import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default=CHECKPOINT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = load_model(args.checkpoint_path, device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = f"videos/minimal_cem"

    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        gravity=-10.0,
        enable_wind=False,
        wind_power=0.0,
        turbulence_power=0.0,
        render_mode="rgb_array",
        max_episode_steps=-1,
    )
    frame_env = FrameCounterWrapper(env)
    env = frame_env
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix=f"minimal_cem_{timestamp}",
    )

    obs, _ = env.reset(seed=SEED)
    total_reward = 0.0
    previous_plan = None
    terminated = False
    truncated = False
    thrusters_off = False
    try:
        for step in range(MAX_STEPS):
            state = encode_obs(obs, device)
            if not ((obs[1] < 1e-2 and math.fabs(obs[3]) < 1e-2) or thrusters_off):
                plan, plan_cost = cem(model, state, previous_plan, device)
                action = plan[0].detach().cpu().numpy()
            else:
                print("Thrusters off")
                thrusters_off = True
                action = [0.0, 0.0]  # no thrusters
            previous_plan = torch.cat([plan[1:], plan[-1:]], dim=0).detach()

            with torch.no_grad():
                action_tensor = torch.tensor(
                    action, device=device, dtype=torch.float32
                ).view(1, 2)
                frame_env.set_predicted_next_latent(model(state, action_tensor))

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            print(
                step,
                "action",
                np.round(action, 3),
                "cost",
                round(plan_cost, 3),
                "reward",
                round(float(reward), 3),
                "model_l2",
                round(info.get("model_l2_error", 0.0), 5),
                "model_break",
                info.get("model_break", False),
                "obs",
                np.round(obs[:6], 3),
            )

            if terminated or truncated:
                break
    finally:
        env.close()

    print("steps", step + 1)
    print("total_reward", total_reward)
    print("terminated", terminated)
    print("truncated", truncated)
    print("final_obs", obs[:6])
    print("model_break_count", frame_env.model_break_count)
    print("max_model_l2_error", frame_env.max_model_l2_error)
    print("video_dir", video_dir)


if __name__ == "__main__":
    main()
