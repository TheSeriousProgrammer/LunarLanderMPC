import argparse
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from train import LunarModelLightning, minimal_obs_encode


OBS_DECODE_SCALE = np.array([5.0, 5.0, 20.0, 20.0, 4 * np.pi, 20.0])


def checkpoint_score(path: Path):
    match = re.search(r"tl_epoch=([0-9]+(?:\.[0-9]+)?)", path.name)
    tl_epoch = float(match.group(1)) if match else float("inf")
    return (tl_epoch, -path.stat().st_mtime)


def load_best_compatible_model(
    checkpoint_dir: str, device: torch.device, checkpoint_path: str | None = None
):
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        print(f"Loading checkpoint: {path}")
        lightning_model = LunarModelLightning.load_from_checkpoint(path)
        model = lightning_model.model.to(device).eval()
        for param in model.parameters():
            param.requires_grad = False
        return model, path

    checkpoint_paths = sorted(Path(checkpoint_dir).glob("*.ckpt"), key=checkpoint_score)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    for path in checkpoint_paths:
        try:
            print(f"Loading checkpoint: {path}")
            lightning_model = LunarModelLightning.load_from_checkpoint(path)
            model = lightning_model.model.to(device).eval()
            for param in model.parameters():
                param.requires_grad = False
            return model, path
        except RuntimeError as exc:
            print(f"Skipping incompatible checkpoint: {path} ({exc.__class__.__name__})")

    raise RuntimeError(f"No compatible checkpoint found in {checkpoint_dir}")


def encode_observation(observation: np.ndarray, device: torch.device):
    return (
        torch.from_numpy(minimal_obs_encode(np.array(observation))[:-2])
        .expand((1, 6))
        .to(device=device, dtype=torch.float32)
    )


def decode_latent(latent: torch.Tensor):
    return latent.detach().cpu().numpy()[0] * OBS_DECODE_SCALE


def latent_deviation_metrics(pred_latent: torch.Tensor, observed_latent: torch.Tensor):
    diff = observed_latent - pred_latent
    return {
        "latent_l2": torch.linalg.vector_norm(diff, dim=1).mean().item(),
        "pos": torch.linalg.vector_norm(diff[:, 0:2], dim=1).mean().item(),
        "vel": torch.linalg.vector_norm(diff[:, 2:4], dim=1).mean().item(),
        "vx": diff[:, 2].abs().mean().item(),
        "vy": diff[:, 3].abs().mean().item(),
        "ang": diff[:, 4].abs().mean().item(),
        "angv": diff[:, 5].abs().mean().item(),
    }


@torch.no_grad()
def rollout_model(model, current_latent: torch.Tensor, actions: torch.Tensor):
    state = current_latent.expand(actions.shape[0], -1)
    for step in range(actions.shape[1]):
        state = model(state, actions[:, step, :])
    return state


@torch.no_grad()
def cem_plan(
    model,
    current_latent: torch.Tensor,
    desired_latent: torch.Tensor,
    loss_weights: torch.Tensor,
    horizon: int,
    population_size: int,
    elite_count: int,
    iterations: int,
    min_std: float,
):
    device = current_latent.device
    mean = torch.zeros(horizon, 2, dtype=current_latent.dtype, device=device)
    std = torch.full_like(mean, 0.6)

    best_actions = None
    best_terminal = None
    best_deviation = float("inf")

    for _ in range(iterations):
        sampled_actions = mean + std * torch.randn(
            population_size, horizon, 2, dtype=current_latent.dtype, device=device
        )
        sampled_actions = sampled_actions.clamp(min=-1.0, max=1.0)

        terminal_latents = rollout_model(model, current_latent, sampled_actions)
        deviations = ((terminal_latents - desired_latent).square() * loss_weights).sum(
            dim=1
        )
        elite_indices = torch.topk(deviations, k=elite_count, largest=False).indices
        elites = sampled_actions[elite_indices]

        mean = elites.mean(dim=0)
        std = elites.std(dim=0, unbiased=False).clamp_min(min_std)

        iteration_best_idx = elite_indices[0]
        iteration_best_deviation = deviations[iteration_best_idx].item()
        if iteration_best_deviation < best_deviation:
            best_deviation = iteration_best_deviation
            best_actions = sampled_actions[iteration_best_idx].clone()
            best_terminal = terminal_latents[iteration_best_idx : iteration_best_idx + 1].clone()

    return best_actions, best_terminal, best_deviation


def make_env(args):
    render_mode = None
    if args.record_video:
        render_mode = "rgb_array"
    elif args.render:
        render_mode = "human"

    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        gravity=args.gravity,
        enable_wind=False,
        wind_power=0.0,
        turbulence_power=0.0,
        render_mode=render_mode,
    )

    if args.record_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.video_dir,
            episode_trigger=lambda episode_id: episode_id == 0,
            name_prefix="world_model_mpc",
        )

    return env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="checkpoints_bro")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-size", type=int, default=256)
    parser.add_argument("--elite-count", type=int, default=32)
    parser.add_argument("--cem-iters", type=int, default=12)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument("--vy-weight", type=float, default=20.0)
    parser.add_argument("--angle-weight", type=float, default=10.0)
    parser.add_argument("--target-vy", type=float, default=-0.01)
    parser.add_argument("--gravity", type=float, default=-10.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", default="videos/mpc")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.elite_count > args.population_size:
        raise ValueError("--elite-count must be <= --population-size")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, checkpoint_path = load_best_compatible_model(
        args.checkpoint_dir, device, args.checkpoint_path
    )
    desired_latent = torch.zeros((1, 6), dtype=torch.float32, device=device)
    desired_latent[:, 3] = args.target_vy / OBS_DECODE_SCALE[3]
    loss_weights = torch.ones((1, 6), dtype=torch.float32, device=device)
    loss_weights[:, 3] = args.vy_weight
    loss_weights[:, 4] = args.angle_weight

    env = make_env(args)
    observation, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    final_step = 0
    terminated = False
    truncated = False
    model_error_sums = {
        "latent_l2": 0.0,
        "pos": 0.0,
        "vel": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "ang": 0.0,
        "angv": 0.0,
    }
    model_error_square_sums = dict(model_error_sums)
    model_error_max = dict(model_error_sums)

    try:
        for step in range(args.max_steps):
            current_latent = encode_observation(observation, device)
            actions, pred_terminal, pred_deviation = cem_plan(
                model=model,
                current_latent=current_latent,
                desired_latent=desired_latent,
                loss_weights=loss_weights,
                horizon=args.horizon,
                population_size=args.population_size,
                elite_count=args.elite_count,
                iterations=args.cem_iters,
                min_std=args.min_std,
            )

            action = actions[0].detach().cpu().numpy()
            with torch.no_grad():
                pred_next_latent = model(current_latent, actions[0:1])
            observation, reward, terminated, truncated, _ = env.step(action)
            observed_next_latent = encode_observation(observation, device)
            step_model_error = latent_deviation_metrics(
                pred_next_latent, observed_next_latent
            )
            for key, value in step_model_error.items():
                model_error_sums[key] += value
                model_error_square_sums[key] += value * value
                model_error_max[key] = max(model_error_max[key], value)

            total_reward += float(reward)
            final_step = step + 1

            if not args.quiet:
                print(
                    "step",
                    step,
                    "action",
                    np.round(action, 4),
                    "reward",
                    round(float(reward), 4),
                    "total_reward",
                    round(total_reward, 4),
                    "pred_dev",
                    round(pred_deviation, 6),
                    "obs",
                    np.round(observation[:6], 4),
                    "model_err_l2",
                    round(step_model_error["latent_l2"], 6),
                    "model_err_vel",
                    round(step_model_error["vel"], 6),
                    "model_err_vy",
                    round(step_model_error["vy"], 6),
                    "pred_terminal",
                    np.round(decode_latent(pred_terminal), 4),
                )

            if terminated or truncated:
                break
    finally:
        env.close()

    print("checkpoint", checkpoint_path)
    print("steps", final_step)
    print("total_reward", total_reward)
    print("terminated", terminated)
    print("truncated", truncated)
    print("final_obs", observation[:6])
    if final_step > 0:
        mean_model_error = {
            key: value / final_step for key, value in model_error_sums.items()
        }
        std_model_error = {
            key: max(
                model_error_square_sums[key] / final_step
                - mean_model_error[key] * mean_model_error[key],
                0.0,
            )
            ** 0.5
            for key in model_error_sums
        }
        print("mean_model_error", mean_model_error)
        print("std_model_error", std_model_error)
        print("max_model_error", model_error_max)
    if args.record_video:
        print("video_dir", args.video_dir)


if __name__ == "__main__":
    main()
