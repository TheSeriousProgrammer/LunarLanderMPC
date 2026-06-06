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


def should_send_nil_action(observation: np.ndarray, args):
    x, y, vx, vy, angle, angv = observation[:6]
    return (
        not args.disable_nil_settle
        and abs(x) <= args.nil_x_threshold
        and y <= args.nil_y_threshold
        and abs(vx) <= args.nil_vx_threshold
        and abs(vy) <= args.nil_vy_threshold
        and abs(angle) <= args.nil_angle_threshold
        and abs(angv) <= args.nil_angv_threshold
    )


def nil_gate_violation(observation: np.ndarray, args):
    x, y, vx, vy, angle, angv = observation[:6]
    checks = np.array(
        [
            abs(x) / args.nil_x_threshold,
            y / args.nil_y_threshold,
            abs(vx) / args.nil_vx_threshold,
            abs(vy) / args.nil_vy_threshold,
            abs(angle) / args.nil_angle_threshold,
            abs(angv) / args.nil_angv_threshold,
        ],
        dtype=np.float32,
    )
    return float(np.max(checks)), checks


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
def rollout_model_trajectory(
    model, current_latent: torch.Tensor, actions: torch.Tensor
):
    state = current_latent.expand(actions.shape[0], -1)
    states = []
    for step in range(actions.shape[1]):
        state = model(state, actions[:, step, :])
        states.append(state)
    return torch.stack(states, dim=1)


def trajectory_cost(
    trajectory: torch.Tensor,
    actions: torch.Tensor,
    current_raw: torch.Tensor,
    raw_scale: torch.Tensor,
    args,
):
    raw = trajectory * raw_scale.view(1, 1, -1)
    x = raw[:, :, 0]
    y = raw[:, :, 1]
    vx = raw[:, :, 2]
    vy = raw[:, :, 3]
    angle = raw[:, :, 4]
    angv = raw[:, :, 5]
    speed = torch.sqrt(vx.square() + vy.square() + 1e-8)
    distance = torch.sqrt(x.square() + y.square() + 1e-8)

    steps = torch.arange(
        1, trajectory.shape[1] + 1, dtype=trajectory.dtype, device=trajectory.device
    ).view(1, -1)
    time_weight = (steps / trajectory.shape[1]).square()

    current_y = current_raw[:, 1:2].expand_as(y).clamp_min(0.0)
    descent_target_y = (current_y - args.descent_per_step * steps).clamp_min(0.0)
    safe_target_vy = -torch.clamp(
        args.target_vy_base + args.target_vy_height_gain * y.clamp_min(0.0),
        min=args.min_descent_speed,
        max=args.max_descent_speed,
    )

    # If the lander is off-center, prefer horizontal velocity back toward the pad.
    target_vx = -args.x_correction_gain * x
    low_altitude = torch.sigmoid((args.low_altitude_threshold - y) * 20.0)

    state_cost = (
        args.x_weight * x.square()
        + args.y_weight * (y - descent_target_y).square()
        + args.vx_weight * (vx - target_vx).square()
        + args.vy_weight * (vy - safe_target_vy).square()
        + args.angle_weight * angle.square()
        + args.angv_weight * angv.square()
    )
    landing_cost = low_altitude * (
        args.landing_x_weight * x.square()
        + args.landing_y_weight * y.clamp_min(0.0).square()
        + args.landing_vx_weight * vx.square()
        + args.landing_vy_weight * vy.square()
        + args.landing_angle_weight * angle.square()
        + args.landing_angv_weight * angv.square()
    )

    reward_shape_cost = (
        args.reward_distance_weight * distance
        + args.reward_speed_weight * speed
        + args.reward_angle_weight * angle.abs()
        + args.reward_angv_weight * angv.abs()
    )
    safety_cost = (
        args.unsafe_angle_weight
        * (angle.abs() - args.safe_angle).clamp_min(0.0).square()
        + args.unsafe_angv_weight
        * (angv.abs() - args.safe_angv).clamp_min(0.0).square()
        + low_altitude
        * args.low_altitude_speed_weight
        * (speed - args.safe_landing_speed).clamp_min(0.0).square()
    )

    final_x = x[:, -1]
    final_y = y[:, -1]
    final_vx = vx[:, -1]
    final_vy = vy[:, -1]
    final_angle = angle[:, -1]
    final_angv = angv[:, -1]
    landing_score = torch.exp(
        -args.landing_bonus_x_weight * final_x.square()
        -args.landing_bonus_y_weight * final_y.clamp_min(0.0).square()
        -args.landing_bonus_vx_weight * final_vx.square()
        -args.landing_bonus_vy_weight * final_vy.square()
        -args.landing_bonus_angle_weight * final_angle.square()
        -args.landing_bonus_angv_weight * final_angv.square()
    )

    terminal_cost = (
        state_cost[:, -1]
        + landing_cost[:, -1]
        + reward_shape_cost[:, -1]
        + safety_cost[:, -1]
    )
    rollout_cost = (
        (state_cost + landing_cost + reward_shape_cost + safety_cost) * time_weight
    ).mean(dim=1)

    main = actions[:, :, 0]
    side = actions[:, :, 1]
    action_cost = (
        args.main_action_weight * main.square()
        + args.side_action_weight * side.square()
    ).mean(dim=1)
    action_cost = action_cost + (
        low_altitude * args.low_altitude_main_weight * main.clamp_min(0.0).square()
    ).mean(dim=1)
    unsafe_attitude = (
        (angle.abs() / args.safe_angle).clamp_min(1.0)
        + (angv.abs() / args.safe_angv).clamp_min(1.0)
        - 2.0
    )
    action_cost = action_cost + (
        args.low_altitude_side_weight * low_altitude * side.square()
        + args.unsafe_attitude_side_weight * unsafe_attitude * side.square()
    ).mean(dim=1)
    if actions.shape[1] > 1:
        action_delta = actions[:, 1:, :] - actions[:, :-1, :]
        action_cost = action_cost + args.action_smoothness_weight * action_delta.square().mean(
            dim=(1, 2)
        )

    return (
        rollout_cost
        + args.terminal_weight * terminal_cost
        + action_cost
        - args.landing_bonus_weight * landing_score
    )


@torch.no_grad()
def cem_plan(
    model,
    current_latent: torch.Tensor,
    raw_scale: torch.Tensor,
    horizon: int,
    population_size: int,
    elite_count: int,
    iterations: int,
    min_std: float,
    args,
    initial_mean: torch.Tensor | None = None,
):
    device = current_latent.device
    current_raw = current_latent * raw_scale.view(1, -1)
    if initial_mean is None:
        mean = torch.zeros(horizon, 2, dtype=current_latent.dtype, device=device)
    else:
        mean = initial_mean.to(device=device, dtype=current_latent.dtype).clone()
    std = torch.full_like(mean, args.initial_std)

    best_actions = None
    best_terminal = None
    best_deviation = float("inf")

    for _ in range(iterations):
        sampled_actions = mean + std * torch.randn(
            population_size, horizon, 2, dtype=current_latent.dtype, device=device
        )
        sampled_actions = sampled_actions.clamp(min=-1.0, max=1.0)
        sampled_actions[:, :, 1] = sampled_actions[:, :, 1].clamp(
            min=-args.side_action_limit, max=args.side_action_limit
        )
        if initial_mean is not None and population_size > 1:
            sampled_actions[1] = initial_mean.to(
                device=device, dtype=current_latent.dtype
            ).clamp(min=-1.0, max=1.0)
            sampled_actions[1, :, 1] = sampled_actions[1, :, 1].clamp(
                min=-args.side_action_limit, max=args.side_action_limit
            )

        trajectory = rollout_model_trajectory(model, current_latent, sampled_actions)
        deviations = trajectory_cost(
            trajectory=trajectory,
            actions=sampled_actions,
            current_raw=current_raw,
            raw_scale=raw_scale,
            args=args,
        )
        elite_indices = torch.topk(deviations, k=elite_count, largest=False).indices
        elites = sampled_actions[elite_indices]

        mean = elites.mean(dim=0)
        mean[:, 1] = mean[:, 1].clamp(
            min=-args.side_action_limit, max=args.side_action_limit
        )
        std = elites.std(dim=0, unbiased=False).clamp_min(min_std)

        iteration_best_idx = elite_indices[0]
        iteration_best_deviation = deviations[iteration_best_idx].item()
        if iteration_best_deviation < best_deviation:
            best_deviation = iteration_best_deviation
            best_actions = sampled_actions[iteration_best_idx].clone()
            best_terminal = trajectory[
                iteration_best_idx : iteration_best_idx + 1, -1, :
            ].clone()

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
    parser.add_argument("--initial-std", type=float, default=0.8)
    parser.add_argument("--x-weight", type=float, default=8.0)
    parser.add_argument("--y-weight", type=float, default=2.0)
    parser.add_argument("--vx-weight", type=float, default=1.5)
    parser.add_argument("--vy-weight", type=float, default=18.0)
    parser.add_argument("--angle-weight", type=float, default=80.0)
    parser.add_argument("--angv-weight", type=float, default=30.0)
    parser.add_argument("--landing-x-weight", type=float, default=120.0)
    parser.add_argument("--landing-y-weight", type=float, default=15.0)
    parser.add_argument("--landing-vx-weight", type=float, default=140.0)
    parser.add_argument("--landing-vy-weight", type=float, default=120.0)
    parser.add_argument("--landing-angle-weight", type=float, default=250.0)
    parser.add_argument("--landing-angv-weight", type=float, default=100.0)
    parser.add_argument("--terminal-weight", type=float, default=2.5)
    parser.add_argument("--x-correction-gain", type=float, default=0.15)
    parser.add_argument("--descent-per-step", type=float, default=0.035)
    parser.add_argument("--target-vy-base", type=float, default=0.08)
    parser.add_argument("--target-vy-height-gain", type=float, default=0.25)
    parser.add_argument("--min-descent-speed", type=float, default=0.08)
    parser.add_argument("--max-descent-speed", type=float, default=0.45)
    parser.add_argument("--low-altitude-threshold", type=float, default=0.25)
    parser.add_argument("--main-action-weight", type=float, default=0.03)
    parser.add_argument("--side-action-weight", type=float, default=1.0)
    parser.add_argument("--side-action-limit", type=float, default=0.49)
    parser.add_argument("--action-smoothness-weight", type=float, default=0.5)
    parser.add_argument("--reward-distance-weight", type=float, default=20.0)
    parser.add_argument("--reward-speed-weight", type=float, default=30.0)
    parser.add_argument("--reward-angle-weight", type=float, default=40.0)
    parser.add_argument("--reward-angv-weight", type=float, default=5.0)
    parser.add_argument("--safe-angle", type=float, default=0.45)
    parser.add_argument("--safe-angv", type=float, default=0.8)
    parser.add_argument("--safe-landing-speed", type=float, default=0.35)
    parser.add_argument("--unsafe-angle-weight", type=float, default=500.0)
    parser.add_argument("--unsafe-angv-weight", type=float, default=120.0)
    parser.add_argument("--low-altitude-speed-weight", type=float, default=200.0)
    parser.add_argument("--low-altitude-main-weight", type=float, default=2.0)
    parser.add_argument("--low-altitude-side-weight", type=float, default=30.0)
    parser.add_argument("--unsafe-attitude-side-weight", type=float, default=20.0)
    parser.add_argument("--landing-bonus-weight", type=float, default=120.0)
    parser.add_argument("--landing-bonus-x-weight", type=float, default=12.0)
    parser.add_argument("--landing-bonus-y-weight", type=float, default=20.0)
    parser.add_argument("--landing-bonus-vx-weight", type=float, default=20.0)
    parser.add_argument("--landing-bonus-vy-weight", type=float, default=20.0)
    parser.add_argument("--landing-bonus-angle-weight", type=float, default=25.0)
    parser.add_argument("--landing-bonus-angv-weight", type=float, default=8.0)
    parser.add_argument("--disable-nil-settle", action="store_true")
    parser.add_argument("--nil-x-threshold", type=float, default=0.18)
    parser.add_argument("--nil-y-threshold", type=float, default=0.10)
    parser.add_argument("--nil-vx-threshold", type=float, default=0.35)
    parser.add_argument("--nil-vy-threshold", type=float, default=0.35)
    parser.add_argument("--nil-angle-threshold", type=float, default=0.18)
    parser.add_argument("--nil-angv-threshold", type=float, default=0.60)
    parser.add_argument("--no-warm-start", action="store_true")
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
    raw_scale = torch.tensor(OBS_DECODE_SCALE, dtype=torch.float32, device=device)

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
    previous_plan = None
    nil_settle_count = 0
    nil_settle_latched = False
    best_nil_violation = float("inf")
    best_nil_checks = None
    best_nil_observation = None

    try:
        for step in range(args.max_steps):
            current_latent = encode_observation(observation, device)
            nil_violation, nil_checks = nil_gate_violation(observation, args)
            if nil_violation < best_nil_violation:
                best_nil_violation = nil_violation
                best_nil_checks = nil_checks
                best_nil_observation = observation[:6].copy()
            actions, pred_terminal, pred_deviation = cem_plan(
                model=model,
                current_latent=current_latent,
                raw_scale=raw_scale,
                horizon=args.horizon,
                population_size=args.population_size,
                elite_count=args.elite_count,
                iterations=args.cem_iters,
                min_std=args.min_std,
                args=args,
                initial_mean=None if args.no_warm_start else previous_plan,
            )

            pred_terminal_raw = decode_latent(pred_terminal)
            nil_settle_latched = nil_settle_latched or should_send_nil_action(
                observation, args
            ) or should_send_nil_action(pred_terminal_raw, args)
            nil_settle = nil_settle_latched
            if nil_settle:
                action = np.zeros(2, dtype=np.float32)
                nil_settle_count += 1
            else:
                action = actions[0].detach().cpu().numpy()
            if args.no_warm_start:
                previous_plan = None
            else:
                if nil_settle:
                    previous_plan = torch.zeros_like(actions)
                else:
                    previous_plan = torch.cat(
                        [actions[1:], actions[-1:].clone()], dim=0
                    ).detach()
            with torch.no_grad():
                executed_action = torch.from_numpy(action).view(1, 2).to(
                    device=device, dtype=torch.float32
                )
                pred_next_latent = model(current_latent, executed_action)
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
                    "nil_settle",
                    nil_settle,
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
    print("nil_settle_count", nil_settle_count)
    print("best_nil_violation", best_nil_violation)
    if best_nil_observation is not None:
        print("best_nil_observation", best_nil_observation)
        print("best_nil_checks", best_nil_checks)
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
