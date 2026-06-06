from pathlib import Path
import re

import torch
import numpy as np
from torch.optim import SGD
from train import (
    LunarModelLightning,
    LunarModel,
    minimal_action_decode,
    minimal_obs_encode,
    minimal_action_encode,
)

def checkpoint_score(path: Path):
    match = re.search(r"tl_epoch=([0-9]+(?:\.[0-9]+)?)", path.name)
    tl_epoch = float(match.group(1)) if match else float("inf")
    return (tl_epoch, -path.stat().st_mtime)


def load_best_compatible_checkpoint():
    for path in sorted(Path("checkpoints_bro").glob("*.ckpt"), key=checkpoint_score):
        try:
            print(f"Loading checkpoint: {path}")
            return LunarModelLightning.load_from_checkpoint(path)
        except RuntimeError as exc:
            print(f"Skipping incompatible checkpoint: {path} ({exc.__class__.__name__})")
    raise RuntimeError("No compatible checkpoint found in checkpoints_bro")


lightning_model = load_best_compatible_checkpoint()
core_model = lightning_model.model
core_model = core_model.cpu().eval()
state_loss_weights = torch.ones(1, 6)
state_loss_weights[:, 3] = 20.0

for param in core_model.parameters():
    param.requires_grad = False


class Actions(torch.nn.Module):
    def __init__(self, core_model: LunarModel, steps: int):
        super(Actions, self).__init__()
        torch.manual_seed(56)
        self.seed_actions = torch.nn.Parameter(
            torch.clamp(torch.randn(steps, 2), min=-0.95, max=0.95)
        )
        self.core_model = core_model
        self.steps = steps

    def get_final_latent(self, current_obs):
        step_counter = self.steps
        state = current_obs

        while step_counter > 0:
            selected_action = self.seed_actions[self.steps - step_counter]
            pred_obs = self.core_model(state, selected_action.expand((1, 2)))
            state = pred_obs
            step_counter -= 1
        return state

    def compute_deviation(self, current_obs, desired_obs):
        final_pred_obs = self.get_final_latent(current_obs)
        return ((final_pred_obs - desired_obs).square() * state_loss_weights).sum()


def propagate(
    current_observation: torch.Tensor,
    desired_observation: torch.Tensor,
    propagatable_steps: int,
):
    searchable_actions = Actions(core_model=core_model, steps=propagatable_steps)
    optim = SGD(
        searchable_actions.parameters(recurse=False),
        lr=1e-1,
    )
    original_predicted_latent = searchable_actions.get_final_latent(current_observation)
    for v in searchable_actions.parameters(recurse=False):
        print(v)
    print("=========")
    prev_deviation = None
    i = 0
    while True:
        deviation = searchable_actions.compute_deviation(
            current_observation, desired_observation
        )
        loss = (
            deviation
            + 0.05 * searchable_actions.seed_actions.std()
            + 0.05 * searchable_actions.seed_actions.mean().abs()
        )
        print(i, searchable_actions.seed_actions, deviation.item())
        optim.zero_grad()
        loss.backward()

        if searchable_actions.seed_actions.grad.norm() < 0.05:
            print("Break at", i)
            break

        optim.step()
        i += 1
    else:
        print(
            "Warning Convergence possibly not reached yet!!, try increasing the epochs"
        )

    return {
        "optim_action": searchable_actions.seed_actions,
        "optim_latent": searchable_actions.get_final_latent(current_observation),
        "orig_latent": original_predicted_latent,
    }


def propagate_cem(
    current_observation: torch.Tensor,
    desired_observation: torch.Tensor,
    propagatable_steps: int,
):
    torch.manual_seed(56)

    population_size = 256
    elite_count = 32
    iterations = 12
    min_std = 0.05

    mean = torch.zeros(
        propagatable_steps,
        2,
        dtype=current_observation.dtype,
        device=current_observation.device,
    )
    std = torch.full_like(mean, 0.6)

    original_seed_actions = torch.clamp(torch.randn_like(mean), min=-0.95, max=0.95)
    searchable_actions = Actions(core_model=core_model, steps=propagatable_steps)
    with torch.no_grad():
        searchable_actions.seed_actions.copy_(original_seed_actions)
    original_predicted_latent = searchable_actions.get_final_latent(current_observation)

    best_actions = original_seed_actions.clone()
    best_deviation = float("inf")

    for i in range(iterations):
        sampled_actions = mean + std * torch.randn(
            population_size,
            propagatable_steps,
            2,
            dtype=current_observation.dtype,
            device=current_observation.device,
        )
        sampled_actions = torch.clamp(sampled_actions, min=-1.0, max=1.0)

        deviations = []
        for candidate in sampled_actions:
            with torch.no_grad():
                searchable_actions.seed_actions.copy_(candidate)
            deviation = searchable_actions.compute_deviation(
                current_observation, desired_observation
            )
            deviations.append(deviation.detach())

        deviations = torch.stack(deviations)
        elite_indices = torch.topk(deviations, k=elite_count, largest=False).indices
        elites = sampled_actions[elite_indices]

        mean = elites.mean(dim=0)
        std = torch.clamp(elites.std(dim=0, unbiased=False), min=min_std)

        best_iteration_idx = elite_indices[0]
        best_iteration_deviation = deviations[best_iteration_idx].item()
        print(i, sampled_actions[best_iteration_idx], best_iteration_deviation)
        if best_iteration_deviation < best_deviation:
            best_deviation = best_iteration_deviation
            best_actions = sampled_actions[best_iteration_idx].clone()

    with torch.no_grad():
        searchable_actions.seed_actions.copy_(best_actions)

    return {
        "optim_action": searchable_actions.seed_actions,
        "optim_latent": searchable_actions.get_final_latent(current_observation),
        "orig_latent": original_predicted_latent,
    }


if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        gravity=-10.0,
        enable_wind=False,
        wind_power=0.0,
        turbulence_power=0.0,
    )

    seed_observation, info = env.reset(seed=42)
    seed_obs = (
        torch.from_numpy(minimal_obs_encode(np.array(seed_observation))[:-2])
        .expand((1, 6))
        .to(torch.float32)
    )

    desired_obs = (
        torch.from_numpy(
            minimal_obs_encode([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])[:-2]
        )
        .expand((1, 6))
        .to(torch.float32)
    )

    vals = propagate(seed_obs, desired_obs, 6)

    print(seed_obs, desired_obs)
    print(vals)
