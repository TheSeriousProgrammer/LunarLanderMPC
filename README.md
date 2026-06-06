# LunarLander World Model MPC

This project trains a tiny world action model for `LunarLander-v3` and uses it inside a simple MPC/CEM solver.

This is not an RL policy. The model learns:

```text
state + action -> next state
```

Then a solver samples possible future actions, rolls them through the learned model, scores the predicted future states, executes one action in the real environment, and replans.

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

If Box2D/pygame dependencies are missing on your system, install the usual system packages for Gymnasium Box2D support.

## Data

The training code expects a sampled dataset at:

```text
LunarLander_Sampled/
```

Generate it with:

```bash
uv run python data_sampler.py
```

The dataset contains transitions like:

```text
previous_observation, action, current_observation
```

The model uses the first six LunarLander observation values:

```text
x, y, vx, vy, angle, angular_velocity
```

The two leg-contact flags are dropped for model training.

## Train The World Model

Run:

```bash
uv run python train.py
```

For a short smoke test:

```bash
uv run python train.py --test
```

Checkpoints are written to the configured checkpoint directory in `train.py`.

## Hard Negative Fine-Tuning

After training a base model, you can fine-tune on high-error transitions.

General prediction-error hard negatives:

```bash
uv run python train_hard_negatives.py \
  --checkpoint-path checkpoints_bro/checkpoint_epoch=39_tl_epoch=4.1204.ckpt
```

Angular-velocity hard negatives:

```bash
uv run python train_hard_negatives_angv.py \
  --checkpoint-path checkpoints_bro/checkpoint_epoch=39_tl_epoch=4.1204.ckpt
```

The minimal solver currently defaults to:

```text
./checkpoints_hard_negative_angv/checkpoint_epoch=0_vl=5.7813.ckpt
```

If your checkpoint has a different filename, pass it explicitly:

```bash
uv run python minimal_cem_solver.py --checkpoint-path path/to/checkpoint.ckpt
```

## Run The Minimal MPC Solver

Recommended entrypoint:

```bash
uv run python minimal_cem_solver.py
```

This runs a simple latent-space MPC solver:

- samples action sequences with CEM
- predicts future latents with the world model
- scores them against a slow-descent latent target
- executes the first action
- observes the real environment
- replans

The current minimal target is:

```python
target = torch.tensor([0.0, 0.01, 0.0, -0.01, 0.0, 0.0])
```

Meaning:

```text
center x, stay slightly above pad, zero horizontal velocity,
slow downward vertical velocity, upright angle, zero angular velocity
```

The solver also turns thrusters off once the lander is almost on the pad with near-zero vertical velocity.

## Videos

`minimal_cem_solver.py` records video automatically to:

```text
videos/minimal_cem/
```

The video includes:

- frame counter in the top-left
- red square in the bottom-right when model prediction error is high

The red square appears when predicted-vs-observed latent L2 error is greater than:

```text
0.01
```

This is the same threshold used for hard-negative mining.

## Research Notes

See:

```text
research_progress.md
```

This file tracks the major experiments, bugs, solver failures, and the final minimal latent-space MPC breakthrough.

## Acknowledgment

This project was built through an interactive debugging/research loop with OpenCode as the coding assistant. It helped with code edits, experiment tracking, solver iterations, video tooling, and documenting the final minimal MPC approach.

## Useful Files

```text
data_sampler.py                  # creates LunarLander transition dataset
train.py                         # trains the world model
train_hard_negatives.py          # hard-negative fine-tuning by L2 prediction error
train_hard_negatives_angv.py     # hard-negative fine-tuning by angular velocity error
minimal_cem_solver.py            # recommended minimal MPC solver
mpc_world_model.py               # older, more experimental MPC script
research_progress.md             # experiment log
```

## Notes

Generated datasets, checkpoints, logs, and videos are intentionally not meant to be committed.

Typical generated paths:

```text
LunarLander_Sampled/
checkpoints_*/
logs/
videos/
```
