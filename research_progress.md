# LunarLander World Model Research Progress

## Goal

Build a world action model for `LunarLander-v3` that supports long-horizon planning through model-predictive control (MPC). The model predicts the next latent observation from the current latent observation and continuous action.

Observation latent currently uses the first six LunarLander observation components:

```text
[x, y, vx, vy, angle, angular_velocity]
```

The two leg-contact flags are dropped in training.

## Data And Encoding

- Official LunarLander-v3 observation bounds include signed values for `x`, `y`, velocities, angle, and angular velocity.
- The landing pad center is `(0, 0)`.
- Current encoding is centered range scaling:

```python
encoded = raw / (max - min)
```

- This maps the six trained components approximately into `[-0.5, 0.5]`.
- This is not min-max normalization because it does not subtract `min_vector`.

## Critical Dataset Bug Found And Fixed

Original `data_sampler.py` did not update `prev_observation` inside the episode loop. Every transition in an episode used the reset observation as `prev_observation`, while `current_observation` came from later steps.

Bad behavior:

```python
prev_observation, info = env.reset()
while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    append(prev_observation, action, observation)
    # missing: prev_observation = observation
```

Fix:

```python
prev_observation = observation
```

Evidence that revealed it:

- Model predicted large negative `vy` for actions whose real environment rollout had positive or mild `vy`.
- Dataset stats showed `prev_y` had very tiny variance while `current_y` had large variance.
- This indicated `prev_observation` was stuck near reset height.

Impact:

- Old checkpoints trained before this fix are not reliable for planning.
- After regenerating the dataset, one-step training and MPC behavior became much more physically plausible.

## Loss Experiments

### Removed Fixed Loss Segment

The old `fixed_loss` / `magnitude_scaler` code was removed because it was unused or unstable.

### Grouped Observation Loss

The output loss was reformulated around physical groups:

- position distance: Euclidean over `[x, y]`
- velocity distance: Euclidean over `[vx, vy]`
- angle error
- angular velocity error

Progress logs now include:

```text
t_pos, t_vel, t_vx, t_vy, t_ang, t_angv
v_pos, v_vel, v_vx, v_vy, v_ang, v_angv
```

### Scaling Small Transition Errors

Because true one-step diffs are very small, raw MSE often under-emphasized meaningful errors. A relative/baseline-normalized loss was tried.

Bad version:

```python
baseline + 1e5
```

This made the loss effectively zero.

Better current approach:

Use fixed dataset-level transition RMS scales:

```python
TRANSITION_POS_SCALE = 0.0032503658
TRANSITION_VEL_SCALE = 0.0040580052
TRANSITION_ANGLE_SCALE = 0.00158635
TRANSITION_ANGULAR_VEL_SCALE = 0.0128266
```

Loss:

```python
output_loss = (
    (pos_dist / TRANSITION_POS_SCALE).square().mean()
    + (vel_dist / TRANSITION_VEL_SCALE).square().mean()
    + (angle_dist / TRANSITION_ANGLE_SCALE).square().mean()
    + (angular_vel_dist / TRANSITION_ANGULAR_VEL_SCALE).square().mean()
)
```

This gave a major performance jump. The best scaled-loss checkpoint observed so far:

```text
checkpoints_bro/checkpoint_epoch=39_tl_epoch=4.1204.ckpt
```

## Architecture Experiments

### Absolute Prediction Failed For Rollouts

Original model predicted absolute next latent. Recursive rollout collapsed after about 3 model steps: different action sequences converged to the same latent.

### Delta/Residual Prediction

The model was changed to a residual update in bounded latent space.

Current bounded residual mapping:

```python
normalized_latent = 2 * (inp_latent - encoded_min) / encoded_range - 1
next_normalized_latent = tanh(atanh(normalized_latent) + delta_h)
next_latent = encoded_min + (next_normalized_latent + 1) * encoded_range / 2
```

This keeps predictions inside valid encoded observation bounds without hard clipping or `ceil`.

### Reduced Dynamics Input

The dynamics network input was changed to exclude position:

```text
[vx, vy, angle, angular_velocity, action_main, action_side]
```

The model still predicts deltas for all six output components. The idea was to reduce direct state-location memorization and focus on dynamics.

### Kinematic Prior Attempt Reverted

A hardcoded kinematic update was briefly tried:

```python
x += 0.0402 * vx
y += 0.0898 * vy
angle += 0.0806 * angular_velocity
```

These constants came from least-squares fits on the regenerated dataset. It was reverted because hardcoding them was too aggressive, especially for angle where residual variance was non-trivial.

Fit diagnostics:

```text
dx_from_vx:        R2 ~= 0.998
dy_from_vy:        R2 ~= 0.980
dangle_from_angv:  R2 ~= 0.729
```

## Solver And MPC Work

### Single-Shot Solvers

Two solvers have been used:

- Gradient/SGD-style action optimization
- Cross-Entropy Method (CEM)

CEM consistently performs better than the gradient solver. The original gradient solver often exits immediately due to a gradient norm threshold.

### MPC Script

Created:

```text
mpc_world_model.py
```

Behavior:

- Plan a horizon of actions with CEM.
- Execute only the first action in the real environment.
- Observe the real next state.
- Replan from the real state.

This is receding-horizon MPC and reduces long open-loop model drift.

Useful command:

```bash
uv run python mpc_world_model.py \
  --checkpoint-path checkpoints_bro/checkpoint_epoch=39_tl_epoch=4.1204.ckpt \
  --max-steps 400 \
  --horizon 6 \
  --population-size 256 \
  --elite-count 32 \
  --cem-iters 12 \
  --record-video \
  --video-dir videos/mpc_latest_scaled \
  --quiet
```

The MPC script also tracks predicted-vs-observed one-step latent error with mean/std/max for:

```text
latent_l2, pos, vel, vx, vy, ang, angv
```

### Planner Objective Tweaks

- `--vy-weight` added, default currently `20.0`.
- `--target-vy` added, default `-0.01` raw velocity units.
- `--angle-weight` added, default currently `10.0`.
- `--gravity` added for experiments, but changing gravity away from training gravity causes dynamics mismatch.

With hard-negative checkpoint and angle weight 10:

```text
total_reward improved from about -185 to about -52
```

But angular velocity remained problematic near termination.

## Hard Negative Mining

Created:

```text
train_hard_negatives.py
```

Current behavior:

- Load a checkpoint.
- Score every training sample by model prediction L2:

```python
score = torch.linalg.vector_norm(pred_observation - current_observation, dim=1)
```

- Select samples where:

```python
score > 0.01
```

- Fine-tune from the checkpoint with `lr=1e-5`.
- Save checkpoints to:

```text
checkpoints_hard_negative_finetune/
```

Observed mining stats from checkpoint `checkpoint_epoch=39_tl_epoch=4.1204.ckpt`:

```text
total_train_samples: 889361
hard_train_samples: 30594
hard_fraction: 0.0344
score_mean: 0.00407
score_std: 0.01323
score_max: 0.49740
selected_score_mean: 0.04580
selected_score_std: 0.05680
selected_score_min: 0.01000
selected_score_max: 0.49740
```

Best hard-negative checkpoint used so far:

```text
checkpoints_hard_negative_finetune/checkpoint_epoch=0_vl=17.7710.ckpt
```

## Angular-Velocity Hard Negative Mining

Created:

```text
train_hard_negatives_angv.py
```

Mining criterion:

```python
angv_error = abs(pred_observation[:, 5] - current_observation[:, 5])
hard_mask = angv_error > 0.01
```

Observed mining stats from checkpoint `checkpoint_epoch=39_tl_epoch=4.1204.ckpt`:

```text
total_train_samples: 889361
hard_train_samples: 16312
hard_fraction: 0.01834

angv_score_mean: 0.00233
angv_score_std: 0.01261
angv_score_max: 0.42075

selected_angv_score_mean: 0.06447
selected_angv_score_std: 0.06836
selected_angv_score_min: 0.01000
selected_angv_score_max: 0.42075
```

Checkpoint used after angular-velocity hard-negative training:

```text
checkpoints_hard_negative_angv/checkpoint_epoch=0_vl=5.9951.ckpt
```

MPC result with `--angle-weight 10`:

```text
steps: 132
total_reward: -269.60
terminated: True
final_obs: [1.0028, 0.0819, 1.5491, -0.6479, 0.6386, 0.9839]
```

Model error improved significantly:

```text
mean_model_error:
  latent_l2: 0.00583
  pos:       0.00311
  vel:       0.00234
  vx:        0.00085
  vy:        0.00201
  ang:       0.00351
  angv:      0.00152

max_model_error:
  latent_l2: 0.01375
  pos:       0.00912
  vel:       0.00699
  vx:        0.00323
  vy:        0.00692
  ang:       0.00799
  angv:      0.00663
```

This dramatically reduced the previous large angular-velocity error spikes. However, control performance worsened because the rollout failed by drifting right with large horizontal velocity.

## Current Failure Mode

After angular-velocity hard-negative fine-tuning, the major observed failure shifted from angular-velocity prediction spikes to horizontal control drift.

Latest failure state:

```text
x:     1.0028
y:     0.0819
vx:    1.5491
vy:   -0.6479
angle: 0.6386
angv:  0.9839
```

This suggests the next model/planner weakness is `x/vx` control under MPC, not just angular velocity.

Likely causes:

- Side thruster torque has threshold-like behavior in continuous LunarLander.
- CEM can exploit side thrusters aggressively.
- The model drops contact flags.
- Hard-negative fine-tuning on one failure component can improve that component while shifting the control failure to another component.
- Terminal-only planning cost can allow high spin at touchdown.
- Planner objective may need stronger `x/vx` weighting or rollout-wide cost, not only terminal cost.

## Next Direction

- Try a horizontal-velocity hard-negative pass using `abs(pred[:, 2] - current[:, 2])` or a combined `x/vx` score.
- Add planner weights for `x` and `vx` to reduce lateral drift.
- Add rollout-wide CEM cost so unstable intermediate states are penalized, not only terminal state.
- Consider side-action/smoothness penalties to reduce side-thruster abuse.

## Solver Fix: Heuristic-Prior Blended CEM

Pure terminal-cost CEM was not enough. It either drifted horizontally or discovered spin-heavy side-thruster plans because the world model does not include leg-contact flags and the old objective only cared about the short-horizon terminal latent.

`mpc_world_model.py` was updated with:

- rollout-wide trajectory cost instead of terminal-only cost
- explicit `x`, `vx`, `y`, `vy`, angle, and angular-velocity penalties
- heavier low-altitude landing penalties
- side-action and action-smoothness penalties
- warm-start from the previous MPC plan
- a Gym-style landing heuristic used as the CEM prior/reference plan
- injected heuristic candidate in every CEM population
- `--cem-action-blend` so the executed action can be constrained toward the stable landing prior

Important finding:

- CEM alone still exploited solver/model blind spots.
- The heuristic prior alone reaches the pad, but CEM deviations can make it hover or destabilize.
- A blended action with default `--cem-action-blend 0.25` landed successfully while still running CEM and tracking model error.

Successful command:

```bash
uv run python mpc_world_model.py \
  --checkpoint-path checkpoints_hard_negative_angv/checkpoint_epoch=0_vl=5.9951.ckpt \
  --max-steps 400 \
  --horizon 8 \
  --population-size 512 \
  --elite-count 64 \
  --cem-iters 16 \
  --quiet
```

Successful result:

```text
steps: 206
total_reward: 271.96
terminated: True
truncated: False
final_obs: [-0.0423, -0.0003, 0.0, 0.0, -0.0025, 0.0]
```

This is the first solver configuration in this run that cleanly lands with the angular-velocity hard-negative checkpoint under the unchanged simulator configuration.

Correction:

- The heuristic-prior blended CEM result is not considered a valid world-model-only solver result because it used a handwritten Gym-style landing controller as a stabilizing prior/action blend.
- It was useful diagnostically, but it should not be treated as the actual solution.

## Actual Solver Breakthrough: Minimal Latent-Space MPC

Created:

```text
minimal_cem_solver.py
```

This solver is intentionally minimal:

- pure learned-world-model CEM/MPC
- no raw-space decode inside the cost
- no handwritten heuristic controller
- no environment configuration changes except intentionally long episode allowance for slow descent experiments
- fixed short horizon
- fixed CEM constants
- only one CLI argument for checkpoint override
- timestamped video recording with frame counter overlay

Key change:

The cost is computed directly in the normalized/encoded latent space. The target latent was changed from a hard landing target to a slow-descent target:

```python
target = torch.tensor([0.0, 0.01, 0.0, -0.01, 0.0, 0.0])
weights = torch.tensor([1.75, 1.0, 1.0, 15.0, 20.0, 20.0])
```

Interpretation:

- center `x`
- stay slightly above pad height with `y ~= 0.01`
- keep horizontal velocity near zero
- descend slowly with `vy ~= -0.01`
- stay upright
- keep angular velocity near zero

Additional shaping:

```python
scaling = torch.linspace(0.8, 2.0, steps=states.shape[1])
diff[:, -1] *= 2
```

This puts more pressure on later horizon states and the final predicted state while keeping the objective very small/simple.

The solver also latches thrusters off when the real environment state is nearly on the pad with near-zero vertical velocity:

```python
if obs[1] < 1e-2 and abs(obs[3]) < 1e-2:
    action = [0.0, 0.0]
```

Successful run:

```bash
uv run python minimal_cem_solver.py
```

Successful result observed:

```text
steps: 391
total_reward: 232.28911815858288
terminated: True
truncated: False
final_obs: [-0.09927054, -0.00047553, 0.0, 0.0, -0.00759417, 0.0]
video_dir: videos/minimal_cem
```

Conclusion:

- The world model was good enough for landing.
- The earlier solver was overcomplicated and used poorly shaped targets/objectives.
- A simple latent-space MPC with a slow-descent target and a final thrusters-off latch landed successfully without a handwritten heuristic landing controller.
