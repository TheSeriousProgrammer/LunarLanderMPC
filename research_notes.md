This project is an attempt to make use of Yann LeCun's world model ideas (JEPA-style) in LunarLander.

Initial suspicion:
- I originally thought the issue might be that I did not put regularization losses on the predicted latent.

Current observation:
- It looks like the world model being developed is becoming invariant to the input latent spaces.
- One practical symptom is that the gradient norm is extremely bad, which suggests the planner is not getting a useful optimization signal from the learned dynamics.

Possible next steps:
- Change the training loss to something like:

```text
mse(pred_obs, current_obs) - mse(pred_obs, prev_obs)
```

- The idea is to discourage the model from collapsing toward simply copying or ignoring the transition structure.

- Alternatively, try implementing the latent regularization used by Yann's team in the original paper, referred to as `Sigreg`.

Status:
- These are research notes only.
- No implementation decision has been made yet.
