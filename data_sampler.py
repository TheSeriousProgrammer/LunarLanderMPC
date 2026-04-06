import gymnasium as gym
import numpy as np
import random
import tqdm


from datasets import Dataset, IterableDataset

action_observation = []

for _ in tqdm.tqdm(range(10000)):
    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        gravity=random.uniform(-8.0, -11.0),
        enable_wind=random.choice([True, False]),
        wind_power=random.uniform(0.0, 20.0),
        turbulence_power=random.uniform(0.0, 2.0),
    )

    prev_observation, info = env.reset()

    episode_over = False

    i = 0
    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        action_observation.append(
            {
                "prev_observation": prev_observation,
                "action": action,
                "current_observation": observation,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "ground_contacted": float(
                    observation[-1] == 1.0 and observation[-2] == 1.0
                ),
            }
        )

        episode_over = terminated or truncated
        i += 1

ds = Dataset.from_list(action_observation)
ds.save_to_disk("LunarLander_Sampled")
