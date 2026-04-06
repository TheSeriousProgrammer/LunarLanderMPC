from datasets.combine import _interleave_map_style_datasets
from numpy import load
import torch
import numpy as np

from datasets import load_dataset, load_from_disk
from torch.nn import BatchNorm1d
import lighting as L

ds = load_from_disk("LunarLander_Sampled")

split_ds = ds.train_test_split(test_size=0.2, seed=42)
train_ds = split_ds["train"]
test_ds = split_ds["test"]

LEVELS = 4
ENCODER_ANGLES = np.linspace(-np.pi / 4, np.pi / 4, num=LEVELS)


def hyperencode(inp_vector=np.array):
    inp_expanded = np.concat(
        [
            np.expand_dims(inp_vector, 1),
        ]
        * LEVELS,
        axis=1,
    )
    sin_encode = np.sin(ENCODER_ANGLES * inp_expanded)
    cos_encode = np.cos(ENCODER_ANGLES * inp_expanded)
    hyperencoded_vector = np.reshape(
        np.concat([sin_encode, cos_encode], axis=1), (sin_encode.shape[0], -1)
    )
    return hyperencoded_vector


def hyperdecode(encoded_vector=np.array):
    sin_encodes = np.reshape(
        encoded_vector[:, : encoded_vector.shape[-1] / 2],
        shape=(encoded_vector.shape[0], ENCODER_ANGLES, -1),
    )
    cos_encodes = np.reshape(
        encoded_vector[:, encoded_vector.shape[-1] / 2 :],
        shape=(encoded_vector.shape[0], ENCODER_ANGLES, -1),
    )

    decoded = np.mean(np.atan2(sin_encodes, cos_encodes) / ENCODER_ANGLES, axis=1)
    print(decoded.shape)
    return decoded


def minimal_obs_encode(inp_vector=np.array):
    min_vector = np.array([-2.5, -2.5, -10.0, -10.0, -2 * np.pi, -10, 0.0, 0.0])
    max_vector = np.array([2.5, 2.5, 10.0, 10.0, 2 * np.pi, 10, 1, 1])

    out_vector = (inp_vector - min_vector) / (max_vector - min_vector)
    return out_vector


def minimal_obs_decode(inp_vector=np.array):
    min_vector = np.array([-2.5, -2.5, -10.0, -10.0, -2 * np.pi, -10, 0.0, 0.0])
    max_vector = np.array([2.5, 2.5, 10.0, 10.0, 2 * np.pi, 10, 1, 1])

    out_vector = inp_vector * (max_vector - min_vector) + min_vector
    return out_vector


def minimal_action_encode(inp_vector=np.array):
    return inp_vector  # action space in continous and discrete both need not be normalized.. we are good


def minimal_action_decode(inp_vector=np.array):
    return inp_vector  # action space in continous and discrete both need not be normalized.. we are good


class LunarModel(torch.nn.Module):
    def lin_batch_gelu(self, inp_features: int, out_features: int):
        return torch.nn.Sequential(
            torch.nn.Linear(inp_features, out_features),
            torch.nn.BatchNorm1d(out_features),
            torch.nn.GELU(),
        )

    def __init__(self):
        super(self, LunarModel).__init__()

        self.propogate = torch.nn.Sequential(
            torch.nn.BatchNorm1d(10),
            self.lin_batch_gelu(10, 32),
            self.lin_batch_gelu(32, 64),
            self.lin_batch_gelu(64, 32),
            self.lin_batch_gelu(32, 8),
            torch.nn.Linear(8, 8),
            torch.nn.Sigmoid(),
        )

    def forward(self, inp_latent: torch.Tensor, action: torch.Tensor):
        inp = torch.concat([inp_latent, action], dim=1)
        return self.propogate(inp)


class LunarModelLightning(L.LightningModule):
    def __init__(self):
        self.model = LunarModel()

    def training_step(self, batch, batch_idx):
        prev_observation = batch["prev_observation"]
        action = batch["action"]
        current_observation = batch["current_observation"]

        predicted_observation = self.model(inp_latent = prev_observation, action = action)

        loss = torch.nn.functional.mse_loss(current_observation, predicted_observation)
        self.log("tl": loss.item())

    def test_step(self, batch, batch_idx):

        prev_observation = batch['prev_observation']
        action = batch["action"]
        current_observation = batch['current_observation']

        predicted_observation = self.model(inp_latent = prev_observation, action = action)
        self.log("vl":loss.item())

