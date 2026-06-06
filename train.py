import argparse
from operator import call
from datasets.combine import _interleave_map_style_datasets
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from numpy import fix, load
import torch
import numpy as np
from torch.optim import SGD, Adam, Muon
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, load_from_disk
from torch.nn import BatchNorm1d
import lightning as L

ds = load_from_disk("LunarLander_Sampled")

split_ds = ds.train_test_split(test_size=0.2, seed=42)
train_ds = split_ds["train"]
test_ds = split_ds["test"]

LEVELS = 4
ENCODER_ANGLES = np.linspace(-np.pi / 4, np.pi / 4, num=LEVELS)
OBS_MIN_VECTOR = np.array([-2.5, -2.5, -10.0, -10.0, -2 * np.pi, -10, 0.0, 0.0])
OBS_MAX_VECTOR = np.array([2.5, 2.5, 10.0, 10.0, 2 * np.pi, 10, 1, 1])
OBS_SCALE_VECTOR = OBS_MAX_VECTOR - OBS_MIN_VECTOR
TRANSITION_POS_SCALE = 0.0032503658
TRANSITION_VEL_SCALE = 0.0040580052
TRANSITION_ANGLE_SCALE = 0.00158635
TRANSITION_ANGULAR_VEL_SCALE = 0.0128266


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
    return decoded


def minimal_obs_encode(inp_vector=np.array):
    out_vector = (inp_vector) / OBS_SCALE_VECTOR
    return out_vector


def minimal_obs_decode(inp_vector=np.array):
    out_vector = inp_vector * OBS_SCALE_VECTOR
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
        super(LunarModel, self).__init__()
        encoded_min = torch.from_numpy((OBS_MIN_VECTOR / OBS_SCALE_VECTOR)[:-2]).to(
            torch.float32
        )
        encoded_max = torch.from_numpy((OBS_MAX_VECTOR / OBS_SCALE_VECTOR)[:-2]).to(
            torch.float32
        )
        self.register_buffer("encoded_obs_min", encoded_min)
        self.register_buffer("encoded_obs_max", encoded_max)
        self.atanh_eps = 1e-6

        self.propogate = torch.nn.Sequential(
            torch.nn.BatchNorm1d(6),
            self.lin_batch_gelu(6, 32),
            self.lin_batch_gelu(32, 64),
            self.lin_batch_gelu(64, 64),
            self.lin_batch_gelu(64, 32),
            self.lin_batch_gelu(32, 6),
            torch.nn.Linear(6, 6),
            torch.nn.Tanh(),
        )

    def forward(self, inp_latent: torch.Tensor, action: torch.Tensor):
        dynamics_features = inp_latent[:, 2:6]
        inp = torch.concat([dynamics_features, action], dim=1)
        delta_h = self.propogate(inp)
        encoded_range = self.encoded_obs_max - self.encoded_obs_min
        normalized_latent = 2 * (inp_latent - self.encoded_obs_min) / encoded_range - 1
        normalized_latent = normalized_latent.clamp(
            -1 + self.atanh_eps, 1 - self.atanh_eps
        )

        next_normalized_latent = torch.tanh(torch.atanh(normalized_latent) + delta_h)
        return self.encoded_obs_min + (next_normalized_latent + 1) * encoded_range / 2


def sigreg_strong_loss(x, sketch_dim=64):
    """
    Forces ECF(x) ~ ECF(Gaussian).
    Matches ALL Moments (Maximum Entropy Cloud).
    Exact implementation of LeJEPA Algorithm 1.
    """
    N, C = x.size()
    x = (x - x.mean(dim=1).unsqueeze(-1)) / x.std(dim=1).unsqueeze(-1)

    # 1. Projection (The Observer)
    # Project channels down to sketch_dim
    A = torch.randn(C, sketch_dim, device=x.device)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)

    # 2. Integration Points
    t = torch.linspace(-5, 5, 17, device=x.device)

    # 3. Theoretical Gaussian CF
    exp_f = torch.exp(-0.5 * t**2)

    # 4. Empirical CF
    # proj: [N, sketch_dim]
    proj = x @ A

    # args: [N, sketch_dim, T]
    args = proj.unsqueeze(2) * t.view(1, 1, -1)

    # ecf: [sketch_dim, T] (Mean over batch)
    ecf = torch.exp(1j * args).mean(dim=0)

    # 5. Weighted L2 Distance
    # |ecf - gauss|^2 * gauss_weight
    diff_sq = (ecf - exp_f.unsqueeze(0)).abs().square()
    err = diff_sq * exp_f.unsqueeze(0)

    # 6. Integrate
    loss = torch.trapz(err, t, dim=1) * N

    return loss.mean()


class LunarModelLightning(L.LightningModule):
    def __init__(self):
        super(LunarModelLightning, self).__init__()
        self.model = LunarModel()

    def _observation_loss_and_distances(
        self,
        current_observation: torch.Tensor,
        pred_observation: torch.Tensor,
        prev_observation: torch.Tensor,
    ):
        diff = current_observation - pred_observation

        baseline = current_observation - prev_observation

        pos_dist = torch.linalg.vector_norm(diff[:, 0:2], dim=1)
        vel_dist = torch.linalg.vector_norm(diff[:, 2:4], dim=1)
        vx_dist = diff[:, 2].abs()
        vy_dist = diff[:, 3].abs()
        angle_dist = diff[:, 4].abs()
        angular_vel_dist = diff[:, 5].abs()

        baseline_pos_dist = torch.linalg.vector_norm(baseline[:, 0:2], dim=1)
        baseline_vel_dist = torch.linalg.vector_norm(baseline[:, 2:4], dim=1)
        baseline_angle_dist = baseline[:, 4].abs()
        baseline_angular_vel_dist = baseline[:, 5].abs()

        raw_loss = (
            pos_dist.square().mean()
            + vel_dist.square().mean()
            + angle_dist.square().mean()
            + angular_vel_dist.square().mean()
        )
        baseline_loss = (
            baseline_pos_dist.square().mean()
            + baseline_vel_dist.square().mean()
            + baseline_angle_dist.square().mean()
            + baseline_angular_vel_dist.square().mean()
        )

        gain = baseline_loss - raw_loss

        output_loss = (
            (pos_dist / TRANSITION_POS_SCALE).square().mean()
            + (vel_dist / TRANSITION_VEL_SCALE).square().mean()
            + (angle_dist / TRANSITION_ANGLE_SCALE).square().mean()
            + (angular_vel_dist / TRANSITION_ANGULAR_VEL_SCALE).square().mean()
        )

        return (
            gain,
            output_loss,
            {
                "pos": pos_dist.mean(),
                "vel": vel_dist.mean(),
                "vx": vx_dist.mean(),
                "vy": vy_dist.mean(),
                "ang": angle_dist.mean(),
                "angv": angular_vel_dist.mean(),
            },
        )

    def training_step(self, batch, batch_idx):
        action = batch["action"]
        current_observation = batch["current_observation"]

        pred_observation = self.model(batch["prev_observation"], action)
        gain, output_loss, distances = self._observation_loss_and_distances(
            current_observation, pred_observation, batch["prev_observation"]
        )

        tl = output_loss
        self.log("t_pos", distances["pos"], prog_bar=True)
        self.log("t_vel", distances["vel"], prog_bar=True)
        self.log("t_vx", distances["vx"], prog_bar=True)
        self.log("t_vy", distances["vy"], prog_bar=True)
        self.log("t_ang", distances["ang"], prog_bar=True)
        self.log("t_angv", distances["angv"], prog_bar=True)
        self.log("tl", tl, prog_bar=True, on_epoch=True)
        self.log("tgain", gain, prog_bar=True, on_epoch=True)
        return tl

    def validation_step(self, batch, batch_idx):
        prev_observation = batch["prev_observation"]
        action = batch["action"]
        current_observation = batch["current_observation"]

        pred_observation = self.model(prev_observation, action)
        gain, output_loss, distances = self._observation_loss_and_distances(
            current_observation, pred_observation, batch["prev_observation"]
        )

        vl = output_loss

        self.log("v_pos", distances["pos"], prog_bar=True)
        self.log("v_vel", distances["vel"], prog_bar=True)
        self.log("v_vx", distances["vx"], prog_bar=True)
        self.log("v_vy", distances["vy"], prog_bar=True)
        self.log("v_ang", distances["ang"], prog_bar=True)
        self.log("v_angv", distances["angv"], prog_bar=True)
        self.log("v_out", output_loss, prog_bar=True)
        self.log("v_gain", gain, prog_bar=True)
        self.log("vl", vl, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = Adam(params=self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.1, patience=1, min_lr=1e-6
                ),
                "monitor": "tl_epoch",
            },
        }


def transform(x: dict) -> dict:
    return {
        "prev_observation": torch.from_numpy(
            minimal_obs_encode(np.array(x["prev_observation"]))
        )[:, :-2].to(
            torch.float32
        ),  # the last 2 contact based stuff are barely useful as in close to 99.994% of the dataset it is 0, the world model would not learn it
        "action": torch.from_numpy(minimal_action_encode(np.array(x["action"]))).to(
            torch.float32
        ),
        "current_observation": torch.from_numpy(
            minimal_obs_encode(np.array(x["current_observation"]))
        )[:, :-2].to(torch.float32),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run three epochs for a quick training smoke test.",
    )
    args = parser.parse_args()

    lunar_lightning = LunarModelLightning()
    ds = load_from_disk("LunarLander_Sampled")
    split_ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=42)

    train_ds = split_ds["train"].with_transform(transform)
    test_ds = split_ds["test"].with_transform(transform)

    trainer = L.Trainer(
        max_epochs=3 if args.test else 40,
        logger=TensorBoardLogger(save_dir="logs/"),
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints_bro",
                save_top_k=2,
                monitor="tl_epoch",
                filename="checkpoint_{epoch}_{tl_epoch:.4f}",
            )
        ],
    )

    trainer.fit(
        lunar_lightning,
        train_dataloaders=DataLoader(train_ds, batch_size=1024 * 8),
        val_dataloaders=DataLoader(test_ds, batch_size=1024 * 8),
    )
