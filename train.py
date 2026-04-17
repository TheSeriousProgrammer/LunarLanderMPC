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
    min_vector = np.array([-2.5, -2.5, -10.0, -10.0, -2 * np.pi, -10, 0.0, 0.0])
    max_vector = np.array([2.5, 2.5, 10.0, 10.0, 2 * np.pi, 10, 1, 1])

    out_vector = (inp_vector) / (max_vector - min_vector)
    return out_vector


def minimal_obs_decode(inp_vector=np.array):
    min_vector = np.array([-2.5, -2.5, -10.0, -10.0, -2 * np.pi, -10, 0.0, 0.0])
    max_vector = np.array([2.5, 2.5, 10.0, 10.0, 2 * np.pi, 10, 1, 1])

    out_vector = inp_vector * (max_vector - min_vector)
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

        self.propogate = torch.nn.Sequential(
            torch.nn.BatchNorm1d(8),
            self.lin_batch_gelu(8, 32),
            self.lin_batch_gelu(32, 64),
            self.lin_batch_gelu(64, 64),
            self.lin_batch_gelu(64, 32),
            self.lin_batch_gelu(32, 6),
            torch.nn.Linear(6, 6),
            torch.nn.Sigmoid(),
        )

    def forward(self, inp_latent: torch.Tensor, action: torch.Tensor):
        inp = torch.concat([inp_latent, action], dim=1)
        return (
            2 * self.propogate(inp) - 1
        )  # Sigmoid gives values between 0-1 , now it gives between -1, 1


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

    def training_step(self, batch, batch_idx):
        prev_observation = batch["prev_observation"]
        action = batch["action"]
        current_observation = batch["current_observation"]

        pred_observation = self.model(prev_observation, action)
        mse = torch.abs(current_observation - pred_observation).mean(dim=1)

        mse = torch.abs(current_observation - pred_observation).mean(dim=1)
        magnitude_scaler = (torch.abs(prev_observation - current_observation)).mean(
            dim=1
        )
        fixed_loss = mse / magnitude_scaler
        fixed_loss = torch.where(
            (fixed_loss < 1.2) & (magnitude_scaler < 1e-3), 0.0, mse / magnitude_scaler
        )
        fixed_loss = fixed_loss.mean()
        sigreg = sigreg_strong_loss(pred_observation)
        sigreg_prev = sigreg_strong_loss(prev_observation)
        self.log("tl", fixed_loss.item(), on_epoch=True, on_step=True, prog_bar=True)
        self.log(
            "ts",
            fixed_loss.item() / sigreg.item(),
            on_step=True,
            prog_bar=True,
        )
        self.log(
            "s",
            sigreg.item(),
            on_step=True,
            prog_bar=True,
        )
        return fixed_loss.mean() + 0.001 * sigreg

    def validation_step(self, batch, batch_idx):
        prev_observation = batch["prev_observation"]
        action = batch["action"]
        current_observation = batch["current_observation"]

        pred_observation = self.model(prev_observation, action)
        mse = torch.abs(current_observation - pred_observation).mean(dim=1)
        magnitude_scaler = (torch.abs(prev_observation - current_observation)).mean(
            dim=1
        )
        fixed_loss = mse / magnitude_scaler
        fixed_loss = torch.where(
            (fixed_loss < 1.2) & (magnitude_scaler < 1e-3), 0.0, mse / magnitude_scaler
        )
        fixed_loss = fixed_loss.mean()

        self.log("vl", fixed_loss.item(), on_epoch=True, on_step=True, prog_bar=True)

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
    lunar_lightning = LunarModelLightning()
    ds = load_from_disk("LunarLander_Sampled")
    split_ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=42)

    train_ds = split_ds["train"].with_transform(transform)
    test_ds = split_ds["test"].with_transform(transform)

    trainer = L.Trainer(
        max_epochs=40,
        logger=TensorBoardLogger(save_dir="logs/"),
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints_bro",
                save_top_k=2,
                monitor="vl_epoch",
                filename="checkpoint_{epoch}_{vl_epoch:.4f}",
            )
        ],
    )

    trainer.fit(
        lunar_lightning,
        train_dataloaders=DataLoader(train_ds, batch_size=1024 * 8),
        val_dataloaders=DataLoader(test_ds, batch_size=1024 * 8),
    )
