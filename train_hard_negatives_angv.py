import argparse
from pathlib import Path

import lightning as L
import torch
from datasets import load_from_disk
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from train import LunarModelLightning, transform


class AngularVelocityHardNegativeLunarModelLightning(LunarModelLightning):
    def __init__(self, lr: float = 1e-5):
        super().__init__()
        self.lr = lr

    def configure_optimizers(self):
        optimizer = Adam(params=self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.1, patience=1, min_lr=1e-7
                ),
                "monitor": "vl",
            },
        }


@torch.no_grad()
def mine_hard_angv_indices(
    model: LunarModelLightning,
    dataset,
    threshold: float,
    batch_size: int,
):
    transformed_ds = dataset.with_transform(transform)
    dataloader = DataLoader(transformed_ds, batch_size=batch_size, shuffle=False)

    model.eval()
    device = next(model.parameters()).device
    hard_indices = []
    angv_scores = []
    latent_l2_scores = []
    start_idx = 0

    for batch in dataloader:
        prev_observation = batch["prev_observation"].to(device)
        action = batch["action"].to(device)
        current_observation = batch["current_observation"].to(device)

        pred_observation = model.model(prev_observation, action)
        diff = pred_observation - current_observation
        angv_score = diff[:, 5].abs()
        latent_l2_score = torch.linalg.vector_norm(diff, dim=1)

        hard_mask = angv_score > threshold
        batch_indices = torch.nonzero(hard_mask, as_tuple=False).flatten() + start_idx
        hard_indices.extend(batch_indices.cpu().tolist())
        angv_scores.append(angv_score.cpu())
        latent_l2_scores.append(latent_l2_score.cpu())
        start_idx += angv_score.shape[0]

    return hard_indices, torch.cat(angv_scores), torch.cat(latent_l2_scores)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset-path", default="LunarLander_Sampled")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024 * 8)
    parser.add_argument("--checkpoint-dir", default="checkpoints_hard_negative_angv")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    ds = load_from_disk(args.dataset_path)
    split_ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_ds = split_ds["train"]
    test_ds = split_ds["test"].with_transform(transform)

    lunar_lightning = (
        AngularVelocityHardNegativeLunarModelLightning.load_from_checkpoint(
            checkpoint_path,
            lr=args.lr,
        )
    )

    hard_indices, angv_scores, latent_l2_scores = mine_hard_angv_indices(
        model=lunar_lightning,
        dataset=train_ds,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )

    hard_count = len(hard_indices)
    total_count = len(train_ds)
    if hard_count == 0:
        raise RuntimeError(
            f"No angular-velocity hard negatives found with threshold {args.threshold}. "
            "Lower --threshold or use a different checkpoint."
        )

    selected_angv_scores = angv_scores[hard_indices]
    selected_l2_scores = latent_l2_scores[hard_indices]
    print("angv_hard_negative_threshold", args.threshold)
    print("total_train_samples", total_count)
    print("hard_train_samples", hard_count)
    print("hard_fraction", hard_count / total_count)
    print("angv_score_mean", angv_scores.mean().item())
    print("angv_score_std", angv_scores.std(unbiased=False).item())
    print("angv_score_max", angv_scores.max().item())
    print("selected_angv_score_mean", selected_angv_scores.mean().item())
    print("selected_angv_score_std", selected_angv_scores.std(unbiased=False).item())
    print("selected_angv_score_min", selected_angv_scores.min().item())
    print("selected_angv_score_max", selected_angv_scores.max().item())
    print("latent_l2_score_mean", latent_l2_scores.mean().item())
    print("latent_l2_score_std", latent_l2_scores.std(unbiased=False).item())
    print("selected_latent_l2_score_mean", selected_l2_scores.mean().item())
    print("selected_latent_l2_score_std", selected_l2_scores.std(unbiased=False).item())

    hard_train_ds = train_ds.select(hard_indices).with_transform(transform)
    lunar_lightning.train()

    trainer = L.Trainer(
        max_epochs=1 if args.test else args.max_epochs,
        logger=TensorBoardLogger(save_dir="logs/", name="hard_negatives_angv"),
        callbacks=[
            ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                save_top_k=2,
                monitor="vl",
                filename="checkpoint_{epoch}_{vl:.4f}",
            )
        ],
    )

    trainer.fit(
        lunar_lightning,
        train_dataloaders=DataLoader(
            hard_train_ds, batch_size=args.batch_size, shuffle=True
        ),
        val_dataloaders=DataLoader(test_ds, batch_size=args.batch_size),
    )


if __name__ == "__main__":
    main()
