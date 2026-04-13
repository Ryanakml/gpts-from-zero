"""Training utilities for chapter 05."""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(sum(losses) / max(1, len(losses)))


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses = []

    progress = tqdm(dataloader, desc="train", leave=False)
    for x, y in progress:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return float(sum(losses) / max(1, len(losses)))


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
) -> Dict[str, list[float]]:
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"epoch={epoch + 1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    return history
