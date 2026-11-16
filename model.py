import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def generate_dataset(
    num_samples: int, seed: int, num_features: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates synthetic non-linear regression data with Gaussian noise."""
    rng = np.random.default_rng(seed)
    X_np = rng.uniform(-3.0, 3.0, size=(num_samples, num_features))

    x1, x2, x3 = X_np[:, 0], X_np[:, 1], X_np[:, 2]
    Y_np = np.sin(x1) + 0.5 * (x2**2) - (x1 * x3)
    Y_np += rng.normal(0.0, 0.2, size=num_samples)

    xs = torch.tensor(X_np, dtype=torch.float32)
    ys = torch.tensor(Y_np.reshape(-1, 1), dtype=torch.float32)
    return xs, ys


def build_baseline_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(3, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 2),
        nn.LeakyReLU(),
        nn.Linear(2, 1),
    )


def build_improved_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(3, 128),
        nn.LeakyReLU(0.1),
        nn.Linear(128, 64),
        nn.LeakyReLU(0.1),
        nn.Linear(64, 32),
        nn.LeakyReLU(0.1),
        nn.Linear(32, 1),
    )


def train_baseline(
    model: nn.Module,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    xs_val: torch.Tensor,
    ys_val: torch.Tensor,
    checkpoint_path: str,
    epochs: int = 10000,
) -> tuple[list[float], list[float]]:
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    train_losses: list[float] = []
    val_losses: list[float] = []
    min_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(xs_train), ys_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(xs_val), ys_val).item()
            val_losses.append(val_loss)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": min_val_loss,
                    },
                    checkpoint_path,
                )

    return train_losses, val_losses


def train_improved(
    model: nn.Module,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    xs_val: torch.Tensor,
    ys_val: torch.Tensor,
    checkpoint_path: str,
    epochs: int = 6000,
) -> float:
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=500, min_lr=1e-6
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion_mse(model(xs_train), ys_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_loss = criterion_mae(model(xs_val), ys_val).item()
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), checkpoint_path)

            if epoch % 1000 == 0:
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Train MSE: {loss.item():.4f} | Val MAE: {val_loss:.4f}"
                )

    return best_val_loss


def plot_losses(train_losses: list[float], val_losses: list[float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("L1 Loss")
    axes[0].set_title("Baseline Training Loss")

    axes[1].plot(val_losses)
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("L1 Loss")
    axes[1].set_title("Baseline Validation Loss")

    plt.tight_layout()
    plt.show()


def evaluate_mae(model: nn.Module, xs: torch.Tensor, ys: torch.Tensor) -> float:
    criterion = nn.L1Loss()
    model.eval()
    with torch.no_grad():
        return criterion(model(xs), ys).item()


def print_sample_predictions(
    model: nn.Module,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    xs_val: torch.Tensor,
    ys_val: torch.Tensor,
    xs_test: torch.Tensor,
    ys_test: torch.Tensor,
    num_samples: int = 5,
) -> None:
    model.eval()
    with torch.no_grad():
        splits = [
            ("Train", model(xs_train), ys_train),
            ("Validation", model(xs_val), ys_val),
            ("Test", model(xs_test), ys_test),
        ]
        for name, preds, targets in splits:
            print(f"\n=== {name} Samples (First {num_samples}) ===")
            for i in range(num_samples):
                print(f"Pred: {preds[i].item():.4f} | GT: {targets[i].item():.4f}")


def main() -> None:
    os.makedirs("checkpoints", exist_ok=True)
    baseline_ckpt = "checkpoints/baseline_model.pt"
    improved_ckpt = "checkpoints/best_improved_model.pt"

    # 1. Generate Data Splits
    xs_train, ys_train = generate_dataset(num_samples=2000, seed=42)
    xs_val, ys_val = generate_dataset(num_samples=400, seed=253)
    xs_test, ys_test = generate_dataset(num_samples=200, seed=422)

    print(
        f"Shapes -> Train: {tuple(xs_train.shape)}, "
        f"Val: {tuple(xs_val.shape)}, Test: {tuple(xs_test.shape)}"
    )

    # 2. Train & Evaluate Baseline Model
    baseline_model = build_baseline_model()
    train_losses, val_losses = train_baseline(
        baseline_model, xs_train, ys_train, xs_val, ys_val, baseline_ckpt
    )
    plot_losses(train_losses, val_losses)

    checkpoint = torch.load(baseline_ckpt, weights_only=True)
    baseline_model.load_state_dict(checkpoint["model_state_dict"])
    baseline_test_mae = evaluate_mae(baseline_model, xs_test, ys_test)
    print_sample_predictions(
        baseline_model, xs_train, ys_train, xs_val, ys_val, xs_test, ys_test
    )

    # 3. Train & Evaluate Improved Model
    print("\n--- Training Improved Model ---")
    improved_model = build_improved_model()
    best_val_mae = train_improved(
        improved_model, xs_train, ys_train, xs_val, ys_val, improved_ckpt
    )
    print(f"\nTraining complete! Best Validation MAE: {best_val_mae:.4f}")

    improved_model.load_state_dict(torch.load(improved_ckpt, weights_only=True))
    improved_test_mae = evaluate_mae(improved_model, xs_test, ys_test)

    # 4. Benchmark Comparison
    print("\n=== Final Test Set Comparison ===")
    print(f"Baseline Model Test MAE: {baseline_test_mae:.4f}")
    print(f"Improved Model Test MAE: {improved_test_mae:.4f}")


if __name__ == "__main__":
    main()
