# Non-Linear Synthetic Regression Benchmark (PyTorch)

A clean, modular PyTorch pipeline that benchmarks two Multi-Layer Perceptron (MLP) architectures on a synthetic, noisy 3D non-linear regression task. It compares a narrow-bottleneck baseline model against a wider, deeper architecture trained with learning rate scheduling and Mean Squared Error (MSE) optimization.

---

## Overview

The pipeline generates reproducible synthetic datasets governed by a non-linear function with feature interactions and additive Gaussian noise:

$$y = \sin(x_1) + 0.5x_2^2 - (x_1 \cdot x_3) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.2^2)$$

where input features $x_1, x_2, x_3 \sim \mathcal{U}(-3.0, 3.0)$.

### Key Features
* **Reproducible Data Splits:** Isolated NumPy `default_rng` seeds for Training ($N=2000$), Validation ($N=400$), and Test ($N=200$) sets.
* **Architecture Comparison:** Evaluates the impact of hidden layer bottleneck width (`64 -> 2 -> 1` vs. `128 -> 64 -> 32 -> 1`) on function approximation capacity.
* **Automated Checkpointing:** Tracks validation Mean Absolute Error (MAE) and saves the best-performing model weights to disk (`checkpoints/`).
* **Loss Visualization:** Plots training and validation loss curves side-by-side for the baseline model.

---

## Model Architectures & Training Setup

| Component | Baseline Model | Improved Model |
| :--- | :--- | :--- |
| **Hidden Layers** | `Linear(3, 64) -> Linear(64, 2) -> Linear(2, 1)` | `Linear(3, 128) -> Linear(128, 64) -> Linear(64, 32) -> Linear(32, 1)` |
| **Activation** | `LeakyReLU(0.01)` | `LeakyReLU(0.1)` |
| **Training Loss** | `L1Loss` (MAE) | `MSELoss` |
| **Evaluation Metric** | `L1Loss` (MAE) | `L1Loss` (MAE) |
| **Optimizer** | `AdamW` (`lr=1e-4`) | `AdamW` (`lr=1e-3`, `weight_decay=1e-4`) |
| **LR Scheduler** | None | `ReduceLROnPlateau` (`factor=0.5`, `patience=500`) |
| **Epochs** | `10,000` | `6,000` |

---

## Project Structure

```text
.
├── pipeline.py              # Main data generation, training, and evaluation script
├── checkpoints/             # Auto-generated directory for saved model weights
│   ├── baseline_model.pt
│   └── best_improved_model.pt
└── README.md
```

---

## Getting Started

### Prerequisites

* Python 3.9+
* PyTorch
* NumPy
* Matplotlib

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required dependencies:
   ```bash
   pip install torch numpy matplotlib
   ```

### Usage

Run the full training and benchmarking pipeline:

```bash
python pipeline.py
```

*(Note: Update `pipeline.py` to match your actual filename if named differently.)*

---

## Example Output

Running the script generates loss plots for the baseline model, prints sample predictions across all splits, and outputs a final benchmark comparison on the unseen test set:

```text
Shapes -> Train: (2000, 3), Val: (400, 3), Test: (200, 3)

=== Test Samples (First 5) ===
Pred: 2.1432 | GT: 2.2105
...

--- Training Improved Model ---
Epoch    0/6000 | Train MSE: 10.4821 | Val MAE: 2.4110
Epoch 1000/6000 | Train MSE: 0.0482 | Val MAE: 0.1745
...
Training complete! Best Validation MAE: 0.1632

=== Final Test Set Comparison ===
Baseline Model Test MAE: 0.6841
Improved Model Test MAE: 0.1658
```
*(Note: Because the target contains Gaussian noise with $\sigma = 0.2$, the theoretical Bayes-optimal MAE lower bound is $\sigma\sqrt{2/\pi} \approx 0.1596$.)*
