"""
Train a CNN classifier on the EuroSAT candidate_tiles, export to ONNX,
and evaluate against the eval_set.

Architecture: lightweight CNN (3 conv blocks + FC head) trained from scratch.
Input contract: (1, 3, 64, 64) float32 normalized to [0,1]
Output contract: (1, 7) softmax probabilities

Usage:
    python scripts/train_model.py

The script will:
  1. Load candidate_tiles/ as training data (with 80/20 train/val split)
  2. Train a small CNN for ~30 epochs
  3. Export the best checkpoint to model.onnx
  4. Evaluate on eval_set/ using eval_labels.csv
  5. Print per-class metrics and overall accuracy
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT.parent / "Galaxeye-BE_MLSys-TakeHome_Assignment-Tiles" / "be-mlsys-assignment-dataset"
CANDIDATE_DIR = DATASET_ROOT / "candidate_tiles"
EVAL_DIR = DATASET_ROOT / "eval_set"
EVAL_LABELS = DATASET_ROOT / "eval_labels.csv"
MODEL_OUTPUT = PROJECT_ROOT / "model.onnx"

# ── Class taxonomy (must match app/inference/constants.py) ──────────
try:
    from app.inference.constants import CLASSES
except ImportError:
    CLASSES = ["Forest", "River", "Residential", "Industrial", "AnnualCrop", "SeaLake", "Highway"]

NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 64  # native size of EuroSAT tiles


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

class TileDataset(Dataset):
    """Loads labelled tiles from class-named subdirectories."""

    def __init__(self, root_dir: Path, augment: bool = False) -> None:
        self.samples: list[tuple[Path, int]] = []
        self.augment = augment
        for class_name in CLASSES:
            class_dir = root_dir / class_name
            if not class_dir.exists():
                print(f"  ⚠  {class_dir} not found, skipping")
                continue
            for fpath in sorted(class_dir.glob("*.png")):
                self.samples.append((fpath, CLASS_TO_IDX[class_name]))
        print(f"  Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]

        if self.augment:
            arr = self._augment(arr)

        # HWC → CHW
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        return tensor, label

    @staticmethod
    def _augment(arr: np.ndarray) -> np.ndarray:
        """Simple augmentations: random horizontal/vertical flip + small noise."""
        if np.random.random() > 0.5:
            arr = arr[:, ::-1, :].copy()  # horizontal flip
        if np.random.random() > 0.5:
            arr = arr[::-1, :, :].copy()  # vertical flip
        # Random rotation (0, 90, 180, 270)
        k = np.random.randint(0, 4)
        arr = np.rot90(arr, k=k, axes=(0, 1)).copy()
        # Small color jitter
        noise = np.random.normal(0, 0.02, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        return arr


class EvalDataset(Dataset):
    """Loads eval tiles with labels from CSV."""

    def __init__(self, eval_dir: Path, labels_csv: Path) -> None:
        self.samples: list[tuple[Path, int]] = []
        with open(labels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fpath = eval_dir / row["filename"]
                label = CLASS_TO_IDX[row["true_label"]]
                self.samples.append((fpath, label))
        print(f"  Loaded {len(self.samples)} eval samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        return tensor, label


# ═══════════════════════════════════════════════════════════════════
# Model Architecture
# ═══════════════════════════════════════════════════════════════════

class TileClassifierCNN(nn.Module):
    """Lightweight CNN for 64×64 satellite tile classification.

    Architecture:
        Conv(3→32, 3×3) → BN → ReLU → MaxPool(2)    → 32×32
        Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)   → 16×16
        Conv(64→128, 3×3) → BN → ReLU → MaxPool(2)  → 8×8
        Conv(128→256, 3×3) → BN → ReLU → AdaptiveAvgPool(4×4)
        Flatten → FC(4096→512) → ReLU → Dropout → FC(512→7)
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64×64 → 32×32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32×32 → 16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 8×8 → 4×4 (adaptive)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_model(
    num_epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 8,
) -> TileClassifierCNN:
    """Train the CNN and return the best model."""
    print("\n═══ Loading training data ═══")
    full_dataset = TileDataset(CANDIDATE_DIR, augment=True)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # Create non-augmented version for validation
    val_dataset_raw = TileDataset(CANDIDATE_DIR, augment=False)

    # Use deterministic split
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size], generator=generator
    )

    train_subset = torch.utils.data.Subset(full_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset_raw, val_indices.indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_subset)} | Val: {len(val_subset)}")

    # ── Model, loss, optimizer ──
    device = torch.device("cpu")
    model = TileClassifierCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # ── Training loop ──
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    print(f"\n═══ Training for up to {num_epochs} epochs (patience={patience}) ═══\n")
    for epoch in range(1, num_epochs + 1):
        # Train phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        train_loss /= train_total

        # Validation phase
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = " ★ best"
        else:
            epochs_no_improve += 1

        print(
            f"  Epoch {epoch:3d}/{num_epochs} │ "
            f"loss={train_loss:.4f} │ "
            f"train_acc={train_acc:.4f} │ "
            f"val_acc={val_acc:.4f}{marker}"
        )

        if epochs_no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Load best weights
    model.load_state_dict(best_state)
    print(f"\n  Best validation accuracy: {best_val_acc:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════
# ONNX Export
# ═══════════════════════════════════════════════════════════════════

def export_to_onnx(model: TileClassifierCNN, output_path: Path) -> None:
    """Export the trained model to ONNX with softmax on the output."""
    model.eval()

    # Wrap model to include softmax (PyTorch model outputs raw logits,
    # but the inference engine expects softmax probabilities)
    class ModelWithSoftmax(nn.Module):
        def __init__(self, base_model: nn.Module) -> None:
            super().__init__()
            self.base = base_model
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.base(x)
            return self.softmax(logits)

    export_model = ModelWithSoftmax(model)
    export_model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )

    # Verify with onnx and ensure single self-contained file
    import onnx
    onnx_model = onnx.load(str(output_path), load_external_data=True)
    onnx.save_model(onnx_model, str(output_path), save_as_external_data=False)
    data_file = output_path.with_name(f"{output_path.name}.data")
    if data_file.exists():
        data_file.unlink()
    onnx.checker.check_model(onnx_model)
    print(f"\n  ✓ ONNX model exported to {output_path}")
    print(f"    Input:  {onnx_model.graph.input[0].name} shape=(1, 3, {IMG_SIZE}, {IMG_SIZE})")
    print(f"    Output: {onnx_model.graph.output[0].name} shape=(1, {NUM_CLASSES})")

    # Quick sanity check with ONNX Runtime
    import onnxruntime as ort
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    test_output = session.run(None, {"input": dummy_input.numpy()})
    probs = test_output[0][0]
    print(f"    Sanity check — softmax sums to: {probs.sum():.4f}")
    print(f"    Sample probs: {dict(zip(CLASSES, [f'{p:.4f}' for p in probs]))}")


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_on_eval_set(model_path: Path) -> None:
    """Run the exported ONNX model on eval_set and print metrics."""
    import onnxruntime as ort

    print("\n═══ Evaluating on eval_set ═══")
    eval_dataset = EvalDataset(EVAL_DIR, EVAL_LABELS)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    correct = 0
    total = 0
    per_class_correct: dict[str, int] = {c: 0 for c in CLASSES}
    per_class_total: dict[str, int] = {c: 0 for c in CLASSES}
    confusion: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in CLASSES} for c in CLASSES}

    for images, labels in eval_loader:
        outputs = session.run(None, {input_name: images.numpy()})
        pred_idx = int(np.argmax(outputs[0][0]))
        true_idx = labels.item()
        pred_class = CLASSES[pred_idx]
        true_class = CLASSES[true_idx]

        per_class_total[true_class] += 1
        confusion[true_class][pred_class] += 1
        if pred_idx == true_idx:
            correct += 1
            per_class_correct[true_class] += 1
        total += 1

    accuracy = correct / total
    print(f"\n  Overall accuracy: {correct}/{total} = {accuracy:.2%}\n")

    # Per-class breakdown
    print(f"  {'Class':<15} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'─'*15} {'─'*8} {'─'*8} {'─'*10}")
    for cls in CLASSES:
        c = per_class_correct[cls]
        t = per_class_total[cls]
        acc = c / t if t > 0 else 0
        print(f"  {cls:<15} {c:>8} {t:>8} {acc:>10.2%}")

    # Confusion matrix
    print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    header = "  " + f"{'':>15}" + "".join(f"{c[:6]:>8}" for c in CLASSES)
    print(header)
    for true_cls in CLASSES:
        row = f"  {true_cls:>15}" + "".join(
            f"{confusion[true_cls][pred_cls]:>8}" for pred_cls in CLASSES
        )
        print(row)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not CANDIDATE_DIR.exists():
        print(f"ERROR: Dataset not found at {CANDIDATE_DIR}")
        print("Expected directory structure:")
        print(f"  {DATASET_ROOT}/")
        print("    candidate_tiles/  (with class subdirectories)")
        print("    eval_set/")
        print("    eval_labels.csv")
        sys.exit(1)

    # Train
    model = train_model(num_epochs=60, batch_size=32, lr=5e-4, patience=12)

    # Export
    export_to_onnx(model, MODEL_OUTPUT)

    # Evaluate
    evaluate_on_eval_set(MODEL_OUTPUT)

    print(f"\n═══ Done! model.onnx is ready at {MODEL_OUTPUT} ═══")
