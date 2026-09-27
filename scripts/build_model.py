"""
Builds a small, fully deterministic ONNX classifier with no external
downloads, no training data, and no internet access — by design.

See README.md "About the model" for the full rationale.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper

# ── Single source of truth ──────────────────────────────────────────
# Import class list from the application package so the index ↔ label
# contract is defined in exactly one place.
try:
    from app.inference.constants import CLASSES
except ImportError:
    # Fallback when running outside the project's virtual environment
    # or before the app package is installed.  Keep in sync with
    # app/inference/constants.py.
    CLASSES = [
        "Forest", "River", "Residential", "Industrial",
        "AnnualCrop", "SeaLake", "Highway",
    ]

INPUT_DIM = 32 * 32 * 3  # resized image, flattened
NUM_CLASSES = len(CLASSES)


def build_and_save(path: str = "model.onnx", seed: int = 42) -> None:
    """Generate a seeded random ``Gemm → Softmax`` ONNX graph."""
    rng = np.random.default_rng(seed)

    # Small random projection, scaled so softmax isn't saturated
    W = (rng.standard_normal((INPUT_DIM, NUM_CLASSES)) * 0.01).astype(np.float32)
    b = (rng.standard_normal((NUM_CLASSES,)) * 0.01).astype(np.float32)

    W_init = helper.make_tensor(
        "W", TensorProto.FLOAT, W.shape, W.flatten().tolist()
    )
    b_init = helper.make_tensor(
        "b", TensorProto.FLOAT, b.shape, b.flatten().tolist()
    )

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, INPUT_DIM]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, NUM_CLASSES]
    )

    gemm_node = helper.make_node(
        "Gemm", inputs=["input", "W", "b"], outputs=["logits"]
    )
    softmax_node = helper.make_node(
        "Softmax", inputs=["logits"], outputs=["output"], axis=1
    )

    graph = helper.make_graph(
        [gemm_node, softmax_node],
        "eurosat_placeholder_classifier",
        [input_tensor],
        [output_tensor],
        initializer=[W_init, b_init],
    )

    model = helper.make_model(graph, producer_name="galaxeye-takehome")
    model.opset_import[0].version = 13
    model.ir_version = 9  # pin below installed onnxruntime's max supported IR version
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"Saved deterministic placeholder model to {path}")
    print(f"Classes (index order): {CLASSES}")


if __name__ == "__main__":
    build_and_save()
