"""
Deterministic preprocessing + ONNX Runtime inference.

This module is the one place the input contract (resize dims, normalization,
class order) is defined.  It must stay in lockstep with ``scripts/train_model.py``
— in a real system this pairing would ship as a single versioned artifact
(model.onnx + preprocessing config) — see the design note, Section 4.
"""

import io
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.inference.constants import CLASSES

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Loads the ONNX model once; reused across requests.

    Mirrors the long-lived worker-process pattern from the design note,
    minus the process pool — this slice is single-process for simplicity.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        resize_dim: int = 64,
        intra_op_threads: int = 1,
    ) -> None:
        # ``intra_op_num_threads`` pinned deliberately: see design note
        # Section 1.3 on avoiding thread oversubscription.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name: str = self.session.get_inputs()[0].name
        self.resize_dim = resize_dim
        logger.info(
            "ONNX model loaded from %s (resize=%d, threads=%d)",
            model_path,
            resize_dim,
            intra_op_threads,
        )

    # ── Preprocessing ───────────────────────────────────────────────

    def preprocess(self, raw_bytes: bytes) -> np.ndarray:
        """Deterministic: decode → RGB → resize (bilinear) → [0,1] → CHW.

        Input contract: (1, 3, resize_dim, resize_dim) float32 in [0, 1].
        Any change here changes model output for a given input, so this
        function's behavior is what must be version-pinned in production,
        not just the model weights.
        """
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img = img.resize((self.resize_dim, self.resize_dim), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
        arr = arr.transpose(2, 0, 1)  # → (3, H, W)
        return arr.reshape(1, 3, self.resize_dim, self.resize_dim)

    # ── Prediction ──────────────────────────────────────────────────

    def predict(self, raw_bytes: bytes) -> tuple[str, float, dict[str, float]]:
        """Returns ``(predicted_class, confidence, all_scores)``."""
        x = self.preprocess(raw_bytes)
        outputs = self.session.run(None, {self.input_name: x})
        scores = outputs[0][0]  # softmax probabilities, shape (N,)
        idx = int(np.argmax(scores))
        all_scores = {cls: round(float(s), 6) for cls, s in zip(CLASSES, scores)}
        return CLASSES[idx], float(scores[idx]), all_scores
