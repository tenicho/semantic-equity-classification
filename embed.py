"""Generate and cache semantic embeddings for company descriptions.

Kept as a standalone script because embedding 6k descriptions takes minutes;
the notebook loads the cached matrix instead of recomputing it on every run.
"""

import os

os.environ["USE_TF"] = "0"  # transformers pulls in TF/Keras 3 otherwise and fails

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
SOURCE = Path("data/companyStatesFull.parquet")
OUT_DIR = Path("data/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_VECS = OUT_DIR / "bge_base_v1p5.npy"
OUT_KEYS = OUT_DIR / "bge_base_v1p5_tickers.npy"


def main() -> None:
    df = pd.read_parquet(SOURCE)
    print(f"loaded {len(df):,} companies from {SOURCE}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"model: {MODEL_NAME} (max_seq_length={model.max_seq_length})")

    # Name + description: the name carries brand signal the description often
    # omits, and it costs only a few tokens.
    texts = (df["companyName"] + ". " + df["description"]).tolist()

    vecs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit sphere -> cosine similarity == dot product
    ).astype(np.float32)

    np.save(OUT_VECS, vecs)
    np.save(OUT_KEYS, df["ticker"].to_numpy())
    print(f"saved {vecs.shape} -> {OUT_VECS}")
    print(f"norm check (should be ~1.0): {np.linalg.norm(vecs, axis=1).mean():.6f}")


if __name__ == "__main__":
    main()
