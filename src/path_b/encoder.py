"""
src/path_b/encoder.py

Ticket B1: Dense multilingual embedding encoder.
Wraps SentenceTransformer with automatic hardware acceleration (CUDA / MPS / CPU),
batched inference, L2 normalization, and disk caching.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_default_device() -> str:
    """Detect the fastest available hardware accelerator with safe fallback."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Test a small SDPA attention operation on MPS (requires Metal shader compilation)
            q = torch.randn(1, 1, 4, 8, device="mps")
            out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            del q, out
            return "mps"
        except Exception:
            return "cpu"
    return "cpu"


class DenseEncoder:
    """Dense embedding encoder supporting multilingual sentence transformers."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_seq_length: int = 128,
        local_files_only: Optional[bool] = None,
    ) -> None:
        self.device = device or get_default_device()
        self.model_name = model_name_or_path
        self.max_seq_length = max_seq_length

        print(f"[DenseEncoder] Loading model '{self.model_name}' on device '{self.device}'...")
        from sentence_transformers import SentenceTransformer

        try:
            kwargs = {}
            if local_files_only is not None:
                kwargs["local_files_only"] = local_files_only
            self.model = SentenceTransformer(self.model_name, device=self.device, **kwargs)
        except Exception as e:
            # If network error, attempt local cached loading
            print(f"[DenseEncoder] Online fetch failed ({e}). Retrying with local_files_only=True...")
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            except Exception as e2:
                fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
                print(f"[DenseEncoder] Local load of '{self.model_name}' failed ({e2}). Falling back to cached '{fallback_model}'...")
                self.model_name = fallback_model
                self.model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)

        self.model.max_seq_length = self.max_seq_length

    def encode(
        self,
        texts: list[str],
        batch_size: int = 256,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True,
        is_query: bool = False,
        prefix: Optional[str] = None,
    ) -> np.ndarray:
        """Encode a list of text strings into normalized dense float32 vectors.

        Parameters
        ----------
        texts : list[str]
            Texts to encode.
        batch_size : int, optional
            Batch size for inference, by default 256.
        show_progress_bar : bool, optional
            Whether to display tqdm progress bar, by default True.
        normalize_embeddings : bool, optional
            If True, L2-normalizes vectors so dot product equals cosine similarity.
            By default True.
        is_query : bool, optional
            Whether the inputs are search queries (used for models with asymmetric prompts
            like intfloat/multilingual-e5), by default False.
        prefix : Optional[str], optional
            Explicit text prefix to prepend to each input, by default None.

        Returns
        -------
        np.ndarray
            Shape (len(texts), embedding_dim) as float32.
        """
        # Determine prefix for models requiring query/passage prompts (e.g., e5)
        active_prefix = prefix
        if active_prefix is None and "e5" in self.model_name.lower():
            active_prefix = "query: " if is_query else "passage: "

        if active_prefix:
            texts = [f"{active_prefix}{t}" for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    @staticmethod
    def save_embeddings(embeddings: np.ndarray, path: Union[str, Path]) -> None:
        """Save embeddings numpy array to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, embeddings)
        print(f"[DenseEncoder] Saved embeddings shape {embeddings.shape} to {target_path}")

    @staticmethod
    def load_embeddings(path: Union[str, Path]) -> np.ndarray:
        """Load embeddings numpy array from disk."""
        target_path = Path(path)
        if not target_path.exists() and not target_path.with_suffix(".npy").exists():
            raise FileNotFoundError(f"Embeddings file not found: {target_path}")
        if not str(target_path).endswith(".npy"):
            target_path = target_path.with_suffix(".npy")
        arr = np.load(target_path)
        print(f"[DenseEncoder] Loaded embeddings shape {arr.shape} from {target_path}")
        return arr
