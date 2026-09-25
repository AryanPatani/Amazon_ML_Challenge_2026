"""
src/path_b/cross_encoder.py

Ticket B3: Cross-Encoder model wrapper, batched re-ranking inference,
and binary cross-entropy fine-tuning for pairwise match probability scoring.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Optional, Union, TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from src.path_b.encoder import get_default_device
from src.path_b.cross_dataset import build_inference_pairs


DEFAULT_CROSS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _configure_cpu_mode() -> None:
    """Safely enforce CPU environment variables when MPS or CUDA is unavailable."""
    os.environ["ACCELERATE_USE_CPU"] = "true"
    if hasattr(torch.backends, "mps"):
        torch.backends.mps.is_available = lambda: False


class CrossEncoderReranker:
    """Wrapper around SentenceTransformer CrossEncoder with safe device handling."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_CROSS_MODEL,
        device: Optional[str] = None,
        max_length: int = 256,
        local_files_only: Optional[bool] = None,
    ) -> None:
        self.device = device or get_default_device()
        self.model_name = str(model_name_or_path)
        self.max_length = max_length

        if self.device == "cpu":
            _configure_cpu_mode()

        print(f"[CrossEncoder] Loading model '{self.model_name}' on device '{self.device}'...")
        from sentence_transformers import CrossEncoder

        kwargs = {}
        if local_files_only is not None:
            kwargs["local_files_only"] = local_files_only
        elif Path(self.model_name).exists():
            kwargs["local_files_only"] = True

        try:
            # Try local cache / files first for instant offline-safe load
            self.model = CrossEncoder(
                self.model_name,
                num_labels=1,
                max_length=self.max_length,
                device=self.device,
                local_files_only=True,
            )
        except Exception:
            try:
                self.model = CrossEncoder(
                    self.model_name,
                    num_labels=1,
                    max_length=self.max_length,
                    device=self.device,
                    **kwargs,
                )
            except Exception as e2:
                print(f"[CrossEncoder] Load on '{self.device}' failed ({e2}). Falling back to CPU...")
                _configure_cpu_mode()
                self.device = "cpu"
                self.model = CrossEncoder(
                    self.model_name,
                    num_labels=1,
                    max_length=self.max_length,
                    device="cpu",
                    local_files_only=True,
                )

    def predict_proba(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 128,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Score (query, candidate) text pairs, returning probabilities in [0.0, 1.0].

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            List of (s1_text, candidate_text) tuples.
        batch_size : int, optional
            Mini-batch size for forward inference, by default 128.
        show_progress_bar : bool, optional
            Whether to display tqdm bar, by default False.

        Returns
        -------
        np.ndarray
            1D array of float probabilities of length len(pairs).
        """
        if not pairs:
            return np.array([], dtype=np.float32)

        try:
            raw_scores = self.model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                activation_fn=torch.nn.Sigmoid(),
            )
        except Exception as e:
            err_msg = str(e)
            if any(term in err_msg for term in ["MPS", "Metal", "Placeholder storage", "Operation not permitted", "monolithic_metal"]):
                print(f"[CrossEncoder] MPS error during inference ({e}). Falling back to CPU...")
                _configure_cpu_mode()
                self.device = "cpu"
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(
                    self.model_name,
                    num_labels=1,
                    max_length=self.max_length,
                    device="cpu",
                    local_files_only=True,
                )
                raw_scores = self.model.predict(
                    pairs,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    activation_fn=torch.nn.Sigmoid(),
                )
            else:
                raise

        scores = np.asarray(raw_scores, dtype=np.float32).flatten()


        # Check if scores need sigmoid activation (if raw logits are outside [0, 1])
        if scores.size > 0 and (scores.min() < 0.0 or scores.max() > 1.0):
            scores = 1.0 / (1.0 + np.exp(-scores))

        return np.clip(scores, 0.0, 1.0)

    def rerank_candidates(
        self,
        candidate_dict: dict[str, list[Union[str, tuple[str, float]]]],
        s1_texts: dict[str, str],
        corpus_texts: dict[str, str],
        batch_size: int = 128,
        show_progress_bar: bool = True,
    ) -> dict[str, list[tuple[str, float]]]:
        """Score all candidate pairs and return per-S1 lists sorted by probability.

        Parameters
        ----------
        candidate_dict : dict[str, list[Union[str, tuple[str, float]]]]
            Mapping s1_id -> candidate IDs.
        s1_texts : dict[str, str]
            Mapping s1_id -> serialized text.
        corpus_texts : dict[str, str]
            Mapping corpus_id -> serialized text.
        batch_size : int, optional
            Batch size for model forward pass, by default 128.
        show_progress_bar : bool, optional
            Whether to show progress bar, by default True.

        Returns
        -------
        dict[str, list[tuple[str, float]]]
            Mapping s1_id -> list of (cand_id, probability) sorted descending by probability.
        """
        text_pairs, id_pairs = build_inference_pairs(candidate_dict, s1_texts, corpus_texts)
        if not text_pairs:
            return {s1_id: [] for s1_id in candidate_dict}

        print(f"[CrossEncoder] Scoring {len(text_pairs):,} candidate pairs in batches of {batch_size}...")
        probabilities = self.predict_proba(text_pairs, batch_size=batch_size, show_progress_bar=show_progress_bar)

        # Re-group by S1 ID
        scored_dict: dict[str, list[tuple[str, float]]] = {s1_id: [] for s1_id in candidate_dict}
        for (s1_id, cand_id), prob in zip(id_pairs, probabilities):
            scored_dict[s1_id].append((cand_id, float(prob)))

        # Sort each candidate list descending by score
        for s1_id in scored_dict:
            scored_dict[s1_id].sort(key=lambda x: x[1], reverse=True)

        return scored_dict


def fine_tune_cross_encoder(
    train_examples: list[Any],
    output_path: Union[str, Path] = "models/cross_encoder_b3",
    base_model_name: str = DEFAULT_CROSS_MODEL,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    device: Optional[str] = None,
    use_amp: bool = False,
    show_progress_bar: bool = True,
    seed: int = 42,
) -> CrossEncoder:
    """Fine-tune a CrossEncoder using binary cross-entropy loss.

    Parameters
    ----------
    train_examples : list[InputExample]
        List of InputExample instances with texts=[s1_text, cand_text] and label in {0.0, 1.0}.
    output_path : Union[str, Path], optional
        Directory where checkpoint will be saved, by default "models/cross_encoder_b3".
    base_model_name : str, optional
        Base transformer identifier, by default DEFAULT_CROSS_MODEL.
    epochs : int, optional
        Number of training epochs, by default 1.
    batch_size : int, optional
        Batch size for training, by default 16.
    learning_rate : float, optional
        AdamW learning rate, by default 2e-5.
    warmup_ratio : float, optional
        Fraction of total steps dedicated to linear warmup, by default 0.1.
    weight_decay : float, optional
        Weight decay for regularization, by default 0.01.
    device : Optional[str], optional
        Target hardware device ("cuda", "mps", "cpu"), by default auto-detected.
    use_amp : bool, optional
        Use automatic mixed precision (fp16), by default False.
    show_progress_bar : bool, optional
        Whether to show progress bars, by default True.
    seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    CrossEncoder
        The fine-tuned cross-encoder instance.
    """
    if not train_examples:
        raise ValueError("Cannot train cross-encoder: train_examples list is empty!")

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_device = device or get_default_device()
    if target_device == "cpu":
        _configure_cpu_mode()

    print(f"\n[CrossTrainer] Initializing base cross-encoder '{base_model_name}' on '{target_device}'...")
    from sentence_transformers import CrossEncoder

    def _load_model(dev: str) -> CrossEncoder:
        try:
            return CrossEncoder(base_model_name, num_labels=1, device=dev, local_files_only=True)
        except Exception:
            try:
                return CrossEncoder(base_model_name, num_labels=1, device=dev)
            except Exception as e:
                print(f"[CrossTrainer] Failed to load '{base_model_name}' on '{dev}' ({e}). Retrying with local_files_only=True on CPU...")
                _configure_cpu_mode()
                return CrossEncoder(base_model_name, num_labels=1, device="cpu", local_files_only=True)

    model = _load_model(target_device)

    eff_batch_size = max(1, min(batch_size, len(train_examples)))
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=eff_batch_size,
    )

    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    print(f"[CrossTrainer] Training: {len(train_examples):,} pairs, {epochs} epoch(s), "
          f"batch_size={eff_batch_size}, total_steps={total_steps:,}, "
          f"warmup_steps={warmup_steps:,}, lr={learning_rate}")

    try:
        model.fit(
            train_dataloader=train_dataloader,
            epochs=epochs,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            optimizer_params={"lr": learning_rate},
            output_path=str(out_dir),
            show_progress_bar=show_progress_bar,
            use_amp=use_amp,
        )
        model.save(str(out_dir))
    except Exception as e:
        err_msg = str(e)
        if any(term in err_msg for term in ["MPS", "Metal", "Placeholder storage", "Operation not permitted", "monolithic_metal"]):
            print(f"\n[CrossTrainer] Hardware accelerator issue on '{target_device}' ({e}). Falling back safely to CPU...")
            _configure_cpu_mode()
            model = _load_model("cpu")
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=eff_batch_size,
            )
            model.fit(
                train_dataloader=train_dataloader,
                epochs=epochs,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                optimizer_params={"lr": learning_rate},
                output_path=str(out_dir),
                show_progress_bar=show_progress_bar,
                use_amp=False,
            )
            model.save(str(out_dir))
        else:
            raise

    print(f"[CrossTrainer] Cross-encoder fine-tuning complete! Model saved to {out_dir}\n")
    return model

