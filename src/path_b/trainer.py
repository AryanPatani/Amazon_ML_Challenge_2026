"""
src/path_b/trainer.py

Ticket B2: Contrastive Bi-Encoder Fine-Tuning.
Trains a SentenceTransformer using MultipleNegativesRankingLoss with in-batch negatives
and mined hard negatives from Ticket B1.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from src.path_b.encoder import get_default_device, DEFAULT_MODEL


import os
import random
import numpy as np


def fine_tune_bi_encoder(
    train_examples: list[Any],
    output_path: Union[str, Path] = "models/bi_encoder_b2",
    base_model_name: str = DEFAULT_MODEL,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    loss_scale: float = 20.0,
    device: Optional[str] = None,
    use_amp: bool = False,
    show_progress_bar: bool = True,
    seed: int = 42,
) -> SentenceTransformer:
    """Fine-tune a bi-encoder using contrastive MultipleNegativesRankingLoss.

    Parameters
    ----------
    train_examples : list[InputExample]
        List of InputExample instances with [anchor, positive, (negative)].
    output_path : Union[str, Path], optional
        Directory where fine-tuned model checkpoint will be saved, by default "models/bi_encoder_b2".
    base_model_name : str, optional
        Pretrained base model identifier, by default DEFAULT_MODEL.
    epochs : int, optional
        Number of training epochs, by default 1.
    batch_size : int, optional
        Batch size for training, by default 32.
    learning_rate : float, optional
        AdamW learning rate, by default 2e-5.
    warmup_ratio : float, optional
        Fraction of total steps dedicated to linear LR warmup, by default 0.1.
    weight_decay : float, optional
        AdamW weight decay for regularization, by default 0.01.
    loss_scale : float, optional
        Cosine similarity scaling factor in MultipleNegativesRankingLoss (1/tau), by default 20.0.
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
    SentenceTransformer
        The fine-tuned model instance.
    """
    if not train_examples:
        raise ValueError("Cannot train bi-encoder: train_examples list is empty!")

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_device = device or get_default_device()
    if target_device == "cpu":
        os.environ["ACCELERATE_USE_CPU"] = "true"
        if hasattr(torch.backends, "mps"):
            torch.backends.mps.is_available = lambda: False

    print(f"\n[Trainer] Loading base model '{base_model_name}' on device '{target_device}'...")
    from sentence_transformers import SentenceTransformer, losses

    def _load_model(dev: str) -> SentenceTransformer:
        try:
            return SentenceTransformer(base_model_name, device=dev)
        except Exception as e:
            print(f"[Trainer] Online load failed ({e}). Loading from local cache...")
            return SentenceTransformer(base_model_name, device=dev, local_files_only=True)

    model = _load_model(target_device)

    # Adjust batch size if dataset is smaller than requested batch
    eff_batch_size = max(1, min(batch_size, len(train_examples)))
    drop_last = len(train_examples) > eff_batch_size * 2

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=eff_batch_size,
        drop_last=drop_last,
    )

    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    train_loss = losses.MultipleNegativesRankingLoss(model, scale=loss_scale)

    print(f"[Trainer] Starting training: {len(train_examples):,} examples, "
          f"{epochs} epoch(s), batch_size={eff_batch_size}, total_steps={total_steps:,}, "
          f"warmup_steps={warmup_steps:,}, lr={learning_rate}, scale={loss_scale}")

    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            optimizer_params={"lr": learning_rate},
            output_path=str(out_dir),
            show_progress_bar=show_progress_bar,
            use_amp=use_amp,
        )
    except Exception as e:
        err_msg = str(e)
        if any(term in err_msg for term in ["MPS", "Metal", "Placeholder storage", "Operation not permitted", "monolithic_metal"]):
            print(f"\n[Trainer] Hardware accelerator issue on '{target_device}' ({e}). Falling back safely to CPU...")
            os.environ["ACCELERATE_USE_CPU"] = "true"
            if hasattr(torch.backends, "mps"):
                torch.backends.mps.is_available = lambda: False

            model = _load_model("cpu")
            train_loss = losses.MultipleNegativesRankingLoss(model, scale=loss_scale)
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=eff_batch_size,
                drop_last=drop_last,
            )
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                optimizer_params={"lr": learning_rate},
                output_path=str(out_dir),
                show_progress_bar=show_progress_bar,
                use_amp=False,
            )
        else:
            raise

    print(f"[Trainer] Fine-tuning complete! Model saved to {out_dir}\n")
    return model

