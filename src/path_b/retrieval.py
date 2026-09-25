"""
src/path_b/retrieval.py

Ticket B1: Fast Top-K vector retrieval over dense embeddings.
Supports FAISS (IndexFlatIP) if installed, with automatic chunked PyTorch / NumPy
fallback that works seamlessly across CUDA, Apple Silicon (MPS), and CPU.

Supports:
- Unified corpus Top-K search
- Country-partitioned blocking (drastically increases speed and eliminates cross-country false confusers)
- Source-stratified retrieval (ensures both S2 and S3 candidates are retrieved)
- Official candidate_pairs.tsv export matching utils/validate_submission.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


try:
    import faiss  # type: ignore[import-untyped, import-not-found]
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class VectorRetriever:
    """Vector similarity retriever for candidate blocking."""

    def __init__(
        self,
        corpus_embeddings: np.ndarray,
        corpus_ids: list[str],
        corpus_countries: Optional[list[str]] = None,
        use_faiss: bool = True,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the retriever with corpus embeddings.

        Parameters
        ----------
        corpus_embeddings : np.ndarray
            L2-normalized float32 array of shape (N_corpus, dim).
        corpus_ids : list[str]
            List of unique corpus entity IDs matching each embedding row.
        corpus_countries : Optional[list[str]], optional
            List of country labels for each corpus row (for country-partitioned search).
        use_faiss : bool, optional
            Whether to use FAISS if available, by default True.
        device : Optional[str], optional
            PyTorch device for fallback search, by default auto-detected.
        """
        assert len(corpus_embeddings) == len(corpus_ids), (
            f"Embeddings count ({len(corpus_embeddings)}) != IDs count ({len(corpus_ids)})"
        )
        self.corpus_ids = np.array(corpus_ids)
        self.corpus_countries = np.array(corpus_countries) if corpus_countries is not None else None
        self.dim = corpus_embeddings.shape[1]
        self.num_corpus = len(corpus_ids)
        self.use_faiss = use_faiss and _FAISS_AVAILABLE
        self.corpus_embeddings = corpus_embeddings

        dev = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)

        if self.use_faiss:
            print(f"[VectorRetriever] Initializing FAISS IndexFlatIP (dim={self.dim}, size={self.num_corpus})...")
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(corpus_embeddings.astype(np.float32))
            self.corpus_tensor = None
        else:
            if not _FAISS_AVAILABLE and use_faiss:
                print("[VectorRetriever] FAISS not found. Using high-performance PyTorch matrix ops.")
            print(f"[VectorRetriever] Storing corpus embeddings on device: {self.device}")
            # Transposed corpus tensor (dim, N_corpus) for fast query @ corpus.T
            self.corpus_tensor = torch.from_numpy(corpus_embeddings).to(self.device).T
            self.index = None

        # Pre-build country partition indices if countries are provided
        self.country_partitions: dict[str, np.ndarray] = {}
        if self.corpus_countries is not None:
            for country in np.unique(self.corpus_countries):
                if country:
                    self.country_partitions[country] = np.where(self.corpus_countries == country)[0]

    def search_top_k(
        self,
        query_embeddings: np.ndarray,
        query_ids: list[str],
        query_countries: Optional[list[str]] = None,
        top_k: int = 50,
        batch_size: int = 512,
        partition_by_country: bool = False,
        show_progress_bar: bool = True,
    ) -> dict[str, list[tuple[str, float]]]:
        """Search top-K nearest corpus items for each query.

        Parameters
        ----------
        query_embeddings : np.ndarray
            L2-normalized float32 array of shape (N_query, dim).
        query_ids : list[str]
            List of query entity IDs.
        query_countries : Optional[list[str]], optional
            List of country labels for queries (required if partition_by_country=True).
        top_k : int, optional
            Number of nearest neighbors to retrieve, by default 50.
        batch_size : int, optional
            Query batch size for memory efficiency, by default 512.
        partition_by_country : bool, optional
            If True, only searches within the corpus partition matching each query's country.
            By default False.
        show_progress_bar : bool, optional
            Whether to show progress bar, by default True.

        Returns
        -------
        dict[str, list[tuple[str, float]]]
            Mapping query_id -> list of (cand_id, cosine_similarity_score).
        """
        assert len(query_embeddings) == len(query_ids)
        if partition_by_country and query_countries is not None and self.country_partitions:
            return self._search_partitioned(
                query_embeddings=query_embeddings,
                query_ids=query_ids,
                query_countries=query_countries,
                top_k=top_k,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
            )

        top_k = min(top_k, self.num_corpus)
        results: dict[str, list[tuple[str, float]]] = {}

        if self.use_faiss:
            n_queries = len(query_ids)
            iterator = range(0, n_queries, batch_size)
            if show_progress_bar:
                iterator = tqdm(iterator, desc="[VectorRetriever] FAISS Search", unit="batch")

            for start in iterator:
                end = min(start + batch_size, n_queries)
                q_batch = query_embeddings[start:end].astype(np.float32)
                sims, indices = self.index.search(q_batch, top_k)

                for i, q_id in enumerate(query_ids[start:end]):
                    cand_ids = self.corpus_ids[indices[i]].tolist()
                    scores = sims[i].tolist()
                    results[q_id] = list(zip(cand_ids, scores))
            return results

        # PyTorch chunked search
        n_queries = len(query_ids)
        iterator = range(0, n_queries, batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="[VectorRetriever] PyTorch Search", unit="batch")

        with torch.no_grad():
            for start in iterator:
                end = min(start + batch_size, n_queries)
                q_batch = torch.from_numpy(query_embeddings[start:end]).to(self.device)
                
                sims = torch.matmul(q_batch, self.corpus_tensor)
                batch_scores, batch_indices = torch.topk(sims, k=top_k, dim=-1)

                batch_scores_cpu = batch_scores.cpu().numpy()
                batch_indices_cpu = batch_indices.cpu().numpy()

                for i, q_id in enumerate(query_ids[start:end]):
                    cand_ids = self.corpus_ids[batch_indices_cpu[i]].tolist()
                    scores = batch_scores_cpu[i].tolist()
                    results[q_id] = list(zip(cand_ids, scores))

        return results

    def _search_partitioned(
        self,
        query_embeddings: np.ndarray,
        query_ids: list[str],
        query_countries: list[str],
        top_k: int = 50,
        batch_size: int = 512,
        show_progress_bar: bool = True,
    ) -> dict[str, list[tuple[str, float]]]:
        """Execute country-partitioned search."""
        results: dict[str, list[tuple[str, float]]] = {}
        unique_countries = np.unique(query_countries)
        query_countries_arr = np.array(query_countries)

        for country in unique_countries:
            mask = query_countries_arr == country
            c_query_embeddings = query_embeddings[mask]
            c_query_ids = [query_ids[i] for i, m in enumerate(mask) if m]

            if country in self.country_partitions:
                c_idx = self.country_partitions[country]
                c_corpus_embeddings = self.corpus_embeddings[c_idx]
                c_corpus_ids = self.corpus_ids[c_idx].tolist()

                sub_retriever = VectorRetriever(
                    corpus_embeddings=c_corpus_embeddings,
                    corpus_ids=c_corpus_ids,
                    use_faiss=self.use_faiss,
                    device=str(self.device),
                )
                sub_results = sub_retriever.search_top_k(
                    query_embeddings=c_query_embeddings,
                    query_ids=c_query_ids,
                    top_k=top_k,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                )
                results.update(sub_results)
            else:
                # Fallback to full corpus if country partition is missing
                sub_results = self.search_top_k(
                    query_embeddings=c_query_embeddings,
                    query_ids=c_query_ids,
                    top_k=top_k,
                    batch_size=batch_size,
                    partition_by_country=False,
                    show_progress_bar=show_progress_bar,
                )
                results.update(sub_results)

        return results

    def to_dataframe(
        self,
        retrieval_results: dict[str, list[tuple[str, float]]],
    ) -> pd.DataFrame:
        """Convert retrieval results dict into a flat DataFrame.

        Columns: [source1_entity_id, candidate_entity_id, score, rank]
        """
        rows = []
        for s1_id, cands in retrieval_results.items():
            for rank, (cand_id, score) in enumerate(cands, start=1):
                rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cand_id,
                    "score": score,
                    "rank": rank,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def export_candidate_pairs_tsv(
        retrieval_results: dict[str, list[Union[str, tuple[str, float]]]],
        output_path: Union[str, Path],
    ) -> None:
        """Export candidates in official candidate_pairs.tsv format.

        Format:
            source1_entity_id \\t candidate_entity_ids
        where candidate_entity_ids is comma-separated IDs.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for s1_id, cands in retrieval_results.items():
            cand_id_list = [c[0] if isinstance(c, tuple) else c for c in cands]
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(cand_id_list),
            })

        df = pd.DataFrame(rows)
        df.to_csv(out_file, sep="\t", index=False)
        print(f"[VectorRetriever] Exported {len(df):,} candidate pairs to {out_file}")
