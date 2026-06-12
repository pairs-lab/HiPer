"""
seliSlam.object_encoder.memory_bank
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In-memory object embedding bank for BYE cross-session re-identification.

Stores one embedding per object (the running mean of all partial
observations from the reference session). At query time, performs
a k-NN search in embedding space and returns the best match by
majority vote — exactly as in BYE §IV-E.
"""

import numpy as np
from typing import Optional, Tuple, List


class ObjectMemoryBank:
    """
    Fixed in-memory embedding store.

    Parameters
    ----------
    embed_dim : int   Embedding dimensionality.
    knn_k     : int   Number of nearest neighbours to retrieve.
    """

    def __init__(self, embed_dim: int = 256, knn_k: int = 10,
                 sigma_u: float = 0.25):
        self._embed_dim = embed_dim
        self._knn_k     = knn_k
        self._sigma_u   = sigma_u

        # Parallel lists — index i corresponds to one stored embedding
        self._ids:        List[str]  = []
        self._labels:     List[str]  = []
        self._uncert:     List[float] = []          # per-object u_j
        self._embeddings: Optional[np.ndarray] = None  # (M, embed_dim)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add(self, object_id: str, embedding: np.ndarray, class_label: str,
            uncertainty: float = 0.0):
        """
        Add or update an object's embedding.
        If object_id already exists, replace with new embedding.
        `uncertainty` is the stored scalar u_j = trace(C_j)/d, used to
        down-weight low-confidence candidates at query time.
        """
        emb = embedding.astype(np.float32).flatten()
        norm = np.linalg.norm(emb)
        emb  = emb / (norm + 1e-8)

        if object_id in self._ids:
            idx = self._ids.index(object_id)
            self._embeddings[idx] = emb
            self._labels[idx] = class_label
            self._uncert[idx] = float(uncertainty)
        else:
            self._ids.append(object_id)
            self._labels.append(class_label)
            self._uncert.append(float(uncertainty))
            if self._embeddings is None:
                self._embeddings = emb[np.newaxis, :]
            else:
                self._embeddings = np.vstack([self._embeddings, emb])

    def clear(self):
        self._ids        = []
        self._labels     = []
        self._uncert     = []
        self._embeddings = None

    def _gamma(self, idx: int, query_uncertainty: float) -> float:
        """Uncertainty down-weight gamma = exp(-(u_i + u_j) / 2 sigma_u^2)."""
        u = float(query_uncertainty) + float(self._uncert[idx])
        return float(np.exp(-u / (2.0 * self._sigma_u ** 2 + 1e-12)))

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, embedding: np.ndarray,
              query_uncertainty: float = 0.0) -> Tuple[str, float]:
        """
        Find the best-matching object for a query embedding.

        The ensemble/KNN score of each candidate is down-weighted by the
        uncertainty factor gamma = exp(-(u_i + u_j) / 2 sigma_u^2), so that
        confident, stable embeddings dominate the match.

        Returns
        -------
        (matched_object_id, similarity_score)
        Returns ("", 0.0) if the bank is empty.
        """
        if self._embeddings is None or len(self._ids) == 0:
            return "", 0.0

        emb = embedding.astype(np.float32).flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        # Cosine similarities with all stored embeddings, gated by confidence
        sims = self._embeddings @ emb                    # (M,)
        gated = np.array(
            [sims[i] * self._gamma(i, query_uncertainty)
             for i in range(len(self._ids))],
            dtype=np.float32)

        k = min(self._knn_k, len(self._ids))
        top_k_idx = np.argpartition(-gated, k - 1)[:k]   # unsorted top-k
        top_k_idx = top_k_idx[np.argsort(-gated[top_k_idx])]  # sorted

        # Majority vote among k neighbours (weighted by gated similarity)
        vote_counts: dict[str, float] = {}
        for idx in top_k_idx:
            oid  = self._ids[idx]
            vote_counts[oid] = vote_counts.get(oid, 0.0) + float(gated[idx])

        best_id    = max(vote_counts, key=vote_counts.__getitem__)
        best_score = float(gated[self._ids.index(best_id)])

        return best_id, best_score

    def query_topk(
        self, embedding: np.ndarray, k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Return the top-k matches as [(object_id, similarity), ...].
        Useful for ensemble scoring.
        """
        if self._embeddings is None:
            return []

        emb = embedding.astype(np.float32).flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        sims = self._embeddings @ emb
        k = min(k, len(self._ids))
        idx = np.argsort(-sims)[:k]
        return [(self._ids[i], float(sims[i])) for i in idx]

    # ── Introspection ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._ids)

    def object_ids(self) -> List[str]:
        return list(self._ids)

    def embeddings_matrix(self) -> Optional[np.ndarray]:
        """(M, embed_dim) matrix of stored embeddings (row i = ids()[i])."""
        return None if self._embeddings is None else self._embeddings.copy()

    def uncertainties(self) -> np.ndarray:
        """(M,) per-object scalar uncertainties u_j."""
        return np.array(self._uncert, dtype=np.float64)

    def get_embedding(self, object_id: str) -> Optional[np.ndarray]:
        if object_id not in self._ids:
            return None
        idx = self._ids.index(object_id)
        return self._embeddings[idx].copy()
