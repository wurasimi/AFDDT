"""Layer 4: Embedding model used for threat-intelligence matching (Section 3.6.3).

Two interchangeable back-ends:

* "tfidf" (default): a TF-IDF vector space fitted on the threat-intelligence
  corpus.  No download, runs instantly on any laptop.
* "st": a sentence-transformer model (default multi-qa-mpnet-base-dot-v1, the
  best model in Othman, Rimawi, Rossi and Russo, 2026).  Needs
  `pip install sentence-transformers` and downloads about 400 MB once.

Both return unit-length vectors, so cosine similarity (Equation 3.1) is a dot
product.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import SETTINGS


class Embedder:
    def __init__(self, backend: str | None = None):
        self.backend = backend or SETTINGS.embedder
        self._st = None
        self._tfidf = None
        if self.backend == "st":
            try:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(SETTINGS.st_model)
            except Exception as exc:
                print(f"[embedding] sentence-transformers unavailable ({exc}); falling back to TF-IDF")
                self.backend = "tfidf"
        self.name = SETTINGS.st_model if self.backend == "st" else "TF-IDF (word 1-2 grams)"

    def fit(self, texts: list[str]) -> None:
        if self.backend == "tfidf":
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english",
                                          min_df=1).fit(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.backend == "st":
            v = np.asarray(self._st.encode(texts, normalize_embeddings=True))
        else:
            v = self._tfidf.transform(texts).toarray()
            n = np.linalg.norm(v, axis=1, keepdims=True)
            v = v / np.where(n == 0, 1, n)
        return v
