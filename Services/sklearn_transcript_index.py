# alice/services/sklearn_transcript_index.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from Domain.constants import MessageRole
from Domain.models import Message
from Infra.jsonl_transcript_reader import JSONLTranscriptReader
from Services.transcript_retriever import format_messages_for_context


@dataclass
class SklearnTranscriptIndex:
    """
    In-memory transcript index backed by sklearn TF-IDF embeddings.

    Key idea:
      - Cache {vectorizer, X, messages} in memory
      - Rebuild ONLY when transcript file changes (mtime/size)
      - Query is fast: transform(query) + cosine similarity

    Notes:
      - This is TF-IDF, not a neural semantic embedding, but it's surprisingly good
        for chat transcript retrieval, especially with analyzer="char" for Chinese.
    """

    reader: JSONLTranscriptReader

    # how many most recent messages to index
    max_index_messages: int = 1200

    # TF-IDF config (defaults work for mixed Chinese/English)
    analyzer: str = "char"                 # "char" handles Chinese w/o tokenizers
    ngram_range: Tuple[int, int] = (2, 4)  # tweak to (3,5) if you want “more semantic-ish”
    min_df: int = 1
    max_features: Optional[int] = 50000

    # retrieval behavior
    recency_boost: float = 0.15            # small bump for newer msgs to break ties
    min_similarity: float = 0.0            # set e.g. 0.05 to drop weak hits

    # internal cache
    _lock: RLock = RLock()
    _sig: Optional[Tuple[int, int]] = None          # (mtime_ns, size)
    _messages: List[Message] = None                 # type: ignore
    _vectorizer: Optional[TfidfVectorizer] = None
    _X = None                                       # scipy sparse matrix
    _built: bool = False

    # ----------------------------
    # Public
    # ----------------------------

    def ensure_built(self) -> None:
        """
        Build or rebuild the index if transcript file changed.
        Safe to call frequently.
        """
        with self._lock:
            sig = self._file_signature()
            if self._built and sig == self._sig:
                return
            self._rebuild(sig)

    def retrieve_messages(
        self,
        query: str,
        *,
        top_k_hits: int = 6,
        context_window: int = 1,
        roles: Optional[Set[MessageRole]] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
    ) -> List[Message]:
        """
        Retrieve relevant messages (chronological) from the indexed tail.
        """
        query = (query or "").strip()
        if not query:
            return []

        self.ensure_built()

        with self._lock:
            if not self._built or not self._messages or self._vectorizer is None or self._X is None:
                return []

            # Apply filters on-the-fly by masking indices (cheap for N ~ 1k)
            candidates = self._messages
            keep_idx: List[int] = []
            for i, m in enumerate(candidates):
                if roles is not None and m.role not in roles:
                    continue
                if start_ts is not None and m.ts < start_ts:
                    continue
                if end_ts is not None and m.ts > end_ts:
                    continue
                keep_idx.append(i)

            if not keep_idx:
                return []

            q = self._vectorizer.transform([query])   # (1, D)
            # cosine against whole X, then mask
            sims_all = cosine_similarity(q, self._X).ravel()

            # mask + recency bump (within the masked set)
            sims = sims_all[keep_idx].astype(np.float32, copy=False)

            if len(sims) > 1 and self.recency_boost > 0:
                idx = np.arange(len(sims), dtype=np.float32)
                idx = idx / float(len(sims) - 1)  # 0..1
                sims = sims + self.recency_boost * idx

            if float(np.max(sims)) <= self.min_similarity:
                return []

            k = max(1, int(top_k_hits))
            # top-k within masked list
            local_order = np.argsort(-sims)[:k].tolist()
            hit_local = [local_order[i] for i in range(len(local_order))]

            # expand with neighbors (in the *original* candidate list indices)
            picked: Set[int] = set()
            w = max(0, int(context_window))
            for li in hit_local:
                if sims[li] <= self.min_similarity:
                    continue
                orig_i = keep_idx[li]
                for j in range(orig_i - w, orig_i + w + 1):
                    if 0 <= j < len(candidates):
                        # still respect filters for neighbors? choose one:
                        # Here: neighbors can be any role/time; if you want strict, re-check filters.
                        picked.add(j)

            if not picked:
                return []

            return [candidates[i] for i in sorted(picked)]

    def retrieve_texts(
        self,
        query: str,
        *,
        top_k_hits: int = 6,
        context_window: int = 1,
        roles: Optional[Set[MessageRole]] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
        max_chars: int = 1600,
    ) -> Tuple[str, ...]:
        msgs = self.retrieve_messages(
            query,
            top_k_hits=top_k_hits,
            context_window=context_window,
            roles=roles,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        return format_messages_for_context(msgs, max_chars=max_chars)

    # ----------------------------
    # Internals
    # ----------------------------

    def _file_signature(self) -> Optional[Tuple[int, int]]:
        """
        Signature used to detect changes: (mtime_ns, size).
        If file missing, returns None.
        """
        p = Path(self.reader.path)
        if not p.exists():
            return None
        st = p.stat()
        return (int(st.st_mtime_ns), int(st.st_size))

    def _rebuild(self, sig: Optional[Tuple[int, int]]) -> None:
        """
        Rebuild index from the last max_index_messages.
        """
        messages = self.reader.tail_messages(n=self.max_index_messages)
        texts = [m.text or "" for m in messages]

        # build vectorizer + X
        if not messages or not any(t.strip() for t in texts):
            self._messages = []
            self._vectorizer = None
            self._X = None
            self._sig = sig
            self._built = True
            return

        vec = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            lowercase=True,
        )
        X = vec.fit_transform(texts)

        self._messages = messages
        self._vectorizer = vec
        self._X = X
        self._sig = sig
        self._built = True
