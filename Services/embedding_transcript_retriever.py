# alice/services/embedding_transcript_retriever.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from Domain.constants import MessageRole
from Domain.models import Message
from Infra.jsonl_transcript_reader import JSONLTranscriptReader
from Services.transcript_retriever import format_messages_for_context


@dataclass
class SklearnTfidfTranscriptRetriever:
    """
    Transcript retriever using TF-IDF embeddings + cosine similarity.

    Approach:
      - tail last `max_scan_messages` transcript messages
      - fit a TF-IDF vectorizer on those messages (small corpus => cheap)
      - embed query and compute cosine similarity to each message
      - pick top hits, then expand with +/- `context_window` neighbors
      - return chronological messages for ctx.retrieved_texts
    """

    reader: JSONLTranscriptReader
    max_scan_messages: int = 800
    top_k_hits: int = 6
    context_window: int = 1

    # Vectorizer config:
    # - analyzer="char" + ngram_range handles both English and Chinese without tokenizers
    # - You can tweak ngram_range to (3,5) if you want longer fragments.
    analyzer: str = "char"
    ngram_range: Tuple[int, int] = (2, 4)
    min_df: int = 1
    max_features: Optional[int] = 50000

    # Optional: small recency bump to prefer newer messages among ties
    recency_boost: float = 0.15

    def retrieve_messages(
        self,
        query: str,
        *,
        roles: Optional[Set[MessageRole]] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
    ) -> List[Message]:
        query = (query or "").strip()
        if not query:
            return []

        candidates = self.reader.tail_messages(
            n=self.max_scan_messages,
            roles=roles,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if not candidates:
            return []

        texts = [m.text or "" for m in candidates]
        if not any(t.strip() for t in texts):
            return []

        # Build TF-IDF embeddings (sparse)
        vec = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            lowercase=True,
        )

        X = vec.fit_transform(texts)            # shape: (N, D)
        q = vec.transform([query])              # shape: (1, D)

        # cosine similarity (dense array length N)
        sims = cosine_similarity(q, X).ravel()

        # If everything is zero similarity, return empty
        if float(np.max(sims)) <= 0.0:
            return []

        # Recency bump (later index = more recent)
        if self.recency_boost > 0 and len(candidates) > 1:
            idx = np.arange(len(candidates), dtype=np.float32)
            idx = idx / float(len(candidates) - 1)  # 0..1
            sims = sims + self.recency_boost * idx

        # Pick top hits by similarity
        top_k = max(1, int(self.top_k_hits))
        hit_idx = np.argsort(-sims)[:top_k].tolist()

        # Expand with neighbors
        picked = set()
        w = max(0, int(self.context_window))
        for i in hit_idx:
            # Ignore zero-ish matches (helps when top_k is large)
            if sims[i] <= 0.0:
                continue
            for j in range(i - w, i + w + 1):
                if 0 <= j < len(candidates):
                    picked.add(j)

        if not picked:
            return []

        return [candidates[i] for i in sorted(picked)]

    def retrieve_texts(
        self,
        query: str,
        *,
        roles: Optional[Set[MessageRole]] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
        max_chars: int = 1600,
    ) -> Tuple[str, ...]:
        msgs = self.retrieve_messages(query, roles=roles, start_ts=start_ts, end_ts=end_ts)
        return format_messages_for_context(msgs, max_chars=max_chars)
