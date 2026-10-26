"""Line-addressable BM25, optional embedding retrieval, and reciprocal rank fusion."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

SKIP = {".git", ".venv", "venv", "node_modules", "__pycache__", ".papertrail",
        "outputs", ".pytest_cache", ".ruff_cache", "dist", "build"}
SUFFIXES = {".md", ".rst", ".txt", ".py", ".toml", ".yaml", ".yml", ".json"}


def tokens(text: str) -> list[str]:
    # Include Chinese characters so a Chinese query can match Chinese documentation.
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())


@dataclass(frozen=True)
class Chunk:
    source: str
    start: int
    end: int
    text: str
    digest: str
    truncated: bool = False

    @property
    def citation(self) -> str:
        return f"{self.source}:{self.start}-{self.end}"

    def record(self) -> dict:
        return {**asdict(self), "citation": self.citation}


class RepositoryIndex:
    def __init__(self, root: Path, *, embeddings=None, max_files: int = 200,
                 max_bytes: int = 8_000_000):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise ValueError(f"Repository directory does not exist: {root}")
        self.chunks: list[Chunk] = []
        self.files: list[str] = []
        self.warnings: list[str] = []
        total_bytes = 0
        # Walk without following symlinks or traversing dependency directories.
        import os
        for directory, dirs, names in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith(".")
                             and not (Path(directory) / d).is_symlink())
            for name in sorted(names):
                path = Path(directory) / name
                if (path.is_symlink() or (path.suffix.lower() not in SUFFIXES and name != "Dockerfile")
                        or name.startswith(".") or any(s in name.lower()
                        for s in ("secret", "credential", "token", "private", ".env"))):
                    continue
                size = path.stat().st_size
                if size > 250_000:
                    self.warnings.append(f"Skipped large file: {path.relative_to(self.root)}")
                    continue
                if len(self.files) >= max_files or total_bytes + size > max_bytes:
                    self.warnings.append("Index budget reached; repository coverage is incomplete.")
                    dirs.clear()
                    break
                try:
                    content = path.read_text(encoding="utf-8")
                except (UnicodeError, OSError):
                    continue
                if "\x00" in content:
                    continue
                rel = path.relative_to(self.root).as_posix()
                self.files.append(rel)
                total_bytes += size
                lines = content.splitlines()
                for offset in range(0, len(lines), 30):
                    piece = "\n".join(lines[offset:offset + 40])
                    self.chunks.append(Chunk(rel, offset + 1, min(offset + 40, len(lines)),
                                             piece[:8000], hashlib.sha256(piece.encode()).hexdigest(),
                                             len(piece) > 8000))
            if self.warnings and self.warnings[-1].startswith("Index budget"):
                break
        self.counts = [Counter(tokens(c.source + "\n" + c.text)) for c in self.chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / max(len(self.lengths), 1)
        self.df = Counter(t for count in self.counts for t in count)
        self.embeddings = embeddings
        self.vectors = embeddings.embed_documents([c.text for c in self.chunks]) \
            if embeddings and self.chunks else None

    @staticmethod
    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
        return dot / norm if norm else 0.0

    def search(self, query: str, k: int = 5, *, mode: str = "bm25") -> list[dict]:
        if not 1 <= k <= 10:
            raise ValueError("k must be between 1 and 10")
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("Unknown retrieval mode")
        if mode != "bm25" and self.vectors is None:
            raise ValueError("Semantic retrieval needs an embedding model")
        scores = []
        for i, count in enumerate(self.counts):
            score = 0.0
            for term in set(tokens(query)):
                tf = count[term]
                df = self.df[term]
                idf = math.log(1 + (len(self.chunks) - df + .5) / (df + .5))
                score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * self.lengths[i]
                                                         / max(self.average, 1)))
            scores.append(score)
        lexical = sorted((i for i, s in enumerate(scores) if s > 0),
                         key=lambda i: (-scores[i], i))
        ranked = lexical
        if mode != "bm25":
            query_vector = self.embeddings.embed_query(query)
            similarities = [self.cosine(v, query_vector) for v in self.vectors]
            dense = sorted(range(len(self.chunks)), key=lambda i: (-similarities[i], i))
            if mode == "dense":
                ranked = dense
                scores = similarities
            else:
                fused = Counter()
                for ordering in (lexical[:20], dense[:20]):
                    for rank, index in enumerate(ordering, 1):
                        fused[index] += 1 / (60 + rank)
                ranked = sorted(fused, key=lambda i: (-fused[i], i))
                scores = [fused[i] for i in range(len(self.chunks))]
        return [{**self.chunks[i].record(), "score": round(scores[i], 6)} for i in ranked[:k]]
