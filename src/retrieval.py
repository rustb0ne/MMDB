"""
Animal Sound Retrieval Engine
==============================
Given a query audio file, extracts its features and returns
the top-K most similar files from the database using
per-feature-group Cosine Similarity averaged across all 10 groups.

Similarity formula (matching the reference system):
  sim(q, d) = mean( cosine_sim(q_group_i, d_group_i) ) for i in 1..10
"""

import json
import sqlite3
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DB = BASE_DIR / "features_db"
DB_PATH = FEATURES_DB / "animal_sounds.db"

FEATURE_GROUPS = [
    "frequency", "amplitude", "temporal", "spectral",
    "waveform", "complexity", "timbre", "brightness",
    "attack", "decay",
]


# ── Cosine Similarity ──────────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Handles the case where vectors have different lengths by padding with zeros.
    Returns a value in [-1, 1].
    """
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)

    # Pad shorter vector
    if len(va) < len(vb):
        va = np.pad(va, (0, len(vb) - len(va)))
    elif len(vb) < len(va):
        vb = np.pad(vb, (0, len(va) - len(vb)))

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    return float(np.dot(va, vb) / (norm_a * norm_b))


# ── Query Processor ────────────────────────────────────────────────────────────

def query_database(query_features_vectors: dict[str, list[float]], top_k: int = 5) -> list[dict]:
    """
    Compare query feature vectors against all records in the SQLite DB.
    Returns top_k most similar records sorted by descending similarity.

    Args:
        query_features_vectors: {group_name: [float, ...]} from flatten_features()
        top_k: number of results to return

    Returns:
        List of result dicts with keys: id, filename, species, filepath,
        similarity, per_group_similarity
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    cols = ", ".join(
        ["id", "filename", "species", "filepath"]
        + [f"vec_{g}" for g in FEATURE_GROUPS]
    )
    rows = conn.execute(f"SELECT {cols} FROM audio_features").fetchall()
    conn.close()

    results = []

    for row in rows:
        per_group: dict[str, float] = {}

        for group in FEATURE_GROUPS:
            db_vec = json.loads(row[f"vec_{group}"] or "[]")
            q_vec = query_features_vectors.get(group, [])

            if not db_vec or not q_vec:
                per_group[group] = 0.0
            else:
                per_group[group] = cosine_similarity(q_vec, db_vec)

        avg_sim = float(np.mean(list(per_group.values())))

        results.append({
            "id": row["id"],
            "filename": row["filename"],
            "species": row["species"],
            "filepath": row["filepath"],
            "similarity": avg_sim,
            "per_group_similarity": per_group,
        })

    # Sort descending by average similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


# ── High-Level API ─────────────────────────────────────────────────────────────

def retrieve_similar(query_path: str | Path, top_k: int = 5) -> list[dict]:
    """
    Full retrieval pipeline:
      1. Extract features from query audio
      2. Query the database
      3. Return top-k results

    Args:
        query_path: Path to the query audio file
        top_k: Number of results (default 5)

    Returns:
        Ranked list of similar audio records.
    """
    # Import here to avoid circular imports
    from extract_features import extract_all_features, flatten_features

    print(f"[RETRIEVAL] Extracting features from: {query_path}")
    features = extract_all_features(query_path)
    if features is None:
        raise ValueError(f"Could not extract features from: {query_path}")

    vectors = flatten_features(features)
    print(f"[RETRIEVAL] Querying database for top-{top_k} similar sounds...")
    results = query_database(vectors, top_k=top_k)

    return results, features, vectors


def print_results(results: list[dict], query_path: str):
    """Pretty-print retrieval results."""
    print("\n" + "=" * 70)
    print(f"  Query: {Path(query_path).name}")
    print(f"  Top-{len(results)} Most Similar Animal Sounds")
    print("=" * 70)

    for rank, r in enumerate(results, start=1):
        print(f"\n  Rank #{rank}")
        print(f"    File     : {r['filename']}")
        print(f"    Species  : {r['species']}")
        print(f"    DB ID    : {r['id']}")
        print(f"    Similarity: {r['similarity']:.4f} ({r['similarity']*100:.2f}%)")
        print("    Per-group similarity:")
        for g, s in r["per_group_similarity"].items():
            bar = "█" * int(s * 20)
            print(f"      {g:<12} {s:.4f}  {bar}")

    print("=" * 70)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python retrieval.py <path_to_query_audio> [top_k]")
        print("Example: python retrieval.py ../query/cat.wav 5")
        sys.exit(1)

    query_file = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    results, features, vectors = retrieve_similar(query_file, top_k=k)
    print_results(results, query_file)
