"""
Retrieval Engine — Animal Sound MMDB v2
========================================
Cosine Similarity trên từng nhóm đặc trưng → trung bình → Top-K.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np

BASE_DIR    = Path(__file__).resolve().parent.parent
DB_PATH     = BASE_DIR / "features_db" / "animal_sounds.db"

FEATURE_GROUPS = [
    "mfcc","centroid","bandwidth","rolloff","contrast",
    "zcr","rms","chroma","attack","decay",
]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    if len(va) < len(vb):
        va = np.pad(va, (0, len(vb) - len(va)))
    elif len(vb) < len(va):
        vb = np.pad(vb, (0, len(va) - len(vb)))
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def query_database(query_vectors: dict[str, list[float]],
                   top_k: int = 5) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cols = ["id","filename","species","filepath"] + [f"vec_{g}" for g in FEATURE_GROUPS]
    rows = conn.execute(f"SELECT {','.join(cols)} FROM audio_features").fetchall()
    conn.close()

    results = []
    for row in rows:
        per_group: dict[str, float] = {}
        for g in FEATURE_GROUPS:
            db_vec = json.loads(row[f"vec_{g}"] or "[]")
            q_vec  = query_vectors.get(g, [])
            per_group[g] = cosine_similarity(q_vec, db_vec) if db_vec and q_vec else 0.0
        avg = float(np.mean(list(per_group.values())))
        results.append({
            "id": row["id"], "filename": row["filename"],
            "species": row["species"], "filepath": row["filepath"],
            "similarity": avg, "per_group_similarity": per_group,
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def retrieve_similar(query_path: str | Path, top_k: int = 5):
    from extract_features import extract_all, flatten
    features = extract_all(query_path)
    if features is None:
        raise ValueError(f"Could not extract features from: {query_path}")
    vectors  = flatten(features)
    results  = query_database(vectors, top_k=top_k)
    return results, features, vectors


def print_results(results: list[dict], query_path: str):
    print("\n" + "=" * 65)
    print(f"  Query : {Path(query_path).name}")
    print(f"  Top-{len(results)} results")
    print("=" * 65)
    for r in results:
        print(f"\n  #{r['similarity']*100:5.2f}%  [{r['species']}]  {r['filename']}")
        for g, s in r["per_group_similarity"].items():
            bar = "█" * int(s * 20)
            print(f"    {g:<12} {s:.4f}  {bar}")
    print("=" * 65)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python retrieval.py <audio_file> [top_k]")
        sys.exit(1)
    k   = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    res, _, _ = retrieve_similar(sys.argv[1], top_k=k)
    print_results(res, sys.argv[1])
