"""
Animal Sound MMDB — Flask Web Application
==========================================
Provides a beautiful web UI for:
  - Uploading a query audio file
  - Viewing top-5 most similar animal sounds
  - Displaying per-feature-group similarity breakdown
  - Streaming audio from the database
"""

import json
import os
import sys
import uuid
import sqlite3
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify,
    send_file, abort, url_for
)

# Make src/ importable
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from extract_features import extract_all, flatten
from retrieval import query_database, FEATURE_GROUPS

BASE_DIR = SRC_DIR.parent
UPLOAD_DIR = BASE_DIR / "query"
FEATURES_DB = BASE_DIR / "features_db"
DB_PATH = FEATURES_DB / "animal_sounds.db"

UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB


# ── Helpers ───────────────────────────────────────────────────────────────────

def db_stats() -> dict:
    """Return basic database statistics."""
    if not DB_PATH.exists():
        return {"total": 0, "species": []}
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM audio_features").fetchone()[0]
    species_rows = conn.execute(
        "SELECT species, COUNT(*) as cnt FROM audio_features GROUP BY species ORDER BY cnt DESC"
    ).fetchall()
    conn.close()
    species = [{"name": r[0], "count": r[1]} for r in species_rows]
    return {"total": total, "species": species}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    stats = db_stats()
    return render_template("index.html", stats=stats, groups=FEATURE_GROUPS)


@app.route("/api/retrieve", methods=["POST"])
def api_retrieve():
    """
    POST /api/retrieve
    Form-data: file=<audio_file>
    Returns JSON with top-5 results.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save upload with unique name
    ext = Path(f.filename).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    upload_path = UPLOAD_DIR / unique_name
    f.save(upload_path)

    try:
        features = extract_all(upload_path)
        if features is None:
            return jsonify({"error": "Could not extract features from audio file"}), 422

        vectors = flatten(features)
        results = query_database(vectors, top_k=5)

        # Convert results for JSON serialization
        serializable = []
        for r in results:
            serializable.append({
                "rank": results.index(r) + 1,
                "id": r["id"],
                "filename": r["filename"],
                "species": r["species"],
                "similarity": round(r["similarity"], 6),
                "similarity_pct": round(r["similarity"] * 100, 2),
                "per_group": {k: round(v, 6) for k, v in r["per_group_similarity"].items()},
                "audio_url": url_for("stream_audio", record_id=r["id"]),
            })

        # Prepare intermediate feature data for display
        intermediate = {}
        for group, vals in features.items():
            intermediate[group] = {}
            for key, val in vals.items():
                if isinstance(val, list):
                    intermediate[group][key] = [round(x, 4) for x in val[:5]]  # truncate for display
                else:
                    intermediate[group][key] = round(float(val), 6)

        return jsonify({
            "query_filename": f.filename,
            "results": serializable,
            "query_features": intermediate,
        })

    finally:
        # Clean up temp upload after processing
        try:
            upload_path.unlink()
        except Exception:
            pass


@app.route("/api/audio/<int:record_id>")
def stream_audio(record_id: int):
    """Stream an audio file from the database by its record ID."""
    if not DB_PATH.exists():
        abort(404)

    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT filepath FROM audio_features WHERE id = ?", (record_id,)
    ).fetchone()
    conn.close()

    if row is None:
        abort(404)

    stored = row[0]
    # Support both relative paths (new, portable) and absolute paths (legacy)
    filepath = Path(stored) if Path(stored).is_absolute() else BASE_DIR / stored
    if not filepath.exists():
        abort(404)

    return send_file(filepath, mimetype="audio/wav", as_attachment=False)


@app.route("/api/stats")
def api_stats():
    return jsonify(db_stats())


@app.route("/api/features/<int:record_id>")
def api_features(record_id: int):
    """Return full feature JSON for a DB record."""
    if not DB_PATH.exists():
        abort(404)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM audio_features WHERE id = ?", (record_id,)
    ).fetchone()
    conn.close()
    if row is None:
        abort(404)

    data = dict(row)
    # Parse JSON fields
    for g in FEATURE_GROUPS:
        data[g] = json.loads(data[g] or "{}")
        data[f"vec_{g}"] = json.loads(data[f"vec_{g}"] or "[]")
    return jsonify(data)


if __name__ == "__main__":
    print("[APP] Starting Animal Sound MMDB Web Application...")
    if not DB_PATH.exists():
        print("[WARN] Database not found. Run extract_features.py first.")
    app.run(debug=True, host="0.0.0.0", port=5000)
