import sqlite3, json
conn = sqlite3.connect('features_db/animal_sounds.db')
conn.row_factory = sqlite3.Row

rows = conn.execute("""
    SELECT filename, species, frequency, amplitude, temporal, spectral,
           waveform, complexity, timbre, brightness, attack, decay
    FROM audio_features
    WHERE species IN ('cat','dog','frog','chirping_birds','cow')
    GROUP BY species LIMIT 5
""").fetchall()

for row in rows:
    sp = row["species"]
    fn = row["filename"]
    print("=== %s | %s ===" % (sp.upper(), fn))
    for col in ['frequency','amplitude','temporal','waveform','complexity','timbre','brightness','attack','decay']:
        d = json.loads(row[col])
        print("  %s: %s" % (col, d))
    spec = json.loads(row['spectral'])
    print("  spectral.centroid: %.2f Hz" % spec["centroid"])
    print("  spectral.bandwidth: %.2f Hz" % spec["bandwidth"])
    print("  spectral.rolloff: %.2f Hz" % spec["rolloff"])
    print("  spectral.mfcc[0..4]: %s" % [round(x,2) for x in spec["mfcc"][:5]])
    print()
conn.close()
