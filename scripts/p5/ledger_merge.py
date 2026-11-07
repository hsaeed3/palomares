"""Merge data/scene/ledger_parts/*.jsonl -> data/scene/ledger.jsonl (atomic). Run after editing a part."""
import json, os, pathlib
D = pathlib.Path(__file__).resolve().parents[2] / "data/scene"
rows = []
for f in sorted((D / "ledger_parts").glob("*.jsonl")):
    rows += [l.strip() for l in open(f) if l.strip()]
rows.sort(key=lambda l: json.loads(l)["s"])
tmp = D / f"ledger.jsonl.{os.getpid()}.tmp"
tmp.write_text("".join(r + "\n" for r in rows)); os.replace(tmp, D / "ledger.jsonl")
print(f"ledger.jsonl: {len(rows)} rows")
