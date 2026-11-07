"""Merge data/scene/ledger_parts/*.jsonl -> data/scene/ledger.jsonl (atomic). Run after editing a part.
dev7 rows (those with a `road` field: station on data/derived/dev7/centerline_<road>.csv) go to data/scene/ledger_dev7.jsonl
instead, so they never mix with the Palomares stations in ledger.jsonl."""
import json, os, pathlib
D = pathlib.Path(__file__).resolve().parents[2] / "data/scene"
rows = []
for f in sorted((D / "ledger_parts").glob("*.jsonl")):
    rows += [l.strip() for l in open(f) if l.strip()]
dev7 = sorted((r for r in rows if "road" in json.loads(r)), key=lambda l: (json.loads(l)["road"], json.loads(l)["s"]))
rows = sorted((r for r in rows if "road" not in json.loads(r)), key=lambda l: json.loads(l)["s"])
for name, rr in (("ledger.jsonl", rows), ("ledger_dev7.jsonl", dev7)):
    tmp = D / f"{name}.{os.getpid()}.tmp"
    tmp.write_text("".join(r + "\n" for r in rr)); os.replace(tmp, D / name)
    print(f"{name}: {len(rr)} rows")
