"""Phase 2: fetch CC0 textures from ambientCG (1K JPG) into data/raw/textures/<id>/. Idempotent."""
import io, zipfile, pathlib, requests
ROOT = pathlib.Path(__file__).resolve().parents[2]
IDS = {"asphalt": "Asphalt026C", "ground": "Ground037", "rock": "Rock030", "dirt": "Ground054"}
for key, aid in IDS.items():
    d = ROOT / "data/raw/textures" / aid
    if d.exists() and any(d.glob("*Color.jpg")):
        continue
    d.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"https://ambientcg.com/get?file={aid}_1K-JPG.zip", timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for n in z.namelist():
            if n.endswith(".jpg") and "/" not in n and ".." not in n:
                (d / n).write_bytes(z.read(n))
    print("fetched", aid, sorted(p.name for p in d.iterdir()))
