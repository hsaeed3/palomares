"""Phase 2: fetch CC0 textures from ambientCG (1K JPG) into data/raw/textures/<id>/. Idempotent."""
import io, zipfile, pathlib, requests
ROOT = pathlib.Path(__file__).resolve().parents[2]
IDS = {"asphalt": "Asphalt026C", "rock": "Rock030", "grass": "Grass004", "litter": "Ground048", "bank": "Ground068",
       "bark_oak": "Bark007", "bark_pale": "Bark012", "bark_grey": "Bark014"}
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

# dev6: CC0 photographic leaf / twig sprite sheets from Poly Haven 3D plant models (polyhaven.com, CC0). Only the
# 2K leaf colour + alpha maps are used: scripts/p2/foliage_photo.py composes them into leaf-cluster cards.
PH = {"searsia_lucida": ("Diffuse", "Alpha"),            # small rounded leathery leaves -> coast live oak
      "island_tree_02": ("leaves_diff", "leaves_alpha"),  # lanceolate glossy leaves -> California bay / (narrowed) euc
      "tree_small_02": ("leaves_diff", "leaves_alpha"),   # broad soft leaves -> sycamore / alder new growth
      "fir_sapling": ("twigs_diff", "twigs_alpha")}       # needle twigs -> cypress / pine
for aid, maps in PH.items():
    d = ROOT / "data/raw/textures/polyhaven" / aid
    if all((d / f"{m}.png").exists() for m in maps):
        continue
    d.mkdir(parents=True, exist_ok=True)
    files = requests.get(f"https://api.polyhaven.com/files/{aid}", timeout=60).json()
    for m in maps:
        u = files[m]["2k"]["png"]["url"]
        r = requests.get(u, timeout=300); r.raise_for_status(); (d / f"{m}.png").write_bytes(r.content)
    print("fetched polyhaven", aid, maps)
