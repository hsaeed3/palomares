#!/usr/bin/env python3
"""Step S: semantic segmentation of GoPro frames (analysis only, nothing ships).
Mask2Former (Mapillary Vistas) -> shared class list (docs/phase2/DEV5_SCENE_LEDGER.md),
then simple colour refinements. Resumable: skips frames whose mask exists (--force to redo).
Usage: nice -n 10 python3 scripts/p5/segment.py [--frames 1-219] [--model ID] [--force] [--refine-only]
"""
import argparse, glob, json, os, re, sys, time
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "assets/gopro")
OUT = os.path.join(ROOT, "data/scene/seg")
RAW = os.path.join(ROOT, "data/scene/seg_raw")   # raw model label ids (Vistas), for re-refinement
W, H = 1021, 743
CLASSES = ["unknown","sky","road","marking_yellow","marking_white","shoulder_dirt","terrain_grass",
 "terrain_bare","rock_cut","vegetation_tree","shrub","trunk","sign_face","sign_back","pole",
 "guardrail","fence","wall","building","bridge","vehicle","person","water","mailbox_prop","other_prop"]
CID = {n: i for i, n in enumerate(CLASSES)}
DEFAULT_MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"

# Vistas label name (lowercased, from model config) -> shared class
def map_label(name):
    n = name.lower()
    rules = [
        ("sky","sky"), ("lane marking","marking_white"), ("crosswalk","marking_white"),
        ("marking","marking_white"), ("road","road"), ("service lane","road"), ("bike lane","road"),
        ("parking","road"), ("pothole","road"), ("manhole","road"), ("catch basin","road"),
        ("curb","shoulder_dirt"), ("sidewalk","shoulder_dirt"), ("pedestrian area","shoulder_dirt"),
        ("rail track","road"), ("sand","terrain_bare"), ("terrain","terrain_grass"),
        ("mountain","rock_cut"), ("snow","terrain_bare"), ("vegetation","vegetation_tree"),
        ("water","water"), ("traffic sign (back)","sign_back"), ("traffic sign (front)","sign_face"),
        ("traffic sign","sign_face"), ("traffic light","other_prop"), ("pole","pole"),
        ("street light","pole"), ("utility pole","pole"), ("guard rail","guardrail"),
        ("barrier","guardrail"), ("fence","fence"), ("wall","wall"), ("building","building"),
        ("bridge","bridge"), ("tunnel","bridge"), ("mailbox","mailbox_prop"),
        ("person","person"), ("rider","person"), ("bicyclist","person"), ("motorcyclist","person"),
        ("car","vehicle"), ("truck","vehicle"), ("bus","vehicle"), ("motorcycle","vehicle"),
        ("bicycle","vehicle"), ("trailer","vehicle"), ("caravan","vehicle"), ("vehicle","vehicle"),
        ("boat","vehicle"), ("ego vehicle","unknown"), ("car mount","unknown"), ("unlabeled","unknown"),
        ("bird","unknown"), ("animal","unknown"),
    ]
    for k, v in rules:
        if k in n: return v
    return "other_prop"

def refine(img, lab):
    """img: HxWx3 uint8 at mask res; lab: shared ids. Simple, visually-checked colour rules."""
    from scipy import ndimage as ndi
    hsv = np.asarray(Image.fromarray(img).convert("HSV")).astype(np.int16)
    h, s, v = hsv[..., 0] * 360 // 255, hsv[..., 1], hsv[..., 2]
    lab = lab.copy()
    # markings: yellow vs white by hue/saturation
    m = lab == CID["marking_white"]
    yel = m & (h >= 25) & (h <= 70) & (s > 60)
    lab[yel] = CID["marking_yellow"]
    # yellow paint the model left as road
    rd = lab == CID["road"]
    lab[rd & (h >= 30) & (h <= 60) & (s > 110) & (v > 120)] = CID["marking_yellow"]
    # wet/dark road stays road: model water on lower half of frame -> road
    yy = np.arange(H)[:, None] * np.ones((1, W), bool)
    lab[(lab == CID["water"]) & (yy > H * 0.45)] = CID["road"]
    # grass that is brown/low-sat -> shoulder dirt; terrain green stays grass
    g = lab == CID["terrain_grass"]
    brown = g & (((h >= 10) & (h < 50) & (s > 40)) | (s < 35)) & (v > 50)
    brown = ndi.binary_opening(brown, np.ones((5, 5)))
    lab[brown] = CID["shoulder_dirt"]
    # gravel shoulder the model calls road: road pixels outside the outermost white edge line
    # (only on rows/sides where an edge line is visible, so unmarked road is left alone)
    wh = lab == CID["marking_white"]; ye = lab == CID["marking_yellow"]; rd = lab == CID["road"]
    sh = np.zeros_like(rd)
    for y in range(int(H * 0.4), H):
        xs = np.flatnonzero(wh[y])
        if xs.size == 0: continue
        yx = np.flatnonzero(ye[y]); mid = yx.mean() if yx.size else W / 2
        l = xs[xs < mid]; r = xs[xs > mid]
        if l.size: sh[y, :l.min()] = True
        if r.size: sh[y, r.max() + 1:] = True
    sh = ndi.binary_opening(sh & rd, np.ones((9, 1)))   # drop 1-row streaks from stray paint pixels
    lab[sh] = CID["shoulder_dirt"]
    # trunks inside tree class: dark, low-sat, not green
    t = lab == CID["vegetation_tree"]
    tr = t & ~((h > 50) & (h < 170) & (s > 50)) & (v < 90) & (s < 90) & (yy > H * 0.3)
    tr = ndi.binary_opening(tr, np.ones((25, 3)))   # keep tall thin dark structures only
    lab[tr] = CID["trunk"]
    # GoPro overlays
    lab[int(H * 0.93):, int(W * 0.90):] = CID["unknown"]
    lab[:int(H * 0.08), int(W * 0.85):] = CID["unknown"]
    return lab

def parse_frames(spec, allf):
    if not spec: return allf
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-"); out.update(range(int(a), int(b or a) + 1))
    return [f for f in allf if f in out]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames"); ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--force", action="store_true"); ap.add_argument("--refine-only", action="store_true")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(RAW, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.png")))
    nums = [int(re.findall(r"\d+", os.path.basename(f))[0]) for f in files]
    todo = parse_frames(a.frames, nums)
    with open(os.path.join(ROOT, "data/scene/seg_classes.json"), "w") as fh:
        json.dump({str(i): n for i, n in enumerate(CLASSES)}, fh, indent=1)
    model = proc = lut = None
    for n in todo:
        stem = f"{n:03d}"; rawp = os.path.join(RAW, stem + ".png"); confp = os.path.join(OUT, stem + "_conf.png")
        img = Image.open(os.path.join(SRC, stem + ".png")).convert("RGB")
        small = np.asarray(img.resize((W, H), Image.BILINEAR))
        if a.refine_only or (os.path.exists(rawp) and not a.force):
            if not os.path.exists(rawp): continue
            raw = np.asarray(Image.open(rawp))
        else:
            if model is None:
                import torch
                from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
                torch.set_num_threads(a.threads)
                proc = AutoImageProcessor.from_pretrained(a.model)
                model = Mask2FormerForUniversalSegmentation.from_pretrained(a.model).eval()
                id2l = model.config.id2label
                lut = np.array([CID[map_label(id2l[i])] for i in range(len(id2l))], np.uint8)
                with open(os.path.join(ROOT, "data/scene/seg_model_labels.json"), "w") as fh:
                    json.dump({id2l[i]: CLASSES[lut[i]] for i in range(len(id2l))}, fh, indent=1)
            import torch
            t0 = time.time()
            inp = proc(images=Image.fromarray(small), do_resize=False, return_tensors="pt")
            with torch.no_grad(): o = model(**inp)
            cl = o.class_queries_logits.softmax(-1)[..., :-1]          # B,Q,C
            mk = o.masks_queries_logits.sigmoid()  # B,Q,h,w (padded to /32, crop after upsample)
            ph, pw = inp["pixel_values"].shape[-2:]
            mk = torch.nn.functional.interpolate(mk, size=(ph, pw), mode="bilinear", align_corners=False)[..., :H, :W]
            seg = torch.einsum("bqc,bqhw->bchw", cl, mk)[0]
            prob = seg / seg.sum(0, keepdim=True).clamp_min(1e-6)
            conf, raw = prob.max(0)
            raw = raw.numpy().astype(np.uint8)
            Image.fromarray((conf.numpy() * 255).astype(np.uint8)).save(confp)
            Image.fromarray(raw).save(rawp)
            print(f"{stem} {time.time()-t0:.1f}s", flush=True)
        if lut is None:
            mp = json.load(open(os.path.join(ROOT, "data/scene/seg_model_labels.json")))
            lut = np.array([CID[v] for v in mp.values()], np.uint8)
        lab = refine(small, lut[raw])
        Image.fromarray(lab).save(os.path.join(OUT, stem + ".png"))

if __name__ == "__main__":
    main()
