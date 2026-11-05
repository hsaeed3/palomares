"""Shared paths, CRS transforms and the fixed local origin."""
import json, os
from pathlib import Path
import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DER = ROOT / "data" / "derived"
CACHE = ROOT / "data" / "cache"
REPORTS = ROOT / "reports"
ASSETS = ROOT / "assets"
CONFIG = ROOT / "config"
for p in (RAW, DER, CACHE, REPORTS, ASSETS):
    p.mkdir(parents=True, exist_ok=True)

UTM = "EPSG:26910"
T_LL_UTM = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
T_UTM_LL = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
T_UTM_WM = Transformer.from_crs(UTM, "EPSG:3857", always_xy=True)
T_WM_UTM = Transformer.from_crs("EPSG:3857", UTM, always_xy=True)

EPT_BASE = "https://usgs-lidar-public.s3.amazonaws.com"
EPT_SETS = ["CA_AlamedaCo_1_2021", "CA_AlamedaCo_2_2021", "CA_AlamedaCo_3_2021"]

def origin():
    """Fixed local origin (UTM 10N), set once in 01_osm_centerline.py."""
    return json.load(open(CONFIG / "origin.json"))
