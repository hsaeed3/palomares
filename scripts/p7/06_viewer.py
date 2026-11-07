#!/usr/bin/env python3
"""dev7 D-06: standalone three.js viewer of the dev7 data -> reports/dev7/viewer.html (single file, data inline).
Terrain: 1 m LiDAR DTM resampled to 4 m inside the 150 m area, coloured by hillshade + canopy (CHM) tint.
Roads: ribbons from every refined centerline (measured width + crossfall). Trees: CHM tree tops (h, crown r).
Features: coloured markers (rail underpass, bridges, signals/stops, gaps, guardrail candidates, walls).
Distance markers every 500 m on Niles Canyon Rd and 100 m on Niles Blvd.  Coordinates = fixed local origin."""
import base64, glob, json
import numpy as np, rasterio
from scipy.ndimage import zoom
from common7 import *

b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
R = rasterio.open(DER7 / "dtm_r100_b150.tif"); dtm = R.read(1); tf = R.transform
chm = rasterio.open(DER7 / "chm_r100_b150.tif").read(1)
ST = 5
dz = dtm[::ST, ::ST]; cz = np.nan_to_num(chm[::ST, ::ST], nan=0)
H, W = dz.shape
x0 = tf.c - X0 + ST / 2; y0 = tf.f - Y0 - ST / 2
gy, gx = np.gradient(np.nan_to_num(dz, nan=np.nanmean(dz)), ST)
shade = np.clip(0.55 + 0.45 * (-gx * 0.6 + gy * 0.6) / np.sqrt(1 + gx ** 2 + gy ** 2), 0.15, 1)
terrain = dict(w=W, h=H, x0=x0, y0=y0, st=ST, z=b64(np.nan_to_num(dz, nan=-9999).astype(np.float32)),
               shade=b64((shade * 255).astype(np.uint8)), canopy=b64(np.clip(cz * 10, 0, 255).astype(np.uint8)))
roads = []
for f in sorted(glob.glob(str(DER7 / "centerline_*.csv"))):
    r = f.split("centerline_")[1][:-4]
    a = np.genfromtxt(f, delimiter=",", names=True, dtype=None, encoding=None)
    k = np.arange(0, len(a), 2)
    roads.append(dict(name=r, x=b64(a["x_local"][k].astype(np.float32)), y=b64(a["y_local"][k].astype(np.float32)),
                      z=b64(a["z_navd88"][k].astype(np.float32)), wl=b64(a["edge_left_m"][k].astype(np.float32)), wr=b64(a["edge_right_m"][k].astype(np.float32)),
                      cl=b64(a["crossfall_left"][k].astype(np.float32)), cr=b64(a["crossfall_right"][k].astype(np.float32)), n=int(len(k)),
                      len=float(a["s_m"][-1])))
marks = []
for r, step in (("niles_canyon_road", 500), ("niles_blvd", 100)):
    a = np.genfromtxt(DER7 / f"centerline_{r}.csv", delimiter=",", names=True, dtype=None, encoding=None)
    for s in range(0, int(a["s_m"][-1]) + 1, step):
        j = int(np.argmin(np.abs(a["s_m"] - s)))
        marks.append([float(a["x_local"][j]), float(a["y_local"][j]), float(a["z_navd88"][j]), f"{'NCR' if r == 'niles_canyon_road' else 'NB'} {s/1000:.1f} km"])
t = np.genfromtxt(DER7 / "trees.csv", delimiter=",", names=True, dtype=None, encoding=None)
trees = dict(n=len(t), x=b64(t["x_local"].astype(np.float32)), y=b64(t["y_local"].astype(np.float32)), z=b64(t["ground_z"].astype(np.float32)),
             h=b64(t["height_m"].astype(np.float32)), r=b64(t["crown_r_m"].astype(np.float32)))
COL = {"rail_underpass": "#ff2a2a", "bridge_osm": "#2ad4ff", "bridge_lidar_class17": "#2a7fff", "traffic_signals": "#00ff66", "stop": "#ff8800",
       "lidar_gap": "#ff00ff", "guardrail_candidate": "#ffff00", "wall_or_steep_cut": "#a0522d", "culvert_candidate": "#55aaaa", "rail_road_crossing": "#ff5555",
       "level_crossing": "#ff5555", "canyon_high_point": "#ffffff", "canyon_low_point": "#cccccc", "junction": "#888888"}
feats = []
for e in json.load(open(DER7 / "features.geojson"))["features"]:
    p = e["properties"]
    if p["kind"] not in COL: continue
    x, y = ll2loc(*e["geometry"]["coordinates"][:2])
    feats.append([round(float(x), 1), round(float(y), 1), COL[p["kind"]], p["kind"] + " | " + str(p.get("road")) + " s=" + str(p.get("s_m", p.get("s_start_m")))])
rail = []
for e in json.load(open(DER7 / "osm_features.geojson"))["features"]:
    p = e["properties"]
    if p["kind"].startswith("rail_") and e["geometry"]["type"] == "LineString":
        c = np.array(e["geometry"]["coordinates"]); x, y = ll2loc(c[:, 0], c[:, 1])
        rail.append(np.round(np.c_[x, y], 1).tolist())
DATA = dict(terrain=terrain, roads=roads, marks=marks, trees=trees, feats=feats, rail=rail)
html = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>dev7 Niles corridor viewer</title>
<style>:root{--bg:#10141a;--fg:#e8eef4;--panel:rgba(16,20,26,.86)}body{margin:0;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;overflow:hidden}
#ui{position:fixed;left:12px;top:12px;max-width:min(340px,calc(100vw - 32px));background:var(--panel);padding:10px 12px;border-radius:8px}
h1{font-size:15px;margin:0 0 6px}button{background:#24303c;color:var(--fg);border:1px solid #3a4a5a;border-radius:5px;padding:4px 8px;margin:2px;cursor:pointer}
#tip{position:fixed;pointer-events:none;background:var(--panel);padding:4px 6px;border-radius:4px;display:none}.lg span{display:inline-block;width:10px;height:10px;margin:0 4px 0 8px;border-radius:2px}
canvas{display:block}</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body><canvas id="c"></canvas><div id="ui"><h1>dev7: Niles town + Niles Canyon Rd (LiDAR 2021)</h1>
<div>Drag to orbit, right-drag to pan, wheel to zoom. Z exaggeration:<input id="ex" type="range" min="1" max="3" step="0.25" value="1"></div>
<div><button data-go="town">Niles town</button><button data-go="ncr">Canyon</button><button data-go="sunol">Sunol</button><button data-go="pal">Palomares jct</button><button id="bt">Trees</button></div>
<div class="lg"><span style="background:#ff2a2a"></span>rail underpass<span style="background:#2ad4ff"></span>bridge<span style="background:#00ff66"></span>signal<span style="background:#ff8800"></span>stop<br><span style="background:#ff00ff"></span>LiDAR gap<span style="background:#ffff00"></span>guardrail cand.<span style="background:#a0522d"></span>wall/cut<span style="background:#ffaa55"></span>rail line</div>
<div style="opacity:.7;margin-top:4px">Data: USGS 3DEP 2021 (public domain), OpenStreetMap (ODbL). Local origin config/origin.json.</div></div><div id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script type="module">
import * as THREE from 'three'; import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D=JSON.parse(document.getElementById('data').textContent);
const f32=s=>{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer)};
const u8=s=>{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u};
const R=new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true});R.setPixelRatio(Math.min(devicePixelRatio,2));
const S=new THREE.Scene();S.background=new THREE.Color(0x9fb8cc);const C=new THREE.PerspectiveCamera(55,1,1,40000);
const O=new OrbitControls(C,R.domElement);S.add(new THREE.HemisphereLight(0xffffff,0x445533,1.0));const sun=new THREE.DirectionalLight(0xffffff,1.2);sun.position.set(-1,2,1);S.add(sun);
const root=new THREE.Group();S.add(root);
// world: x=east, y=up, z=-north
const T=D.terrain,tz=f32(T.z),sh=u8(T.shade),cp=u8(T.canopy);
{const pos=[],col=[],idx=[];const id=new Int32Array(T.w*T.h).fill(-1);let n=0;
for(let r=0;r<T.h;r++)for(let c=0;c<T.w;c++){const k=r*T.w+c;if(tz[k]<-9000)continue;id[k]=n++;const x=T.x0+c*T.st,y=T.y0-r*T.st;pos.push(x,tz[k],-y);
 const s=sh[k]/255,v=cp[k]/255;col.push((0.62-0.35*v)*s,(0.58-0.12*v)*s,(0.40-0.2*v)*s);}
for(let r=0;r<T.h-1;r++)for(let c=0;c<T.w-1;c++){const a=id[r*T.w+c],b=id[r*T.w+c+1],d=id[(r+1)*T.w+c],e=id[(r+1)*T.w+c+1];
 if(a>=0&&b>=0&&d>=0)idx.push(a,d,b);if(b>=0&&d>=0&&e>=0)idx.push(b,d,e);}
const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));g.setAttribute('color',new THREE.Float32BufferAttribute(col,3));g.setIndex(idx);g.computeVertexNormals();
root.add(new THREE.Mesh(g,new THREE.MeshLambertMaterial({vertexColors:true})));}
for(const rd of D.roads){const x=f32(rd.x),y=f32(rd.y),z=f32(rd.z),wl=f32(rd.wl),wr=f32(rd.wr),cl=f32(rd.cl),cr=f32(rd.cr),n=rd.n;const pos=[],idx=[];
 for(let i=0;i<n;i++){const j=Math.min(i+1,n-1),k=Math.max(i-1,0);let tx=x[j]-x[k],ty=y[j]-y[k];const L=Math.hypot(tx,ty)||1;tx/=L;ty/=L;const nx=-ty,ny=tx;
  pos.push(x[i]+nx*wl[i],z[i]+cl[i]*wl[i]+0.08,-(y[i]+ny*wl[i]), x[i]-nx*wr[i],z[i]+cr[i]*-wr[i]+0.08,-(y[i]-ny*wr[i]));
  if(i<n-1)idx.push(2*i,2*i+1,2*i+2,2*i+1,2*i+3,2*i+2);}
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));g.setIndex(idx);g.computeVertexNormals();
 const m=new THREE.Mesh(g,new THREE.MeshLambertMaterial({color:rd.name==='niles_canyon_road'?0x3a3a40:(rd.name==='niles_blvd'?0x4a3a50:0x505055),side:THREE.DoubleSide}));m.userData.tip=rd.name+' ('+Math.round(rd.len)+' m)';root.add(m);}
{const t=D.trees,x=f32(t.x),y=f32(t.y),z=f32(t.z),h=f32(t.h),r=f32(t.r);const geo=new THREE.ConeGeometry(1,1,6);geo.translate(0,0.5,0);
 const im=new THREE.InstancedMesh(geo,new THREE.MeshLambertMaterial({color:0x2f5a2a}),t.n);const M=new THREE.Matrix4();
 for(let i=0;i<t.n;i++){M.makeScale(r[i],h[i],r[i]);M.setPosition(x[i],z[i],-y[i]);im.setMatrixAt(i,M);}root.add(im);window.TREES=im;}
const zAt=(x,y)=>{const c=Math.round((x-T.x0)/T.st),r=Math.round((T.y0-y)/T.st);if(c<0||r<0||c>=T.w||r>=T.h)return 0;const v=tz[r*T.w+c];return v<-9000?60:v;};
const pickables=[];
for(const f of D.feats){const m=new THREE.Mesh(new THREE.SphereGeometry(2.2,10,8),new THREE.MeshBasicMaterial({color:f[2]}));m.position.set(f[0],zAt(f[0],f[1])+4,-f[1]);m.userData.tip=f[3];root.add(m);pickables.push(m);}
for(const L of D.rail){const pts=L.map(p=>new THREE.Vector3(p[0],zAt(p[0],p[1])+0.6,-p[1]));root.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0xffaa55})));}
function label(t){const cv=document.createElement('canvas');cv.width=256;cv.height=64;const x=cv.getContext('2d');x.fillStyle='rgba(0,0,0,.6)';x.fillRect(0,0,256,64);x.fillStyle='#fff';x.font='bold 30px sans-serif';x.fillText(t,10,42);
 const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(cv),depthTest:false}));s.scale.set(40,10,1);return s;}
for(const m of D.marks){const s=label(m[3]);s.position.set(m[0],m[2]+14,-m[1]);root.add(s);}
const ex=document.getElementById('ex');ex.oninput=()=>root.scale.y=+ex.value;
document.getElementById('bt').onclick=()=>window.TREES.visible=!window.TREES.visible;
const go={town:[-1900,-6200],ncr:[1500,-5500],sunol:[7800,-6200],pal:[2126,-5326]};
function goto(k){const[x,y]=go[k];const z=zAt(x,y);O.target.set(x,z,-y);C.position.set(x-350,z+380,-y+450);O.update();}
document.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>goto(b.dataset.go));goto('town');
const ray=new THREE.Raycaster(),mv=new THREE.Vector2(),tip=document.getElementById('tip');
addEventListener('pointermove',e=>{mv.set(e.clientX/innerWidth*2-1,-e.clientY/innerHeight*2+1);ray.setFromCamera(mv,C);const h=ray.intersectObjects(pickables)[0];
 if(h){tip.style.display='block';tip.style.left=e.clientX+12+'px';tip.style.top=e.clientY+12+'px';tip.textContent=h.object.userData.tip;}else tip.style.display='none';});
function rs(){R.setSize(innerWidth,innerHeight);C.aspect=innerWidth/innerHeight;C.updateProjectionMatrix();}addEventListener('resize',rs);rs();
R.setAnimationLoop(()=>R.render(S,C));
</script></body></html>"""
# set camera presets from data
nb = np.genfromtxt(DER7 / "centerline_niles_blvd.csv", delimiter=",", names=True, dtype=None, encoding=None)
nc = np.genfromtxt(DER7 / "centerline_niles_canyon_road.csv", delimiter=",", names=True, dtype=None, encoding=None)
pre = {"town": [float(nb["x_local"][len(nb) // 2]), float(nb["y_local"][len(nb) // 2])],
       "ncr": [float(nc["x_local"][len(nc) // 2]), float(nc["y_local"][len(nc) // 2])],
       "sunol": [float(nc["x_local"][-1]), float(nc["y_local"][-1])]}
html = html.replace("const go={town:[-1900,-6200],ncr:[1500,-5500],sunol:[7800,-6200],", "const go={" + ",".join(f"{k}:[{v[0]:.0f},{v[1]:.0f}]" for k, v in pre.items()) + ",")
(REP7 / "viewer.html").write_text(html.replace("__DATA__", json.dumps(DATA, separators=(",", ":"))))
print("viewer", (REP7 / "viewer.html").stat().st_size // 1024, "KB; terrain", W, "x", H, "trees", len(t), "feats", len(feats))
