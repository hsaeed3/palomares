// Phase 2 visual check, step 2: render driver's-eye views from the round-tripped kn5 scene (headless Chromium).
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '../..');
const lay = process.argv[2] || 'road_summit';
const dir = path.join(root, 'build/vis', lay);
// optional: CAMS=<json in build/vis/<lay>> and OUT=<reports/visual subdir> (e.g. terrain views)
const outDir = path.join(root, 'reports/visual', 'decals'); fs.mkdirSync(outDir, { recursive: true });
const CAMS_ = process.env.CAMS || '';
const html = `<!doctype html><html><body style="margin:0;background:#000">
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"}}</script>
<script type="module">
import * as THREE from 'three';
const CAMS = '${CAMS_}';
const sc = await (await fetch('scene.json')).json(); sc.cams = (await (await fetch('cams_decals.json')).json()).cams;
if (CAMS) sc.cams = await (await fetch(CAMS)).json();
const bin = await (await fetch('scene.bin')).arrayBuffer();
const r = new THREE.WebGLRenderer({antialias:true}); r.setSize(1280,720); document.body.appendChild(r.domElement);
r.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene(); scene.background = new THREE.Color(0xa9c4dc); scene.fog = new THREE.Fog(0xb8c8d6, 300, 2500);
scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x4a4030, 1.3));
const sun = new THREE.DirectionalLight(0xfff2dd, 1.6); sun.position.set(-0.5, 0.8, 0.3); scene.add(sun);
const tl = new THREE.TextureLoader(); const texs = {};
const mats = sc.materials.map(m => { let t = null;
  if (m.tex) { t = texs[m.tex] || (texs[m.tex] = tl.load(m.tex)); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; }
  const mm = new THREE.MeshLambertMaterial({map: t, alphaTest: m.alpha ? 0.5 : 0, side: THREE.FrontSide, transparent: !!m.blend, depthWrite: !m.blend});
  if (m.det) { const d = texs[m.det] || (texs[m.det] = tl.load(m.det)); d.wrapS = d.wrapT = THREE.RepeatWrapping; d.colorSpace = THREE.SRGBColorSpace; d.anisotropy = 8;
    // ksPerPixelMultiMap: diffuse * detail(uv * detailUVMultiplier) where diffuse alpha = 0 (terrain macro layer);
    // terrain is opaque: force alpha 1, else the alpha-0 macro map writes a=0 into the premultiplied canvas (washed-out white)
    mm.onBeforeCompile = sh => { sh.uniforms.tDet = {value: d}; sh.uniforms.dMul = {value: m.dmul};
      sh.fragmentShader = 'uniform sampler2D tDet; uniform float dMul;\n' + sh.fragmentShader.replace('#include <map_fragment>',
        '#include <map_fragment>\n diffuseColor.rgb *= mix(texture2D(tDet, vMapUv * dMul).rgb, vec3(1.0), texture2D(map, vMapUv).a);\n diffuseColor.a = 1.0;'); };
    mm.customProgramCacheKey = () => 'det' + m.dmul; }
  return mm; });
const near = (m) => { const p = new Float32Array(bin, m.off, m.nv * 3); let x0=1e9,x1=-1e9,z0=1e9,z1=-1e9;
  for (let i = 0; i < p.length; i += 3) { if (p[i]<x0) x0=p[i]; if (p[i]>x1) x1=p[i]; if (p[i+2]<z0) z0=p[i+2]; if (p[i+2]>z1) z1=p[i+2]; }
  return sc.cams.some(c => Math.max(x0 - c.eye[0], 0, c.eye[0] - x1) ** 2 + Math.max(z0 - c.eye[2], 0, c.eye[2] - z1) ** 2 < 220 * 220); };  // only meshes within 220 m of a camera (faster page load)
for (const m of sc.meshes) { if (!near(m)) continue; const g = new THREE.BufferGeometry(); let o = m.off;
  const f = (n, k) => { const a = new Float32Array(bin, o, n * k); o += n * k * 4; return a; };
  g.setAttribute('position', new THREE.BufferAttribute(f(m.nv, 3), 3));
  g.setAttribute('normal', new THREE.BufferAttribute(f(m.nv, 3), 3));
  const uv = f(m.nv, 2).slice(); for (let i = 1; i < uv.length; i += 2) uv[i] = -uv[i]; // kn5 stores v flipped
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(new THREE.BufferAttribute(new Uint32Array(bin, o, m.ni), 1));
  // AC coords are a pure rotation of local ENU, so CCW-from-above stays front-facing in three.js
  scene.add(new THREE.Mesh(g, mats[m.mat])); }
const cam = new THREE.PerspectiveCamera(56, 1280/720, 0.3, 6000);
window.shot = i => { const c = sc.cams[i]; cam.position.set(...c.eye); cam.lookAt(...c.look); r.render(scene, cam); return c.s; };
window.ncams = sc.cams.length;
await Promise.all(Object.values(texs).map(t => new Promise(res => { const iv = setInterval(() => { if (t.image && t.image.width) { clearInterval(iv); res(); } }, 100); })));
window.__ready = true;
</script></body></html>`;
fs.writeFileSync(path.join(dir, 'index_decals.html'), html);
const srv = http.createServer((q, s) => { const p = path.join(dir, decodeURIComponent(q.url.split('?')[0]) === '/' ? 'index_decals.html' : decodeURIComponent(q.url.split('?')[0]));
  fs.readFile(p, (e, d) => { if (e) { s.writeHead(404); s.end(); } else { s.writeHead(200, {'Content-Type': p.endsWith('.html') ? 'text/html' : p.endsWith('.json') ? 'application/json' : p.endsWith('.png') ? 'image/png' : 'application/octet-stream'}); s.end(d); } }); }).listen(8871);
(async () => {
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 1280, height: 720 } });
  p.on('console', m => console.log('console:', m.type(), m.text().slice(0, 200))); p.on('pageerror', e => console.log('pageerror:', String(e).slice(0, 300)));
  await p.goto('http://localhost:8871/');
  await p.waitForFunction(() => window.__ready === true, null, { timeout: 150000 });
  const n = await p.evaluate(() => window.ncams);
  for (let i = 0; i < n; i++) {
    const s = await p.evaluate(i => window.shot(i), i);
    await p.screenshot({ path: path.join(outDir, (typeof s === 'string' ? s : `s${String(s).padStart(5, '0')}`) + '.jpg'), type: 'jpeg', quality: 82, timeout: 300000 });
    console.log('shot', lay, s);
  }
  await b.close(); srv.close();
})();
