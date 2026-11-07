// Litmus test, step 2: render each GoPro pose (config/litmus_poses.json, from 27_litmus_poses.py), one page load per pose with
// only the meshes within LITMUS_RADIUS (400 m) of the eye, from the round-tripped kn5 scene.
// Same scene/lighting as 24_visual_render.js; pinhole camera with the pose's vertical fov, output = photo aspect.
// Usage: node 27_litmus.js [layout=road_full|road_summit fallback] [width=1280]   -> reports/litmus/render_<layout>_NNN.png
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '../..');
const lay = process.argv[2] || (fs.existsSync(path.join(root, 'build/vis/road_full/scene.json')) ? 'road_full' : 'road_summit');
const dir = path.join(root, 'build/vis', lay);
const cfg = JSON.parse(fs.readFileSync(path.join(root, 'config/litmus_poses.json')));
const OW = parseInt(process.argv[3] || '1280'), OH = Math.round(OW * cfg.size[1] / cfg.size[0]);
const outDir = path.join(root, 'reports/litmus'); fs.mkdirSync(outDir, { recursive: true });
const html = `<!doctype html><html><body style="margin:0;background:#000">
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"}}</script>
<script type="module">
import * as THREE from 'three';
const PI = new URLSearchParams(location.search).get('i');
const sc = await (await fetch('sub.json?i=' + PI)).json();
const bin = await (await fetch('sub.bin?i=' + PI)).arrayBuffer();
const poses = await (await fetch('litmus_poses.json')).json();
const r = new THREE.WebGLRenderer({antialias:true}); r.setSize(${OW},${OH}); document.body.appendChild(r.domElement);
r.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene(); scene.background = new THREE.Color(0xa9c4dc); scene.fog = new THREE.Fog(0xb8c8d6, 300, 2500);
scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x4a4030, 1.3));
const sun = new THREE.DirectionalLight(0xfff2dd, 1.6); sun.position.set(-0.5, 0.8, 0.3); scene.add(sun);
const tl = new THREE.TextureLoader(); const texs = {};
const mats = sc.materials.map(m => { let t = null;
  if (m.tex) { t = texs[m.tex] || (texs[m.tex] = tl.load(m.tex)); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; }
  const mm = new THREE.MeshLambertMaterial({map: t, alphaTest: m.alpha ? 0.5 : 0, side: THREE.FrontSide});
  if (m.det) { const d = texs[m.det] || (texs[m.det] = tl.load(m.det)); d.wrapS = d.wrapT = THREE.RepeatWrapping; d.colorSpace = THREE.SRGBColorSpace; d.anisotropy = 8;
    // ksPerPixelMultiMap: diffuse * detail(uv * detailUVMultiplier) where diffuse alpha = 0 (terrain macro layer);
    // terrain is opaque: force alpha 1 (as 24_visual_render.js)
    mm.onBeforeCompile = sh => { sh.uniforms.tDet = {value: d}; sh.uniforms.dMul = {value: m.dmul};
      sh.fragmentShader = 'uniform sampler2D tDet; uniform float dMul;\n' + sh.fragmentShader.replace('#include <map_fragment>',
        '#include <map_fragment>\n diffuseColor.rgb *= mix(texture2D(tDet, vMapUv * dMul).rgb, vec3(1.0), texture2D(map, vMapUv).a);\n diffuseColor.a = 1.0;'); };
    mm.customProgramCacheKey = () => 'det' + m.dmul; }
  return mm; });
for (const m of sc.meshes) { const g = new THREE.BufferGeometry(); let o = m.off;
  const f = (n, k) => { const a = new Float32Array(bin, o, n * k); o += n * k * 4; return a; };
  g.setAttribute('position', new THREE.BufferAttribute(f(m.nv, 3), 3));
  g.setAttribute('normal', new THREE.BufferAttribute(f(m.nv, 3), 3));
  const uv = f(m.nv, 2).slice(); for (let i = 1; i < uv.length; i += 2) uv[i] = -uv[i];
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(new THREE.BufferAttribute(new Uint32Array(bin, o, m.ni), 1));
  scene.add(new THREE.Mesh(g, mats[m.mat])); }
const cam = new THREE.PerspectiveCamera(poses.fov, ${OW}/${OH}, 0.3, 6000);
window.shot = i => { const c = poses.poses[i]; cam.fov = c.fov; cam.updateProjectionMatrix();
  cam.position.set(...c.eye); cam.up.set(...c.up); cam.lookAt(...c.look); r.render(scene, cam); return c.frame; };
window.ncams = poses.poses.length;
if (Object.keys(texs).length) await new Promise(res => THREE.DefaultLoadingManager.onLoad = res);
window.__ready = true;
</script></body></html>`;
fs.writeFileSync(path.join(dir, 'litmus.html'), html);
fs.copyFileSync(path.join(root, 'config/litmus_poses.json'), path.join(dir, 'litmus_poses.json'));
// per-pose subset: only meshes whose bbox is within RADIUS m (horizontal) of the eye -> keeps swiftshader page load small
const RADIUS = parseFloat(process.env.LITMUS_RADIUS || '400');
const SC = JSON.parse(fs.readFileSync(path.join(dir, 'scene.json'))), BIN = fs.readFileSync(path.join(dir, 'scene.bin'));
const bbox = SC.meshes.map(m => { const a = new Float32Array(BIN.buffer, BIN.byteOffset + m.off, m.nv * 3);
  let x0 = 1e9, x1 = -1e9, z0 = 1e9, z1 = -1e9;
  for (let i = 0; i < a.length; i += 3) { const x = a[i], z = a[i + 2]; if (x < x0) x0 = x; if (x > x1) x1 = x; if (z < z0) z0 = z; if (z > z1) z1 = z; }
  return [x0, x1, z0, z1]; });
const subs = cfg.poses.map(c => { const [ex, , ez] = c.eye; const ms = [], parts = []; let off = 0;
  SC.meshes.forEach((m, k) => { const [x0, x1, z0, z1] = bbox[k];
    const dx = Math.max(x0 - ex, 0, ex - x1), dz = Math.max(z0 - ez, 0, ez - z1);
    if (Math.hypot(dx, dz) > RADIUS) return;
    const len = m.nv * 8 * 4 + m.ni * 4; parts.push(BIN.subarray(m.off, m.off + len)); ms.push({ ...m, off }); off += len; });
  return { json: JSON.stringify({ ...SC, meshes: ms }), bin: Buffer.concat(parts), n: ms.length }; });
const types = { '.html': 'text/html', '.json': 'application/json', '.png': 'image/png' };
const srv = http.createServer((q, s) => { const u = decodeURIComponent(q.url.split('?')[0]);
  const qi = new URLSearchParams(q.url.split('?')[1] || '').get('i');
  if (u === '/sub.json' || u === '/sub.bin') { const S_ = subs[+qi]; s.writeHead(200); s.end(u === '/sub.json' ? S_.json : S_.bin); return; } const p = path.join(dir, u === '/' ? 'litmus.html' : u);
  fs.readFile(p, (e, d) => { if (e) { s.writeHead(404); s.end(); } else { s.writeHead(200, {'Content-Type': types[path.extname(p)] || 'application/octet-stream'}); s.end(d); } }); }).listen(8766);
(async () => {
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: OW, height: OH } });
  p.on('console', m => { if (m.type() === 'error') console.log('console:', m.text()); }); p.on('pageerror', e => console.log('pageerror:', e.message));
  const n = cfg.poses.length, t0 = Date.now();
  for (let i = 0; i < n; i++) {
    const t1 = Date.now();
    await p.goto('http://localhost:8766/?i=' + i);
    await p.waitForFunction(() => window.__ready === true, null, { timeout: parseInt(process.env.LITMUS_TIMEOUT || '600000') });
    const f = await p.evaluate(i => window.shot(i), i);
    await p.screenshot({ path: path.join(outDir, `render_${lay}_${String(f).padStart(3, '0')}.png`), timeout: 300000 });
    console.log('litmus', lay, f, 'meshes', subs[i].n, 'MB', (subs[i].bin.length / 1e6).toFixed(0), 's', ((Date.now() - t1) / 1000).toFixed(1));
  }
  console.log('total s', ((Date.now() - t0) / 1000).toFixed(1));
  await b.close(); srv.close();
})();
