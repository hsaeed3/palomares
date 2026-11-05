// Litmus test, step 2: render each fitted GoPro pose (config/litmus_poses.json) from the round-tripped kn5 scene.
// Same scene/lighting as 24_visual_render.js; pinhole camera with the pose's vertical fov, output = photo aspect.
// Usage: node 27_litmus.js [layout=road_summit] [width=1280]   -> reports/litmus/render_<layout>_NNN.png
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '../..');
const lay = process.argv[2] || 'road_summit';
const dir = path.join(root, 'build/vis', lay);
const cfg = JSON.parse(fs.readFileSync(path.join(root, 'config/litmus_poses.json')));
const OW = parseInt(process.argv[3] || '1280'), OH = Math.round(OW * cfg.size[1] / cfg.size[0]);
const outDir = path.join(root, 'reports/litmus'); fs.mkdirSync(outDir, { recursive: true });
const html = `<!doctype html><html><body style="margin:0;background:#000">
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"}}</script>
<script type="module">
import * as THREE from 'three';
const sc = await (await fetch('scene.json')).json();
const bin = await (await fetch('scene.bin')).arrayBuffer();
const poses = await (await fetch('litmus_poses.json')).json();
const r = new THREE.WebGLRenderer({antialias:true}); r.setSize(${OW},${OH}); document.body.appendChild(r.domElement);
r.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene(); scene.background = new THREE.Color(0xa9c4dc); scene.fog = new THREE.Fog(0xb8c8d6, 300, 2500);
scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x4a4030, 1.3));
const sun = new THREE.DirectionalLight(0xfff2dd, 1.6); sun.position.set(-0.5, 0.8, 0.3); scene.add(sun);
const tl = new THREE.TextureLoader(); const texs = {};
const mats = sc.materials.map(m => { let t = null;
  if (m.tex) { t = texs[m.tex] || (texs[m.tex] = tl.load(m.tex)); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; }
  return new THREE.MeshLambertMaterial({map: t, alphaTest: m.alpha ? 0.5 : 0, side: THREE.FrontSide}); });
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
await new Promise(res => THREE.DefaultLoadingManager.onLoad = res);
window.__ready = true;
</script></body></html>`;
fs.writeFileSync(path.join(dir, 'litmus.html'), html);
fs.copyFileSync(path.join(root, 'config/litmus_poses.json'), path.join(dir, 'litmus_poses.json'));
const types = { '.html': 'text/html', '.json': 'application/json', '.png': 'image/png' };
const srv = http.createServer((q, s) => { const u = decodeURIComponent(q.url.split('?')[0]); const p = path.join(dir, u === '/' ? 'litmus.html' : u);
  fs.readFile(p, (e, d) => { if (e) { s.writeHead(404); s.end(); } else { s.writeHead(200, {'Content-Type': types[path.extname(p)] || 'application/octet-stream'}); s.end(d); } }); }).listen(8766);
(async () => {
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: OW, height: OH } });
  p.on('console', m => { if (m.type() === 'error') console.log('console:', m.text()); });
  await p.goto('http://localhost:8766/');
  await p.waitForFunction(() => window.__ready === true, null, { timeout: 600000 });
  const n = await p.evaluate(() => window.ncams);
  for (let i = 0; i < n; i++) {
    const f = await p.evaluate(i => window.shot(i), i);
    await p.screenshot({ path: path.join(outDir, `render_${lay}_${String(f).padStart(3, '0')}.png`), timeout: 300000 });
    console.log('litmus', lay, f);
  }
  await b.close(); srv.close();
})();
