// Render viewer screenshots for each corner in assets/corners.json (driver's eye ~45 m before apex + overhead).
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '..');
const corners = JSON.parse(fs.readFileSync(path.join(root, 'assets/corners.json'))).filter(c => !process.env.ONLY || process.env.ONLY.split(',').some(o => c.name.includes(o)));
(async () => {
  const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
  page.on('console', m => { if (m.type() === 'error') console.log('console:', m.text()); });
  const url = 'file://' + path.join(root, 'reports/viewer.html');
  await page.goto(url + '?noui=1&clean=1&static=1&s=0&mode=drive');
  await page.waitForFunction(() => window.__ready === true, null, { timeout: 180000 });
  for (const c of corners) {
    const sDrive = Math.max(5, c.s - (c.radius_m ? 45 : 0));
    await page.evaluate(s => window.setView(s, 'drive'), sDrive);
    await page.waitForTimeout(300);
    await page.screenshot({ path: path.join(root, 'assets', c.name + '_drive.jpg'), type: 'jpeg', quality: 85, timeout: 240000 });
    await page.evaluate(s => window.setView(s, 'orbit'), c.s);
    await page.waitForTimeout(300);
    await page.screenshot({ path: path.join(root, 'assets', c.name + '_orbit.jpg'), type: 'jpeg', quality: 85, timeout: 240000 });
    console.log('shot', c.name);
  }
  await browser.close();
})();
