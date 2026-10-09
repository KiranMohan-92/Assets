// Drive scene.js in headless Chromium and encode frames with ffmpeg.
// usage: node render.mjs --assets DIR --out out.mp4 [--w 1920 --h 1080 --fps 30 --from 0 --to 32]
//                        [--audio score.wav] [--preview] [--stills 1,8.5,13 --stillsDir DIR]
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { chromium } = await import(process.env.PLAYWRIGHT_PATH || '/opt/node22/lib/node_modules/playwright/index.mjs');

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
  if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : true]);
  return acc;
}, []));
const w = +(args.w || 1920), h = +(args.h || 1080), fps = +(args.fps || 30);
const from = +(args.from || 0), to = +(args.to || 32);
const assets = path.resolve(args.assets);
const threePath = path.join(path.dirname(require.resolve('three')), 'three.module.js');

const types = { '.js': 'text/javascript', '.html': 'text/html', '.json': 'application/json', '.jpg': 'image/jpeg', '.png': 'image/png', '.bin': 'application/octet-stream' };
const server = http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  let file;
  if (url === '/three.module.js') file = threePath;
  else if (url.startsWith('/assets/')) file = path.join(assets, url.slice(8));
  else file = path.join(here, url === '/' ? 'index.html' : url);
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'content-type': types[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: w, height: h } });
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.error('[page]', m.text()); });
page.on('pageerror', (e) => console.error('[pageerror]', e.message));
await page.goto(`http://127.0.0.1:${port}/`);
await page.addStyleTag({ content: `:root{--w:${w}px;--h:${h}px}` });
await page.waitForFunction(() => window.__ready && window.init);
const info = await page.evaluate((cfg) => window.init(cfg), { assets: '/assets', w, h, fps, preview: !!args.preview, maxSub: +(args.maxSub || 6) });
console.error('shots', JSON.stringify(info.shots));

const grab = () => page.screenshot({ type: 'jpeg', quality: 94, clip: { x: 0, y: 0, width: w, height: h } });

if (args.stills) {
  const dir = path.resolve(args.stillsDir || '.');
  fs.mkdirSync(dir, { recursive: true });
  for (const t of String(args.stills).split(',').map(Number)) {
    await page.evaluate((t) => window.renderFrame(t), t);
    fs.writeFileSync(path.join(dir, `still_${t.toFixed(2)}.jpg`), await grab());
    console.error('still', t);
  }
} else {
  const out = path.resolve(args.out || 'out.mp4');
  const ff = ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'mjpeg', '-i', '-'];
  if (args.audio) ff.push('-ss', String(from), '-t', String(to - from), '-i', path.resolve(args.audio), '-c:a', 'aac', '-b:a', '256k', '-shortest');
  ff.push('-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out);
  const enc = spawn('ffmpeg', ff, { stdio: ['pipe', 'inherit', 'inherit'] });
  const total = Math.round((to - from) * fps);
  const t0 = Date.now();
  for (let i = 0; i < total; i++) {
    const t = from + i / fps;
    const n = await page.evaluate((t) => window.renderFrame(t), t);
    const buf = await grab();
    if (!enc.stdin.write(buf)) await new Promise((r) => enc.stdin.once('drain', r));
    if (i % 30 === 0) console.error(`frame ${i}/${total} t=${t.toFixed(2)} sub=${n} ${((Date.now() - t0) / 1000 / (i + 1)).toFixed(2)}s/f`);
  }
  enc.stdin.end();
  await new Promise((r) => enc.on('close', r));
  console.error('wrote', out);
}
await browser.close();
server.close();
