// Three.js renderer for the Last Supper fly-through.
// The page exposes window.init(cfg) and window.renderFrame(t); render.mjs drives it.
import * as THREE from 'three';

const $ = (s) => document.querySelector(s);
let R, S, cam, scn, dustScn, rt, rtDof, rtBloomA, rtBloomB, rtFinal, accRT;
let mats = {}, quad, quadScn, quadCam, beams = [], dust;
let fgDepth, G, CFG;

// ---------------------------------------------------------------- helpers
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = {
  lin: (t) => t,
  io: (t) => t * t * (3 - 2 * t),
  io3: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  outExpo: (t) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t)),
  inExpo: (t) => (t <= 0 ? 0 : Math.pow(2, 10 * t - 10)),
  out3: (t) => 1 - Math.pow(1 - t, 3),
  in3: (t) => t * t * t,
};

// world point for normalized image coords at z-distance z (metres from the eye plane)
function W(u, v, z) {
  const O = G.origin;
  const px = (u - 0.5) * G.PW, py = (0.5 - v) * G.PH;
  const k = z / G.Dp;
  return V(O[0] + (px - O[0]) * k, O[1] + (py - O[1]) * k, -z);
}
function depthAt(u, v) {
  const [gw, gh] = G.fgGrid;
  const x = clamp(Math.round(u * gw - 0.5), 0, gw - 1), y = clamp(Math.round(v * gh - 0.5), 0, gh - 1);
  return fgDepth[y * gw + x];
}
// subject point on the figure layer, optionally pulled toward camera
function subj(name, dz = 0) {
  const [u, v] = G.subjects[name];
  return W(u, v, depthAt(u, v) + dz);
}
function catmull(pts, t) {
  if (pts.length === 1) return pts[0].clone();
  const n = pts.length - 1, f = clamp(t) * n, i = Math.min(Math.floor(f), n - 1), s = f - i;
  const p0 = pts[Math.max(i - 1, 0)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(i + 2, n)];
  const s2 = s * s, s3 = s2 * s;
  return V(0, 0, 0)
    .addScaledVector(p0, -0.5 * s3 + s2 - 0.5 * s)
    .addScaledVector(p1, 1.5 * s3 - 2.5 * s2 + 1)
    .addScaledVector(p2, -1.5 * s3 + 2 * s2 + 0.5 * s)
    .addScaledVector(p3, 0.5 * s3 - 0.5 * s2);
}
const noise1 = (t, seed) => Math.sin(t * 13.1 + seed) * 0.5 + Math.sin(t * 7.3 + seed * 2.1) * 0.3 + Math.sin(t * 23.7 + seed * 0.7) * 0.2;

// ---------------------------------------------------------------- shot list
// Each shot: [t0, t1, fn(p, t) -> partial state]; p is 0..1 through the shot.
let SHOTS = [];
function buildShots() {
  const O = V(...G.origin);
  const zF = G.table.Zf, zT = G.table.Zt, zB = -G.room.zb, Dp = G.Dp;
  const center = W(0.5, 0.5, Dp);
  const heroShift = -G.origin[1] / Dp;
  const heroFov = (2 * Math.atan(G.PW / 2 / Dp / (16 / 9)) * 180) / Math.PI;
  const face = subj('christFace', -0.15);

  SHOTS = [
    // 1. the painting floats in darkness; slow push in
    [0, 4, (p) => ({
      pos: catmull([O.clone().add(V(0, -0.6, 16)), O.clone().add(V(0, -0.2, 7)), O.clone().add(V(0, 0, 2.5))], ease.io(p)),
      look: center, fov: 34 - 4 * p, roll: -2 + 2 * p, focus: 'look', ap: 0.6,
      fade: 1 - ease.out3(clamp(p * 2.2)), exposure: 0.85 + 0.15 * p, rays: 0.4 * p, lb: 1,
      spot: [lerp(0.15, 0.7, p), 0.4, 1.4],
    })],
    // 2. through the picture plane, crane over the table
    [4, 8, (p) => {
      const e = ease.in3(p) * 0.6 + ease.io(p) * 0.4;
      return {
        pos: catmull([O.clone().add(V(0, 0, 2.5)), W(0.5, 0.2, Dp + 0.5), W(0.5, 0.16, zT - 1.2)], e),
        look: catmull([center, W(0.5, 0.5, zT), subj('bread')], ease.io(p)),
        fov: 40 - 6 * p, roll: 3 * Math.sin(p * Math.PI), focus: 'look', ap: 1.2, exposure: 1, rays: 0.6 + 0.4 * p, lb: 1,
        spot: [0.5, 0.6, 1.2], shake: 0.002 + 0.01 * ease.in3(p),
      };
    }],
    // 3. DROP: crash zoom into Christ's face
    [8, 10, (p) => {
      const e = ease.outExpo(p);
      return {
        pos: catmull([face.clone().add(V(0.4, 0.3, 7)), face.clone().add(V(0.05, 0.08, 1.25))], e),
        look: face, fov: 52 - 26 * e, roll: 4 * (1 - e), focus: face, ap: 2.6,
        flash: Math.max(0, 1 - p * 6), shake: 0.05 * Math.max(0, 1 - p * 3) + 0.004, ca: 1 - e, exposure: 1.05, rays: 1, lb: 1,
        spot: [0.5, 0.38, 0.6],
      };
    }],
    // 4. lateral truck across the left group
    [10, 12, (p) => ({
      pos: catmull([W(0.33, 0.44, zF - 2.4), W(0.10, 0.42, zF - 2.0)], ease.io(p)),
      look: catmull([subj('groupLeft').add(V(1.2, 0, 0)), subj('groupLeft').add(V(-0.5, 0, 0))], ease.io(p)),
      fov: 36, roll: lerp(-4, 2, p), focus: 'look', ap: 2.0, shake: 0.004, ca: 0.3 * (1 - p), exposure: 1, rays: 0.7, lb: 1,
      flash: Math.max(0, 0.5 - p * 4), spot: [0.18, 0.42, 0.6],
    })],
    // 5. low-angle push on Judas, Peter and John
    [12, 14, (p) => ({
      pos: catmull([W(0.37, 0.66, zF - 3.4), W(0.39, 0.55, zF - 1.5)], ease.out3(p)),
      look: subj('judas').add(V(0.25, 0.1, 0)), fov: 38 - 6 * p, roll: lerp(5, 1, p), focus: subj('judas'), ap: 2.6,
      flash: Math.max(0, 0.6 - p * 5), shake: 0.006, ca: 0.6 * (1 - ease.out3(p)), exposure: 0.95, rays: 0.8, lb: 1,
      spot: [0.4, 0.42, 0.5],
    })],
    // 6. orbit around Thomas, James and Philip
    [14, 16, (p) => {
      const c = subj('philip').add(V(-0.25, 0, 0));
      const a = lerp(0.55, -0.35, ease.io(p)), r = 2.8;
      return {
        pos: c.clone().add(V(Math.sin(a) * r, 0.25 + 0.2 * p, Math.cos(a) * r)), look: c, fov: 34, roll: lerp(-3, 3, p),
        focus: 'look', ap: 2.2, flash: Math.max(0, 0.5 - p * 4), shake: 0.004, exposure: 1, rays: 0.8, lb: 1,
        spot: [0.66, 0.4, 0.6],
      };
    }],
    // 7. dutch pull-out on Matthew, Thaddeus and Simon
    [16, 18, (p) => ({
      pos: catmull([W(0.85, 0.43, zF - 1.1), W(0.80, 0.47, zF - 3.6)], ease.out3(p)),
      look: subj('groupRight'), fov: 28 + 18 * ease.out3(p), roll: lerp(9, -3, ease.out3(p)), focus: 'look', ap: 2.0,
      flash: Math.max(0, 0.5 - p * 4), shake: 0.005, ca: 0.4 * (1 - p), exposure: 1, rays: 0.8, lb: 1,
      spot: [0.84, 0.42, 0.6],
    })],
    // 8. fly over Christ's head toward the central window
    [18, 20, (p) => {
      const e = ease.in3(p) * 0.7 + p * 0.3;
      return {
        pos: catmull([W(0.5, 0.6, zT - 1.6), W(0.5, 0.21, zF + 0.3), W(0.5, 0.32, zB - 1.6)], e),
        look: catmull([face, W(0.5, 0.33, zB)], ease.io(clamp(p * 1.6))),
        fov: 38 + 14 * e, roll: 6 * Math.sin(p * Math.PI), focus: 'look', ap: 1.2, shake: 0.004 + 0.012 * e,
        exposure: 1 + 0.8 * ease.in3(p), flash: ease.in3(clamp((p - 0.85) / 0.15)), rays: 1 + p, ca: e * 0.6, lb: 1,
        spot: [0.5, 0.33, 0.5],
      };
    }],
    // 9. rip back out of the window, revealing the whole room
    [20, 22, (p) => {
      const e = ease.outExpo(p);
      return {
        pos: catmull([W(0.5, 0.33, zB - 2.2), W(0.5, 0.18, zF + 0.2), O.clone().add(V(0, 0.15, -1.0))], e),
        look: catmull([W(0.5, 0.33, zB), W(0.5, 0.42, zF)], e), fov: 60 - 18 * e, roll: lerp(-10, 0, e),
        focus: 'look', ap: 0.9, flash: Math.max(0, 1 - p * 5), shake: 0.03 * (1 - e) + 0.003, ca: 1 - e,
        exposure: 1.3 - 0.3 * e, rays: 1.4, lb: 1, spot: [0.5, 0.45, 1.0],
      };
    }],
    // 10. snap zooms on beat: hands, Judas, Peter, Thomas
    ...['christHands', 'judas', 'peter', 'thomas'].map((name, i) => [22 + i * 0.5, 22.5 + i * 0.5, (p) => {
      const tgt = subj(name, -0.05), e = ease.outExpo(clamp(p * 1.4));
      const side = i % 2 ? 1 : -1;
      return {
        pos: tgt.clone().add(V(0.35 * side, 0.25, lerp(2.6, 0.95, e) + 0.15 * p)), look: tgt, fov: lerp(44, 30, e),
        roll: side * lerp(7, 3, e), focus: tgt, ap: 3.0, flash: Math.max(0, 0.7 - p * 4), shake: 0.02 * (1 - e) + 0.003,
        ca: 0.8 * (1 - e), exposure: 1, rays: 1, lb: 1, spot: [G.subjects[name][0], G.subjects[name][1], 0.4],
      };
    }]),
    // 11. silence: black
    [24, 24.5, () => ({ pos: O.clone(), look: center, fov: 40, fade: 1, lb: 1 })],
    // 12. epic crane: low left, sweep across and rise back to the full room
    [24.5, 29, (p) => {
      const e = ease.io3(p);
      return {
        pos: catmull([W(0.2, 0.74, zT - 2.2), W(0.38, 0.62, zT - 3.4), W(0.62, 0.5, Dp + 0.2), O.clone().add(V(0, 0.1, 3.0))], e),
        look: catmull([face, face, center], e), fov: lerp(36, 40, e), roll: lerp(-5, 0, e), focus: 'look', ap: lerp(2.2, 0.5, e),
        flash: Math.max(0, 1 - p * 10), shake: 0.03 * Math.max(0, 1 - p * 6) + 0.002, ca: Math.max(0, 1 - p * 5),
        exposure: 1.05, rays: 1.5 - 0.5 * e, lb: 1, spot: [0.5, 0.4, 1.4 * (1 - e) + 0.6],
      };
    }],
    // 13. settle on the exact original viewpoint; title
    [29, 32.01, (p) => {
      const e = ease.io3(clamp(p * 2.0));
      return {
        pos: catmull([O.clone().add(V(0, 0.1, 3.0)), O.clone()], e), look: catmull([center, O.clone().add(V(0, 0, -1))], e),
        shift: heroShift * e, fov: lerp(40, heroFov, e), roll: 0, focus: Dp, ap: 0.3 * (1 - e),
        exposure: 1 - 0.35 * ease.io(clamp((p - 0.3) / 0.4)), rays: 1, lb: 1 - 0.6 * e, fade: ease.in3(clamp((p - 0.8) / 0.2)),
        spot: [0.5, 0.4, 2.0],
      };
    }],
  ];
}

const DEFAULTS = { fov: 40, roll: 0, shift: 0, focus: 'look', ap: 1, fade: 0, flash: 0, shake: 0, ca: 0, exposure: 1, rays: 0.6, lb: 1, spot: [0.5, 0.4, 1] };
function stateAt(t) {
  let shot = SHOTS[SHOTS.length - 1];
  for (const s of SHOTS) if (t >= s[0] && t < s[1]) { shot = s; break; }
  const p = clamp((t - shot[0]) / (shot[1] - shot[0]));
  const st = Object.assign({}, DEFAULTS, shot[2](p, t));
  if (st.shake) {
    const k = st.shake;
    st.pos = st.pos.clone().add(V(noise1(t * 3, 1) * k, noise1(t * 3, 2) * k, noise1(t * 3, 3) * k * 0.5));
    st.look = st.look.clone().add(V(noise1(t * 2.3, 4) * k * 2, noise1(t * 2.3, 5) * k * 2, 0));
  }
  st.focusDist = st.focus === 'look' ? st.pos.distanceTo(st.look) : (typeof st.focus === 'number' ? st.focus : st.pos.distanceTo(st.focus));
  return st;
}

// ---------------------------------------------------------------- shaders
const vsQuad = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0., 1.); }`;
const paintFS = `
uniform sampler2D map; uniform float exposure; uniform vec3 spotPos; uniform float spotR; uniform float flicker;
varying vec2 vUv; varying vec3 vW;
void main(){
  vec4 c = texture2D(map, vUv);
  if (c.a < 0.02) discard;
  float s = exp(-pow(distance(vW, spotPos) / spotR, 2.));
  vec3 col = c.rgb * (0.78 + 0.45 * s) * flicker;
  gl_FragColor = vec4(col, c.a);
}`;
const paintVS = `varying vec2 vUv; varying vec3 vW;
void main(){ vUv = uv; vec4 w = modelMatrix * vec4(position,1.); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;

const dofFS = `
uniform sampler2D tCol; uniform sampler2D tDep; uniform float near; uniform float far;
uniform float focus; uniform float aperture; uniform vec2 res; varying vec2 vUv;
float lin(float d){ float z = d * 2. - 1.; return 2. * near * far / (far + near - z * (far - near)); }
float coc(float z){ return clamp(aperture * abs(z - focus) / max(z, 0.01) * 0.012, 0., 0.022); }
float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
void main(){
  float zc = lin(texture2D(tDep, vUv).r);
  float cc = coc(zc);
  vec3 acc = texture2D(tCol, vUv).rgb; float wsum = 1.;
  float maxR = 0.022;
  if (aperture > 0.01) {
    float rot = hash(vUv * res) * 6.2831;
    for (int i = 0; i < 40; i++) {
      float fi = float(i) + 0.5;
      float r = sqrt(fi / 40.) * maxR;
      float a = fi * 2.39996 + rot;
      vec2 off = vec2(cos(a) * res.y / res.x, sin(a)) * r;
      vec2 uv2 = vUv + off;
      float zs = lin(texture2D(tDep, uv2).r);
      float cs = coc(zs);
      if (zs > zc) cs = min(cs, cc);       // background must not bleed onto sharp foreground
      float w = clamp((cs - r) * res.y * 0.5 + 1., 0., 1.);
      acc += texture2D(tCol, uv2).rgb * w; wsum += w;
    }
  }
  gl_FragColor = vec4(acc / wsum, 1.);
}`;
const brightFS = `uniform sampler2D tCol; varying vec2 vUv;
void main(){ vec3 c = texture2D(tCol, vUv).rgb; float l = dot(c, vec3(.299,.587,.114)); gl_FragColor = vec4(c * smoothstep(.55, 1., l), 1.); }`;
const blurFS = `uniform sampler2D tCol; uniform vec2 dir; varying vec2 vUv;
void main(){ vec3 c = vec3(0.); float w[5]; w[0]=.227; w[1]=.195; w[2]=.122; w[3]=.054; w[4]=.016;
  c += texture2D(tCol, vUv).rgb * w[0];
  for (int i = 1; i < 5; i++){ c += (texture2D(tCol, vUv + dir * float(i)).rgb + texture2D(tCol, vUv - dir * float(i)).rgb) * w[i]; }
  gl_FragColor = vec4(c, 1.); }`;
const finalFS = `
uniform sampler2D tCol; uniform sampler2D tBloom; uniform float exposure; uniform float flash; uniform float fade;
uniform float ca; uniform float lb; uniform float seed; uniform vec2 res; uniform float weight; varying vec2 vUv;
float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233)) + seed) * 43758.5453); }
void main(){
  vec2 d = (vUv - .5);
  float cr = 0.004 * ca + 0.0006;
  vec3 col;
  col.r = texture2D(tCol, vUv - d * cr).r;
  col.g = texture2D(tCol, vUv).g;
  col.b = texture2D(tCol, vUv + d * cr).b;
  col += texture2D(tBloom, vUv).rgb * 0.55;
  col *= exposure;
  col = col / (1. + 0.22 * max(col - 0.75, 0.));            // soft shoulder
  float l = dot(col, vec3(.299, .587, .114));
  col = mix(col, col * vec3(.86, 1.0, 1.10), (1. - l) * .35); // teal shadows
  col = mix(col, col * vec3(1.10, 1.0, .86), l * .30);        // warm highlights
  col = (col - .5) * 1.08 + .5;
  float vig = smoothstep(.95, .25, length(d * vec2(1., .85)));
  col *= mix(.55, 1., vig);
  col = mix(col, vec3(1., .97, .9), flash);
  col *= 1. - fade;
  col += (hash(vUv * res) - .5) * .05;
  float half_ = mix(.5, .372, lb);
  if (abs(vUv.y - .5) > half_) col = vec3(0.);
  gl_FragColor = vec4(clamp(col, 0., 1.) * weight, 1.);
}`;
const copyFS = `uniform sampler2D tCol; varying vec2 vUv; void main(){ gl_FragColor = vec4(texture2D(tCol, vUv).rgb, 1.); }`;

const beamVS = `varying vec2 vUv; varying vec3 vW; void main(){ vUv = uv; vec4 w = modelMatrix * vec4(position,1.); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;
const beamFS = `uniform float strength; uniform float time; varying vec2 vUv;
float h(float x){ return fract(sin(x * 91.7) * 4375.5); }
void main(){
  float across = sin(vUv.x * 3.14159);
  float along = pow(1. - vUv.y, 1.6) * smoothstep(0., .08, vUv.y);
  float streak = .7 + .3 * sin(vUv.x * 40. + time * .7) * sin(vUv.x * 17. - time * .4);
  float a = across * across * along * streak * strength * .09;
  gl_FragColor = vec4(vec3(1., .88, .62) * a, 1.);
}`;
const dustVS = `uniform float time; uniform float focus; uniform float aperture; uniform float pxScale; attribute float seed;
varying float vA; varying float vBlur;
void main(){
  vec3 p = position + vec3(sin(time * .3 + seed * 6.) * .15, sin(time * .21 + seed * 9.) * .1 - time * .02, cos(time * .25 + seed * 4.) * .15);
  vec4 mv = modelViewMatrix * vec4(p, 1.);
  float z = -mv.z;
  float coc = clamp(aperture * abs(z - focus) / max(z, .01) * .012, 0., .03);
  gl_PointSize = clamp((0.012 / z + coc) * pxScale, 1., 90.);
  vBlur = coc * pxScale;
  vA = (.35 + .65 * fract(seed * 13.7)) / (1. + vBlur * vBlur * .015) * smoothstep(.2, 1.2, z);
  gl_Position = projectionMatrix * mv;
}`;
const dustFS = `uniform sampler2D tDep; uniform vec2 res; uniform float near; uniform float far; varying float vA; varying float vBlur;
float lin(float d){ float z = d * 2. - 1.; return 2. * near * far / (far + near - z * (far - near)); }
void main(){
  vec2 c = gl_PointCoord - .5;
  float r = length(c) * 2.;
  float edge = vBlur > 4. ? smoothstep(1., .85, r) * (.8 + .2 * r) : smoothstep(1., 0., r);
  float sceneZ = lin(texture2D(tDep, gl_FragCoord.xy / res).r);
  if (1. / gl_FragCoord.w > sceneZ) discard;
  gl_FragColor = vec4(vec3(1., .9, .72) * edge * vA * .5, 1.);
}`;

// ---------------------------------------------------------------- setup
function loadTex(url) {
  return new Promise((res) => new THREE.TextureLoader().load(url, (t) => {
    t.colorSpace = THREE.NoColorSpace; t.anisotropy = 8; t.generateMipmaps = true; t.minFilter = THREE.LinearMipmapLinearFilter;
    res(t);
  }));
}
function gridGeom(gw, gh, posFn) {
  const pos = new Float32Array((gw + 1) * (gh + 1) * 3), uv = new Float32Array((gw + 1) * (gh + 1) * 2);
  let k = 0;
  for (let j = 0; j <= gh; j++) for (let i = 0; i <= gw; i++) {
    const u = i / gw, v = j / gh, p = posFn(u, v);
    pos.set([p.x, p.y, p.z], k * 3); uv.set([u, 1 - v], k * 2); k++;
  }
  const idx = [];
  for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
    const a = j * (gw + 1) + i, b = a + 1, c = a + gw + 1, d = c + 1;
    idx.push(a, c, b, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(idx);
  return g;
}
function roomHit(u, v) {
  const O = G.origin, r = G.room;
  const px = (u - 0.5) * G.PW, py = (0.5 - v) * G.PH;
  const d = V(px - O[0], py - O[1], -G.Dp);
  let t = Infinity;
  const test = (num, den) => { if (Math.abs(den) > 1e-9) { const tt = num / den; if (tt > 0 && tt < t) t = tt; } };
  test(r.xl - O[0], d.x); test(r.xr - O[0], d.x); test(r.yb - O[1], d.y); test(r.yt - O[1], d.y); test(r.zb - 0, d.z);
  return V(O[0], O[1], 0).addScaledVector(d, t);
}
const mkQuadMat = (fs, uniforms, extra = {}) => new THREE.ShaderMaterial({ vertexShader: vsQuad, fragmentShader: fs, uniforms, depthTest: false, depthWrite: false, ...extra });

window.init = async (cfg) => {
  CFG = cfg;
  G = await (await fetch(cfg.assets + '/scene.json')).json();
  fgDepth = new Float32Array(await (await fetch(cfg.assets + '/fgdepth.bin')).arrayBuffer());
  const [bgTex, fgTex] = await Promise.all([loadTex(cfg.assets + '/bg.jpg'), loadTex(cfg.assets + '/fg.png')]);
  const { w, h } = cfg;
  R = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
  R.setPixelRatio(1); R.setSize(w, h); R.outputColorSpace = THREE.LinearSRGBColorSpace;
  R.autoClear = false;
  $('#stage').appendChild(R.domElement);

  scn = new THREE.Scene(); scn.background = new THREE.Color(0, 0, 0);
  cam = new THREE.PerspectiveCamera(40, w / h, 0.05, 200);

  const paintUniforms = (map) => ({ map: { value: map }, exposure: { value: 1 }, spotPos: { value: V(0, 0, 0) }, spotR: { value: 1 }, flicker: { value: 1 } });
  mats.bg = new THREE.ShaderMaterial({ vertexShader: paintVS, fragmentShader: paintFS, uniforms: paintUniforms(bgTex), side: THREE.DoubleSide });
  mats.fg = new THREE.ShaderMaterial({ vertexShader: paintVS, fragmentShader: paintFS, uniforms: paintUniforms(fgTex), side: THREE.DoubleSide, transparent: true });
  const bgMesh = new THREE.Mesh(gridGeom(640, Math.round(640 / G.aspect), roomHit), mats.bg);
  const [gw, gh] = G.fgGrid;
  const fgMesh = new THREE.Mesh(gridGeom(gw - 1, gh - 1, (u, v) => {
    const uu = (u * (gw - 1) + 0.5) / gw, vv = (v * (gh - 1) + 0.5) / gh;
    return W(uu, vv, depthAt(uu, vv));
  }), mats.fg);
  // the fg grid samples pixel centres; remap its uvs to match
  const uvA = fgMesh.geometry.attributes.uv;
  for (let i = 0; i < uvA.count; i++) {
    uvA.setX(i, (uvA.getX(i) * (gw - 1) + 0.5) / gw);
    uvA.setY(i, 1 - ((1 - uvA.getY(i)) * (gh - 1) + 0.5) / gh);
  }
  fgMesh.renderOrder = 1;
  scn.add(bgMesh, fgMesh);

  // god-ray beams from the three windows
  const beamMat = new THREE.ShaderMaterial({ vertexShader: beamVS, fragmentShader: beamFS, uniforms: { strength: { value: 1 }, time: { value: 0 } },
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide });
  mats.beam = beamMat;
  const zB = -G.room.zb;
  for (const [u, wdt] of [[0.395, 0.05], [0.5, 0.075], [0.605, 0.05]]) {
    for (let k = 0; k < 3; k++) {
      const top = W(u, 0.27, zB - 0.05), len = zB - G.table.Zf + 1.5;
      const g = new THREE.PlaneGeometry(1, 1, 1, 1);
      g.translate(0, -0.5, 0);
      const m = new THREE.Mesh(g, beamMat);
      const width = wdt * G.PW * (zB / G.Dp) * (0.8 + 0.25 * k);
      m.scale.set(width, len, 1);
      m.position.copy(top);
      m.rotation.set(-Math.PI / 2 + 0.32 + k * 0.05, (k - 1) * 0.5, 0);
      beams.push(m); scn.add(m);
    }
  }

  // dust motes in front of the figures
  dustScn = new THREE.Scene();
  const n = 1800, dp = new Float32Array(n * 3), ds = new Float32Array(n);
  let s = 12345; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < n; i++) {
    const p = W(0.05 + 0.9 * rnd(), 0.08 + 0.75 * rnd(), lerp(G.Dp * 0.2, G.table.Zf + 3, rnd()));
    dp.set([p.x, p.y, p.z], i * 3); ds[i] = rnd();
  }
  const dg = new THREE.BufferGeometry();
  dg.setAttribute('position', new THREE.BufferAttribute(dp, 3));
  dg.setAttribute('seed', new THREE.BufferAttribute(ds, 1));

  const mkRT = (ww, hh, depth) => {
    const r = new THREE.WebGLRenderTarget(ww, hh, { type: THREE.HalfFloatType, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
    if (depth) { r.depthTexture = new THREE.DepthTexture(ww, hh); r.depthTexture.type = THREE.UnsignedIntType; }
    return r;
  };
  rt = mkRT(w, h, true); rtDof = mkRT(w, h); rtFinal = mkRT(w, h); accRT = mkRT(w, h);
  rtBloomA = mkRT(w >> 2, h >> 2); rtBloomB = mkRT(w >> 2, h >> 2);

  mats.dust = new THREE.ShaderMaterial({ vertexShader: dustVS, fragmentShader: dustFS, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
    uniforms: { time: { value: 0 }, focus: { value: 5 }, aperture: { value: 1 }, pxScale: { value: h }, tDep: { value: rt.depthTexture }, res: { value: new THREE.Vector2(w, h) }, near: { value: 0.05 }, far: { value: 200 } } });
  dust = new THREE.Points(dg, mats.dust);
  dustScn.add(dust);

  const res = new THREE.Vector2(w, h);
  mats.dof = mkQuadMat(dofFS, { tCol: { value: rt.texture }, tDep: { value: rt.depthTexture }, near: { value: 0.05 }, far: { value: 200 }, focus: { value: 5 }, aperture: { value: 1 }, res: { value: res } });
  mats.bright = mkQuadMat(brightFS, { tCol: { value: rtDof.texture } });
  mats.blur = mkQuadMat(blurFS, { tCol: { value: null }, dir: { value: new THREE.Vector2() } });
  mats.final = mkQuadMat(finalFS, { tCol: { value: rtDof.texture }, tBloom: { value: rtBloomA.texture }, exposure: { value: 1 }, flash: { value: 0 }, fade: { value: 0 },
    ca: { value: 0 }, lb: { value: 1 }, seed: { value: 0 }, res: { value: res }, weight: { value: 1 } }, { blending: THREE.AdditiveBlending, transparent: true });
  mats.copy = mkQuadMat(copyFS, { tCol: { value: accRT.texture } });
  quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mats.copy);
  quad.frustumCulled = false;
  quadScn = new THREE.Scene(); quadScn.add(quad);
  quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  buildShots();
  return { shots: SHOTS.map((s) => [s[0], s[1]]) };
};

function pass(mat, target, clear = true) {
  quad.material = mat;
  R.setRenderTarget(target);
  if (clear) R.clear();
  R.render(quadScn, quadCam);
}

function applyCamera(st) {
  cam.position.copy(st.pos);
  cam.up.set(0, 1, 0);
  cam.lookAt(st.look);
  cam.rotateZ((st.roll * Math.PI) / 180);
  const near = 0.05, far = 200, tanV = Math.tan((st.fov * Math.PI) / 360), tanH = tanV * cam.aspect;
  const sh = st.shift || 0;
  cam.near = near; cam.far = far;
  cam.projectionMatrix.makePerspective(-tanH * near, tanH * near, (tanV + sh) * near, (-tanV + sh) * near, near, far);
  cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
  cam.updateMatrixWorld();
}

function renderOne(t, weight, frameSeed) {
  const st = stateAt(t);
  applyCamera(st);
  const [su, sv, sr] = st.spot;
  const spotPos = W(su, sv, G.table.Zf - 0.4);
  const flicker = 0.96 + 0.04 * Math.sin(t * 17.0) * Math.sin(t * 5.3 + 1.0);
  for (const m of [mats.bg, mats.fg]) {
    m.uniforms.spotPos.value.copy(spotPos); m.uniforms.spotR.value = sr; m.uniforms.flicker.value = flicker;
  }
  mats.beam.uniforms.strength.value = st.rays; mats.beam.uniforms.time.value = t;

  R.setRenderTarget(rt); R.clear(); R.render(scn, cam);

  mats.dof.uniforms.focus.value = st.focusDist; mats.dof.uniforms.aperture.value = st.ap;
  pass(mats.dof, rtDof);
  mats.dust.uniforms.time.value = t; mats.dust.uniforms.focus.value = st.focusDist; mats.dust.uniforms.aperture.value = st.ap;
  R.setRenderTarget(rtDof); R.render(dustScn, cam);

  pass(mats.bright, rtBloomA);
  for (let i = 0; i < 2; i++) {
    mats.blur.uniforms.tCol.value = rtBloomA.texture; mats.blur.uniforms.dir.value.set(2.5 / rtBloomA.width, 0); pass(mats.blur, rtBloomB);
    mats.blur.uniforms.tCol.value = rtBloomB.texture; mats.blur.uniforms.dir.value.set(0, 2.5 / rtBloomA.height); pass(mats.blur, rtBloomA);
  }
  const f = mats.final.uniforms;
  f.exposure.value = st.exposure; f.flash.value = clamp(st.flash); f.fade.value = clamp(st.fade); f.ca.value = st.ca; f.lb.value = st.lb;
  f.seed.value = frameSeed; f.weight.value = weight;
  pass(mats.final, accRT, false);
  return st;
}

// motion blur: average sub-frames over a 180-degree shutter; more samples on fast moves
window.renderFrame = (t) => {
  const fps = CFG.fps;
  const a = stateAt(t - 0.25 / fps), b = stateAt(t + 0.25 / fps);
  const speed = a.pos.distanceTo(b.pos) + a.look.distanceTo(b.look) * 0.3 + Math.abs(a.fov - b.fov) * 0.02 + Math.abs(a.roll - b.roll) * 0.01;
  const n = CFG.preview ? 1 : clamp(Math.ceil(speed * 40), 1, CFG.maxSub || 6);
  R.setRenderTarget(accRT); R.setClearColor(0x000000, 1); R.clear();
  let st;
  for (let i = 0; i < n; i++) {
    const ts = n === 1 ? t : t + ((i + 0.5) / n - 0.5) * (0.5 / fps);
    st = renderOne(ts, 1 / n, (t * 1000) % 97);
  }
  pass(mats.copy, null);
  overlay(t);
  return n;
};

// ---------------------------------------------------------------- titles
const CAPS = [
  { id: 'c1', t0: 1.2, t1: 3.7 },
  { id: 'c2', t0: 5.0, t1: 7.6 },
  { id: 'c3', t0: 8.25, t1: 9.8 },
  { id: 'c4', t0: 12.3, t1: 13.8 },
  { id: 'c5', t0: 25.2, t1: 28.4 },
  { id: 'title', t0: 29.6, t1: 32.5 },
];
function overlay(t) {
  for (const c of CAPS) {
    const el = document.getElementById(c.id);
    if (!el) continue;
    const inn = clamp((t - c.t0) / 0.6), out = clamp((c.t1 - t) / 0.5);
    const o = Math.min(inn, out);
    el.style.opacity = o.toFixed(3);
    const blur = (1 - ease.out3(inn)) * 12;
    el.style.filter = `blur(${blur.toFixed(2)}px)`;
    el.style.letterSpacing = `${(0.18 + 0.25 * (1 - ease.out3(inn)) + 0.04 * clamp((t - c.t0) / (c.t1 - c.t0))).toFixed(3)}em`;
  }
}
