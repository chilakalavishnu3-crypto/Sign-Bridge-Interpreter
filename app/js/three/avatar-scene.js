/**
 * Clerk-reply signing hand.
 *
 * Builds the Stitch landmark hand from a base pose and drives it with
 * forward kinematics: per-finger curl + wrist orientation + an optional
 * small motion cue (wave, nod, glide…) per sign. Playback is clock-driven
 * from the render loop, so starting a new reply cleanly cancels the old one.
 */

import { HandRig, addStudioLights, createRenderer, pixelRatioCap, prefersReducedMotion } from "./hand-rig.js";

const THREE = window.THREE;

// Base pose: palm facing the viewer, fingers up (Stitch design coordinates).
const BASE = [
  [0, -1.3, 0],
  [-0.35, -0.9, 0.15], [-0.68, -0.5, 0.22], [-0.92, -0.1, 0.2], [-1.15, 0.25, 0.16],
  [-0.38, -0.15, 0.06], [-0.42, 0.48, 0.08], [-0.44, 0.95, 0.07], [-0.45, 1.38, 0.05],
  [-0.04, -0.1, 0], [-0.05, 0.58, 0], [-0.05, 1.1, 0], [-0.05, 1.58, 0],
  [0.3, -0.15, -0.06], [0.34, 0.48, -0.06], [0.36, 0.96, -0.06], [0.38, 1.38, -0.06],
  [0.6, -0.28, -0.12], [0.7, 0.28, -0.12], [0.76, 0.72, -0.12], [0.8, 1.08, -0.12],
].map(([x, y, z]) => new THREE.Vector3(x, y, z));

const FINGERS = [
  [1, 2, 3, 4], // thumb
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20],
];
const JOINT_GAIN = [1.25, 1.55, 1.05]; // radians at full curl: knuckle, middle, end

const THUMB_AXIS = (() => {
  const dir = new THREE.Vector3().subVectors(BASE[4], BASE[1]).normalize();
  const toward = new THREE.Vector3(0.7, -0.2, 0.7).normalize();
  return new THREE.Vector3().crossVectors(dir, toward).normalize();
})();
const FINGER_AXIS = new THREE.Vector3(1, 0, 0);

const P = (rx, ry, rz, curls, motion = null) => ({ rot: [rx, ry, rz], curls, motion });
const INDEX_UP = [0.9, 0, 0.95, 0.95, 0.95];
const OPEN = [0, 0, 0, 0, 0];
const FIST = [0.55, 0.95, 0.95, 0.95, 0.95];

export const POSTURES = {
  0: P(0.1, 0.25, 0, [0.8, 0.85, 0.9, 0.9, 0.9]),
  1: P(0, 0, 0, INDEX_UP),
  2: P(0, 0, 0, [0.95, 0, 0, 0.95, 0.95]),
  3: P(0, 0, 0, [0, 0, 0, 0.95, 0.95]),
  4: P(0, 0, 0, [0.95, 0, 0, 0, 0]),
  5: P(0, 0, 0, OPEN),
  HELLO: P(0, 0, 0.1, OPEN, "wave"),
  "THANK YOU": P(-0.3, 0, 0, OPEN, "nod"),
  YES: P(0, 0, 0, FIST, "nod"),
  NO: P(0, 0, 0, [0.9, 0, 0, 0.95, 0.95], "shake"),
  HELP: P(0, 0, 0, [0, 0.95, 0.95, 0.95, 0.95], "rise"),
  WHERE: P(0, 0, 0, INDEX_UP, "shake"),
  WHEN: P(0, 0, 0, INDEX_UP, "circle"),
  WATER: P(0, 0, 0, [0.9, 0, 0, 0, 0.95], "nod"),
  TOILET: P(0, 0, 0, [0.5, 0.95, 0.95, 0.95, 0.95], "shake"),
  MONEY: P(-0.3, 0.2, 0, [0.45, 0.45, 0.45, 0.95, 0.95], "rub"),
  DOCTOR: P(-0.85, 0, 0, [0.9, 0, 0, 0, 0.95]),
  TICKET: P(-0.2, 0.2, 0, [0.9, 0.5, 0.5, 0.95, 0.95], "nod"),
  TRAIN: P(-0.65, -0.35, 0, [0.9, 0, 0, 0.9, 0.9], "glide"),
  PLATFORM: P(-1.25, 0, 0, OPEN, "glide"),
  PAIN: P(0, 0, 0, INDEX_UP, "twist"),
  MEDICINE: P(-0.3, 0.25, 0, [0.55, 0.55, 0.9, 0.9, 0.9], "rub"),
  FEVER: P(0.25, 0, 0, OPEN),
  HEAD: P(0, 0, 0.6, INDEX_UP),
  STOMACH: P(-1.0, 0, 0, OPEN, "circle"),
  WAIT: P(0, 0, 0, OPEN),
  STOP: P(0, 0, 0, OPEN, "push"),
  HERE: P(-0.95, 0, 0, INDEX_UP, "nod"),
  COME: P(-0.35, 0.25, 0.15, [0.35, 0.35, 0.35, 0.35, 0.35], "beckon"),
  TOMORROW: P(0.25, -0.45, 0, [0.2, 0, 0.95, 0.95, 0.95], "push"),
  GO: P(-1.2, 0, 0, INDEX_UP, "push"),
  ROOM: P(0, 1.2, 0, OPEN, "nod"),
  DAY: P(0, 0, 0.3, INDEX_UP, "twist"),
};
// One-handed digits 6-9 (thumb touches pinky / ring / middle / index).
Object.assign(POSTURES, {
  6: P(0, 0, 0, [0.6, 0, 0, 0, 0.75]),
  7: P(0, 0, 0, [0.6, 0, 0, 0.75, 0]),
  8: P(0, 0, 0, [0.6, 0, 0.75, 0, 0]),
  9: P(0, 0, 0, [0.6, 0.75, 0, 0, 0]),
});

/**
 * One-handed fingerspelling alphabet (approximate handshapes: this rig has
 * finger bends and wrist rotation, not finger spread or crossing). The
 * caption always shows the exact letters being spelled.
 */
export const LETTERS = {
  A: P(0, 0, 0, [0.15, 1, 1, 1, 1]),
  B: P(0, 0, 0, [0.95, 0, 0, 0, 0]),
  C: P(0, 0.7, 0, [0.4, 0.5, 0.5, 0.5, 0.5]),
  D: P(0, 0, 0, [0.6, 0, 0.75, 0.75, 0.75]),
  E: P(0, 0, 0, [0.95, 0.75, 0.75, 0.75, 0.75]),
  F: P(0, 0, 0, [0.6, 0.75, 0, 0, 0]),
  G: P(0, 0, 1.35, [0.3, 0, 1, 1, 1]),
  H: P(0, 0, 1.35, [0.9, 0, 0, 1, 1]),
  I: P(0, 0, 0, [0.9, 1, 1, 1, 0]),
  J: P(0, 0, 0, [0.9, 1, 1, 1, 0], "twist"),
  K: P(0, 0.3, 0, [0.3, 0, 0, 1, 1]),
  L: P(0, 0, 0, [0, 0, 1, 1, 1]),
  M: P(0.3, 0, 0, [0.95, 0.85, 0.85, 0.85, 1]),
  N: P(0.3, 0, 0, [0.95, 0.85, 0.85, 1, 1]),
  O: P(0, 0.4, 0, [0.6, 0.65, 0.65, 0.65, 0.65]),
  P: P(0, 0.3, 2.5, [0.3, 0, 0, 1, 1]),
  Q: P(0, 0, 2.6, [0.3, 0, 1, 1, 1]),
  R: P(0, 0.25, 0, [0.9, 0, 0.1, 1, 1]),
  S: P(0, 0, 0, [0.7, 1, 1, 1, 1]),
  T: P(0, 0, 0, [0.45, 0.9, 1, 1, 1]),
  U: P(0, 0, 0, [0.9, 0, 0, 1, 1]),
  V: P(0, 0, 0.15, [0.95, 0, 0, 0.95, 0.95]),
  W: P(0, 0, 0, [0.9, 0, 0, 0, 1]),
  X: P(0, 0, 0, [0.9, 0.55, 1, 1, 1]),
  Y: P(0, 0, 0, [0, 1, 1, 1, 0]),
  Z: P(0, 0, 0, INDEX_UP, "glide"),
};

const REST = P(0, 0, 0, [0.15, 0.05, 0.05, 0.08, 0.12]);

export const SIGNABLE = new Set(Object.keys(POSTURES));

const CLIP_SECONDS = 1.2;
const DURATION = { sign: 1.2, letter: 0.6, pose: 1.4, gap: 0.35 };

/**
 * Converts a taught sign's normalized 126-feature pose into avatar space:
 * features are wrist-relative, image-oriented (y down) and scaled to the
 * wrist->middle-knuckle length. Uses the right-hand slot, else the left.
 */
export function poseFromFeatures(features) {
  if (!Array.isArray(features) || features.length !== 126) return null;
  const right = features.slice(63);
  const slot = right.some((v) => v !== 0) ? right : features.slice(0, 63);
  if (!slot.some((v) => v !== 0)) return null;
  const k = BASE[0].distanceTo(BASE[9]);
  const pts = [];
  for (let i = 0; i < 21; i++) {
    pts.push(new THREE.Vector3(slot[i * 3] * k, -slot[i * 3 + 1] * k, -slot[i * 3 + 2] * k).add(BASE[0]));
  }
  return pts;
}

export class SigningAvatarScene {
  constructor(canvas) {
    this.canvas = canvas;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.1, 50);
    this.camera.position.set(0, 0.1, 6.4);
    this.renderer = createRenderer(canvas);
    addStudioLights(this.scene);

    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.rig = new HandRig({ boneOpacity: 0.85 });
    this.root.add(this.rig.group);

    // Floor grid + aura ring from the design system.
    const grid = new THREE.GridHelper(4.2, 14, 0x0ea5e9, 0xdbeafe);
    grid.position.y = -1.75;
    grid.material.transparent = true;
    grid.material.opacity = 0.7;
    this.scene.add(grid);
    this.aura = new THREE.Mesh(
      new THREE.TorusGeometry(1.45, 0.014, 8, 64),
      new THREE.MeshBasicMaterial({ color: 0x0ea5e9, transparent: true, opacity: 0.35 }),
    );
    this.aura.rotation.x = Math.PI / 2;
    this.aura.position.y = -1.6;
    this.scene.add(this.aura);
    this.grid = grid;

    this.curls = [...REST.curls];
    this.rot = new THREE.Vector3(...REST.rot);
    this.target = REST;
    this.points = BASE.map((p) => p.clone());
    this._posePoints = null;
    this._poseWeight = 0;
    this._tmp = new THREE.Vector3();

    this.sequence = null;
    this._timers = [];
    this.reduced = prefersReducedMotion();
    this.clock = new THREE.Clock();
    this.t = 0;
    this._visible = true;
    this._raf = 0;
    this._loop = this._loop.bind(this);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.intersection = new IntersectionObserver(([entry]) => {
      this._visible = entry.isIntersecting;
      if (this._visible) this._start();
    });
    this.intersection.observe(canvas);
    document.addEventListener("visibilitychange", () => !document.hidden && this._start());

    this.resize();
    this._applyPose();
    this._start();
  }

  resize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setPixelRatio(pixelRatioCap());
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // Keep the whole hand in frame on narrow canvases.
    this.camera.position.z = w / h < 0.9 ? 7.6 : 6.4;
    this.camera.updateProjectionMatrix();
    this._start?.();
  }

  /**
   * Play a list of sign names. Steps are scheduled on the wall clock, so
   * progress stays correct even when rendering is throttled or off-screen;
   * the render loop only animates toward the current target posture.
   */
  play(sequence, { onStep, onDone } = {}) {
    if (!sequence?.length) return;
    // Items: "SIGN" | {kind:"sign"|"letter"|"pose"|"gap", sign?, letter?, points?}
    const items = sequence.map((it) => (typeof it === "string" ? { kind: "sign", sign: it } : it));
    this._cancelTimers();
    const prev = this.sequence;
    this.sequence = { items, onStep, onDone };
    prev?.onDone?.(true);
    let at = 0;
    items.forEach((item, idx) => {
      this._timers.push(setTimeout(() => {
        this.target = this._postureFor(item);
        onStep?.(idx, item);
        this._start();
      }, at * 1000));
      at += DURATION[item.kind] ?? CLIP_SECONDS;
    });
    this._timers.push(setTimeout(() => {
      this.sequence = null;
      this.target = REST;
      onDone?.(false);
      this._start();
    }, at * 1000));
    this._start();
  }

  _postureFor(item) {
    if (item.kind === "letter") return LETTERS[item.letter] || POSTURES[item.letter] || REST;
    if (item.kind === "pose" && item.points) return { rot: [0, 0, 0], curls: REST.curls, points: item.points };
    if (item.kind === "gap") return REST;
    return POSTURES[item.sign] || REST;
  }

  stop() {
    this._cancelTimers();
    const seq = this.sequence;
    this.sequence = null;
    this.target = REST;
    seq?.onDone?.(true);
  }

  _cancelTimers() {
    this._timers.forEach(clearTimeout);
    this._timers = [];
  }

  /** Show one sign (dictionary preview). */
  preview(sign) {
    this.play([sign]);
  }

  /** Show a taught sign's recorded hand shape. */
  previewPose(features) {
    const points = poseFromFeatures(features);
    if (points) this.play([{ kind: "pose", points }]);
  }

  _start() {
    // Keep ticking while a reply plays (even off-screen) so steps stay in sync.
    if (!this._raf && (this._visible || this.sequence) && !document.hidden) {
      this.clock.getDelta();
      this._raf = requestAnimationFrame(this._loop);
    }
  }

  _motionOffset(motion, t) {
    if (!motion || this.reduced) return null;
    const s = Math.sin(t * 6);
    switch (motion) {
      case "wave": return { rz: s * 0.25 };
      case "nod": return { rx: s * 0.18 };
      case "shake": return { ry: s * 0.3 };
      case "twist": return { rz: s * 0.35 };
      case "rise": return { y: (Math.sin(t * 3) + 1) * 0.12 };
      case "glide": return { x: Math.sin(t * 3) * 0.25 };
      case "push": return { z: (Math.sin(t * 3) + 1) * 0.2 };
      case "circle": return { x: Math.cos(t * 4) * 0.15, y: Math.sin(t * 4) * 0.15 };
      case "rub": return { curl0: s * 0.12 };
      case "beckon": return { curlAll: (s + 1) * 0.25 };
      default: return null;
    }
  }

  _applyPose(offset = null) {
    const curls = this.curls.map((c, i) => {
      let v = c;
      if (offset?.curlAll) v += offset.curlAll;
      if (i === 0 && offset?.curl0) v += offset.curl0;
      return Math.max(0, Math.min(1.1, v));
    });

    for (let f = 0; f < 5; f++) {
      const chain = FINGERS[f];
      const axis = f === 0 ? THUMB_AXIS : FINGER_AXIS;
      // Walk the chain: each joint's bend accumulates onto later segments.
      let angle = 0;
      this.points[chain[0]].copy(BASE[chain[0]]);
      for (let j = 1; j < chain.length; j++) {
        angle += curls[f] * JOINT_GAIN[j - 1] * (f === 0 ? 0.6 : 1);
        this._tmp.subVectors(BASE[chain[j]], BASE[chain[j - 1]]).applyAxisAngle(axis, angle);
        this.points[chain[j]].copy(this.points[chain[j - 1]]).add(this._tmp);
      }
    }
    this.points[0].copy(BASE[0]);
    // Recorded (taught) poses replace the kinematic hand; blend for smoothness.
    const w = this._poseWeight;
    if (w > 0 && this._posePoints) {
      for (let i = 0; i < 21; i++) this.points[i].lerp(this._posePoints[i], w);
    }
    this.rig.setLandmarks(this.points);

    this.root.rotation.set(
      this.rot.x + (offset?.rx || 0),
      this.rot.y + (offset?.ry || 0),
      this.rot.z + (offset?.rz || 0),
    );
    this.root.position.set(offset?.x || 0, offset?.y || 0, offset?.z || 0);
  }

  _loop() {
    this._raf = 0;
    const dt = Math.min(0.05, this.clock.getDelta());
    this.t += dt;

    const k = this.reduced ? 1 : 1 - Math.exp(-dt * 10);
    this.target.curls.forEach((c, i) => { this.curls[i] += (c - this.curls[i]) * k; });
    this.rot.x += (this.target.rot[0] - this.rot.x) * k;
    this.rot.y += (this.target.rot[1] - this.rot.y) * k;
    this.rot.z += (this.target.rot[2] - this.rot.z) * k;
    if (this.target.points) this._posePoints = this.target.points;
    this._poseWeight += ((this.target.points ? 1 : 0) - this._poseWeight) * k;

    let offset = this._motionOffset(this.target.motion, this.t);
    if (!this.sequence && !this.reduced) {
      // Gentle idle sway, as in the design system's hero hand.
      offset = { ry: Math.sin(this.t * 0.7) * 0.3, rx: 0.06 + Math.cos(this.t * 0.5) * 0.05, y: Math.sin(this.t * 1.2) * 0.04 };
      this.aura.rotation.z = this.t * 0.4;
    }
    this._applyPose(offset);
    if (this._visible) this.renderer.render(this.scene, this.camera);

    const settled = this.reduced && !this.sequence;
    if ((this._visible || this.sequence) && !document.hidden && !settled) {
      this._raf = requestAnimationFrame(this._loop);
    }
  }

  dispose() {
    this._cancelTimers();
    cancelAnimationFrame(this._raf);
    this.resizeObserver.disconnect();
    this.intersection.disconnect();
    this.rig.dispose();
    this.aura.geometry.dispose();
    this.aura.material.dispose();
    this.grid.geometry.dispose();
    this.grid.material.dispose();
    this.renderer.dispose();
  }
}
