/**
 * HandRig — the Stitch-style 21-landmark hand used by both 3D scenes.
 *
 * Visual language (from the SignBridge design system): sky-blue joints,
 * deep-blue wrist, amber fingertips with halo rings, translucent sky bones.
 * Everything is instanced: one hand is 4 draw calls regardless of joints.
 *
 * Sizes are relative to the hand's own scale (wrist -> middle knuckle), so
 * the rig looks the same whether it is a small overlay or a large avatar.
 */

const THREE = window.THREE;

export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

const TIP_IDS = [4, 8, 12, 16, 20];
const PALM_IDS = [0, 5, 9, 13, 17];

// Radii relative to wrist->middle-MCP distance (Stitch proportions).
const JOINT_SIZE = [0.1, 0.067, 0.063, 0.058, 0.054, 0.067, 0.063, 0.058, 0.054, 0.067, 0.063, 0.058, 0.054,
  0.063, 0.058, 0.054, 0.05, 0.058, 0.054, 0.05, 0.046];
const BONE_RADIUS = 0.018;

export const PALETTE = {
  wrist: 0x0284c7,
  joint: 0x0ea5e9,
  tip: 0xf59e0b,
  bone: 0x38bdf8,
  confirmed: 0x0d9488,
  locking: 0xf59e0b,
  tracking: 0x0ea5e9,
};

const UP = new THREE.Vector3(0, 1, 0);

export class HandRig {
  constructor({ boneOpacity = 0.85, withHoldRing = false } = {}) {
    this.group = new THREE.Group();
    this.points = Array.from({ length: 21 }, () => new THREE.Vector3());
    this._m = new THREE.Matrix4();
    this._q = new THREE.Quaternion();
    this._s = new THREE.Vector3();
    this._dir = new THREE.Vector3();
    this._tmpColor = new THREE.Color();
    this._flash = 0;
    this._flashColor = new THREE.Color(PALETTE.confirmed);

    this.baseColors = Array.from({ length: 21 }, (_, i) =>
      new THREE.Color(i === 0 ? PALETTE.wrist : TIP_IDS.includes(i) ? PALETTE.tip : PALETTE.joint));

    this.jointGeo = new THREE.SphereGeometry(1, 20, 14);
    this.jointMat = new THREE.MeshStandardMaterial({ roughness: 0.25, metalness: 0.35, emissive: 0x0369a1, emissiveIntensity: 0.18 });
    this.joints = new THREE.InstancedMesh(this.jointGeo, this.jointMat, 21);
    for (let i = 0; i < 21; i++) this.joints.setColorAt(i, this.baseColors[i]);

    this.boneGeo = new THREE.CylinderGeometry(1, 1, 1, 10, 1, true);
    this.boneGeo.translate(0, 0.5, 0);
    this.boneMat = new THREE.MeshStandardMaterial({ color: PALETTE.bone, roughness: 0.3, metalness: 0.3, transparent: true, opacity: boneOpacity });
    this.bones = new THREE.InstancedMesh(this.boneGeo, this.boneMat, HAND_CONNECTIONS.length);

    this.ringGeo = new THREE.TorusGeometry(1, 0.09, 8, 28);
    this.ringMat = new THREE.MeshBasicMaterial({ color: PALETTE.tip, transparent: true, opacity: 0.6 });
    this.rings = new THREE.InstancedMesh(this.ringGeo, this.ringMat, TIP_IDS.length);

    for (const mesh of [this.joints, this.bones, this.rings]) {
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.frustumCulled = false;
      this.group.add(mesh);
    }

    if (withHoldRing) {
      // Aura ring around the palm. Its arc length shows real hold progress.
      const segments = 96;
      const pts = new Float32Array((segments + 1) * 3);
      for (let i = 0; i <= segments; i++) {
        const a = Math.PI / 2 - (i / segments) * Math.PI * 2;
        pts[i * 3] = Math.cos(a);
        pts[i * 3 + 1] = Math.sin(a);
      }
      this.holdSegments = segments;
      this.holdGeo = new THREE.BufferGeometry();
      this.holdGeo.setAttribute("position", new THREE.BufferAttribute(pts, 3));
      this.holdGeo.setDrawRange(0, 0);
      this.holdMat = new THREE.LineBasicMaterial({ color: PALETTE.locking, transparent: true, opacity: 0.95 });
      this.holdLine = new THREE.Line(this.holdGeo, this.holdMat);
      this.holdLine.frustumCulled = false;

      const trackGeo = new THREE.BufferGeometry();
      trackGeo.setAttribute("position", new THREE.BufferAttribute(pts.slice(), 3));
      this.trackGeo = trackGeo;
      this.trackMat = new THREE.LineBasicMaterial({ color: PALETTE.tracking, transparent: true, opacity: 0.35 });
      this.trackLine = new THREE.Line(trackGeo, this.trackMat);
      this.trackLine.frustumCulled = false;
      this.group.add(this.trackLine, this.holdLine);
    }
  }

  set visible(v) { this.group.visible = v; }
  get visible() { return this.group.visible; }

  /** points: array of 21 THREE.Vector3 in scene space. */
  setLandmarks(points) {
    for (let i = 0; i < 21; i++) this.points[i].copy(points[i]);
    const p = this.points;
    const scale = Math.max(1e-4, p[0].distanceTo(p[9]));

    for (let i = 0; i < 21; i++) {
      const r = JOINT_SIZE[i] * scale;
      this._m.makeScale(r, r, r).setPosition(p[i]);
      this.joints.setMatrixAt(i, this._m);
    }

    HAND_CONNECTIONS.forEach(([a, b], idx) => {
      this._dir.subVectors(p[b], p[a]);
      const len = this._dir.length();
      if (len > 1e-6) this._q.setFromUnitVectors(UP, this._dir.multiplyScalar(1 / len));
      const r = BONE_RADIUS * scale;
      this._s.set(r, len, r);
      this._m.compose(p[a], this._q, this._s);
      this.bones.setMatrixAt(idx, this._m);
    });

    TIP_IDS.forEach((id, idx) => {
      const r = JOINT_SIZE[id] * scale * 1.6;
      this._m.makeScale(r, r, r).setPosition(p[id]);
      this.rings.setMatrixAt(idx, this._m);
    });

    this.joints.instanceMatrix.needsUpdate = true;
    this.bones.instanceMatrix.needsUpdate = true;
    this.rings.instanceMatrix.needsUpdate = true;

    if (this.holdLine) {
      const c = this._dir.set(0, 0, 0);
      PALM_IDS.forEach((id) => c.add(p[id]));
      c.multiplyScalar(1 / PALM_IDS.length);
      const radius = scale * 1.35;
      for (const line of [this.holdLine, this.trackLine]) {
        line.position.copy(c);
        line.scale.setScalar(radius);
      }
    }
  }

  /** progress 0..1; state: "tracking" | "locking" | "confirmed" */
  setHold(progress, state = "locking") {
    if (!this.holdLine) return;
    const n = Math.round(Math.max(0, Math.min(1, progress)) * this.holdSegments);
    this.holdGeo.setDrawRange(0, n > 0 ? n + 1 : 0);
    this.holdMat.color.setHex(PALETTE[state] ?? PALETTE.locking);
    this.trackMat.color.setHex(state === "confirmed" ? PALETTE.confirmed : PALETTE.tracking);
  }

  flash(hex = PALETTE.confirmed) {
    this._flashColor.setHex(hex);
    this._flash = 1;
  }

  /** Advances time-based effects. Returns true while still animating. */
  update(dt) {
    if (this._flash <= 0) return false;
    this._flash = Math.max(0, this._flash - dt * 1.6);
    for (let i = 0; i < 21; i++) {
      this._tmpColor.copy(this.baseColors[i]).lerp(this._flashColor, this._flash);
      this.joints.setColorAt(i, this._tmpColor);
    }
    this.joints.instanceColor.needsUpdate = true;
    return this._flash > 0;
  }

  dispose() {
    this.group.removeFromParent?.();
    [this.jointGeo, this.boneGeo, this.ringGeo, this.holdGeo, this.trackGeo].forEach((g) => g?.dispose());
    [this.jointMat, this.boneMat, this.ringMat, this.holdMat, this.trackMat].forEach((m) => m?.dispose());
  }
}

/** Shared renderer bootstrap with sizing tied to the canvas' box (not the window). */
export function createRenderer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "high-performance" });
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.setClearColor(0x000000, 0);
  return renderer;
}

export function addStudioLights(scene) {
  // Stitch lighting: soft sky ambient, blue key, sky rim, warm gold top.
  scene.add(new THREE.AmbientLight(0xe0f2fe, 1.1));
  const key = new THREE.PointLight(0x7dd3fc, 1.4, 30);
  key.position.set(3, 4, 3);
  const rim = new THREE.PointLight(0x0ea5e9, 1.0, 20);
  rim.position.set(-3, -2, -2);
  const gold = new THREE.PointLight(0xfbbf24, 0.8, 12);
  gold.position.set(0, 3, 2);
  scene.add(key, rim, gold);
}

export const prefersReducedMotion = () =>
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

/** Lowers GPU cost on small / low-power screens. */
export function pixelRatioCap() {
  const small = Math.min(window.innerWidth, window.innerHeight) < 700;
  return Math.min(window.devicePixelRatio || 1, small ? 1.5 : 2);
}
