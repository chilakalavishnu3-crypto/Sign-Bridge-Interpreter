/**
 * Live hand overlay: draws the landmarks MediaPipe actually detected on the
 * server on top of the (mirrored) camera feed.
 *
 *  - Alignment: landmark coords are normalized to the frame we sent; we map
 *    them through the same object-fit: cover crop the <video> uses.
 *  - Smoothing: server results arrive at ~15-25 fps; the render loop eases
 *    toward the latest result every display frame, so motion stays fluid.
 *  - Honest state: the palm ring is the real hold progress from the server,
 *    a confirmed sign flashes the joints teal.
 *  - Cost: renders only while hands are visible or an effect is running.
 */

import { HandRig, PALETTE, addStudioLights, createRenderer, pixelRatioCap, prefersReducedMotion } from "./hand-rig.js";

const THREE = window.THREE;
const LABELS = ["Left", "Right"];

export class HandOverlayScene {
  constructor(canvas, video) {
    this.canvas = canvas;
    this.video = video;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
    this.camera.position.set(0, 0, 3);
    this.renderer = createRenderer(canvas);
    addStudioLights(this.scene);

    this.hands = {};
    this.targets = {};
    for (const label of LABELS) {
      const rig = new HandRig({ withHoldRing: true, boneOpacity: 0.9 });
      rig.visible = false;
      this.scene.add(rig.group);
      this.hands[label] = rig;
      this.targets[label] = { points: Array.from({ length: 21 }, () => new THREE.Vector3()), fresh: false, seen: 0 };
    }

    this.burst = this._createBurst();
    this.scene.add(this.burst.points);

    this.reduced = prefersReducedMotion();
    this.clock = new THREE.Clock();
    this._running = false;
    this._activeUntil = 0;
    this._raf = 0;
    this._loop = this._loop.bind(this);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
  }

  resize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setPixelRatio(pixelRatioCap());
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this._wake();
  }

  /** Size of the displayed video in scene units at z=0 (object-fit: cover). */
  _videoPlane() {
    const d = this.camera.position.z;
    const viewH = 2 * d * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const viewW = viewH * this.camera.aspect;
    const va = (this.video.videoWidth || 4) / (this.video.videoHeight || 3);
    return this.camera.aspect > va ? { w: viewW, h: viewW / va } : { w: viewH * va, h: viewH };
  }

  /**
   * hands: [{label, landmarks:[{x,y,z}...21]}] in the mirrored frame we sent.
   */
  setHands(hands) {
    const plane = this._videoPlane();
    const now = performance.now();
    const seen = new Set();
    for (const hand of hands || []) {
      if (!hand.landmarks || hand.landmarks.length !== 21) continue;
      let label = LABELS.includes(hand.label) ? hand.label : "Right";
      if (seen.has(label)) label = label === "Left" ? "Right" : "Left";
      seen.add(label);
      const target = this.targets[label];
      hand.landmarks.forEach((lm, i) => {
        target.points[i].set((lm.x - 0.5) * plane.w, (0.5 - lm.y) * plane.h, -(lm.z || 0) * plane.w * 0.6);
      });
      const rig = this.hands[label];
      if (!rig.visible || this.reduced) {
        rig.setLandmarks(target.points); // snap on first sight
      }
      target.seen = now;
      rig.visible = true;
    }
    for (const label of LABELS) {
      if (!seen.has(label)) this.hands[label].visible = false;
    }
    this._wake();
  }

  /** Real hold progress (0..1) from the server, applied to visible hands. */
  setHold(progress, state) {
    for (const label of LABELS) this.hands[label].setHold(progress, state);
    this._wake();
  }

  confirm() {
    let origin = null;
    for (const label of LABELS) {
      const rig = this.hands[label];
      if (!rig.visible) continue;
      rig.flash(PALETTE.confirmed);
      origin = origin || rig.points[9];
    }
    if (!this.reduced && origin) this._fireBurst(origin);
    this._wake(900);
  }

  _createBurst() {
    const count = 48;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(count * 3);
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const vel = Array.from({ length: count }, () => new THREE.Vector3());
    const mat = new THREE.PointsMaterial({ color: PALETTE.confirmed, size: 0.035, transparent: true, opacity: 0, depthWrite: false });
    const points = new THREE.Points(geo, mat);
    points.frustumCulled = false;
    return { points, vel, age: 1 };
  }

  _fireBurst(origin) {
    const attr = this.burst.points.geometry.attributes.position;
    this.burst.vel.forEach((v, i) => {
      v.set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).normalize().multiplyScalar(0.6 + Math.random() * 0.6);
      attr.setXYZ(i, origin.x, origin.y, origin.z);
    });
    attr.needsUpdate = true;
    this.burst.age = 0;
  }

  _wake(ms = 400) {
    this._activeUntil = Math.max(this._activeUntil, performance.now() + ms);
    if (!this._running) {
      this._running = true;
      this.clock.getDelta();
      this._raf = requestAnimationFrame(this._loop);
    }
  }

  _loop() {
    const dt = Math.min(0.05, this.clock.getDelta());
    let animating = false;

    // Ease each rig toward its latest detected pose (frame-rate independent).
    const k = 1 - Math.exp(-dt * 18);
    for (const label of LABELS) {
      const rig = this.hands[label];
      if (!rig.visible) continue;
      const target = this.targets[label].points;
      let moving = false;
      for (let i = 0; i < 21; i++) {
        rig.points[i].lerp(target[i], k);
        if (rig.points[i].distanceToSquared(target[i]) > 1e-7) moving = true;
      }
      rig.setLandmarks(rig.points);
      if (rig.update(dt)) animating = true;
      if (moving) animating = true;
    }

    const b = this.burst;
    if (b.age < 1) {
      b.age += dt * 1.4;
      const attr = b.points.geometry.attributes.position;
      b.vel.forEach((v, i) => {
        attr.setXYZ(i, attr.getX(i) + v.x * dt, attr.getY(i) + v.y * dt, attr.getZ(i) + v.z * dt);
      });
      attr.needsUpdate = true;
      b.points.material.opacity = Math.max(0, 1 - b.age);
      animating = true;
    } else {
      b.points.material.opacity = 0;
    }

    this.renderer.render(this.scene, this.camera);

    const anyVisible = LABELS.some((l) => this.hands[l].visible);
    if (animating || anyVisible || performance.now() < this._activeUntil) {
      this._raf = requestAnimationFrame(this._loop);
    } else {
      this._running = false;
    }
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    this.resizeObserver.disconnect();
    for (const label of LABELS) this.hands[label].dispose();
    this.burst.points.geometry.dispose();
    this.burst.points.material.dispose();
    this.renderer.dispose();
  }
}
