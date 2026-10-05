/**
 * Live recognition client for /ws/live.
 *
 * Frame pump: capture a small mirrored JPEG, send it, and wait for the
 * matching frame_result before sending the next one. The frame rate thus
 * follows real server capacity (no queue build-up, no stale results).
 */

const FRAME_WIDTH = 320;
const MIN_FRAME_INTERVAL_MS = 50; // cap at ~20 fps
const RESULT_TIMEOUT_MS = 2000;
const JPEG_QUALITY = 0.7;

export class LiveClient {
  constructor({ onMessage, onStatus }) {
    this.onMessage = onMessage;
    this.onStatus = onStatus;
    this.ws = null;
    this.retry = 0;
    this.config = {};
    this.pump = { camera: null, inflight: false, sentAt: 0, timer: 0, running: false };
    this.stats = { fps: 0, latencyMs: 0, serverMs: 0, _frames: 0, _windowStart: performance.now() };
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: false });
    document.addEventListener("visibilitychange", () => this._schedule(0));
  }

  get open() {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    this.onStatus("connecting");
    const ws = new WebSocket(`${proto}//${location.host}/ws/live`);
    this.ws = ws;

    ws.onopen = () => {
      this.retry = 0;
      this.pump.inflight = false;
      this.send({ action: "configure", ...this.config });
      this.onStatus("open");
      this._schedule(0);
    };
    ws.onmessage = (event) => {
      let msg;
      try { msg = JSON.parse(event.data); } catch { return; }
      if (msg.type === "frame_result" || (msg.type === "error" && msg.code === "bad_frame")) {
        this._frameDone(msg);
      }
      this.onMessage(msg);
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.onStatus("closed");
      const delay = Math.min(10000, 1000 * 2 ** this.retry++);
      setTimeout(() => this.connect(), delay);
    };
    ws.onerror = () => ws.close();
  }

  send(payload) {
    if (!this.open) return false;
    this.ws.send(JSON.stringify(payload));
    return true;
  }

  configure(config) {
    Object.assign(this.config, config);
    this.send({ action: "configure", ...config });
  }

  startFrames(camera) {
    this.pump.camera = camera;
    this.pump.running = true;
    this._schedule(0);
  }

  stopFrames() {
    this.pump.running = false;
    clearTimeout(this.pump.timer);
  }

  _schedule(delay) {
    clearTimeout(this.pump.timer);
    if (!this.pump.running || document.hidden) return;
    this.pump.timer = setTimeout(() => this._tick(), delay);
  }

  _tick() {
    const p = this.pump;
    if (!p.running) return;
    const now = performance.now();
    if (p.inflight && now - p.sentAt < RESULT_TIMEOUT_MS) return this._schedule(100);
    if (!this.open || !p.camera?.ready) return this._schedule(250);

    const video = p.camera.video;
    const w = FRAME_WIDTH;
    const h = Math.round((FRAME_WIDTH * video.videoHeight) / video.videoWidth);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    // Mirror like a selfie view (matches the desktop runner & training convention).
    this.ctx.setTransform(-1, 0, 0, 1, w, 0);
    this.ctx.drawImage(video, 0, 0, w, h);
    const image = this.canvas.toDataURL("image/jpeg", JPEG_QUALITY);

    p.inflight = this.send({ action: "frame", image });
    p.sentAt = now;
    if (!p.inflight) this._schedule(250);
  }

  _frameDone(msg) {
    const p = this.pump;
    if (!p.inflight) return;
    p.inflight = false;
    const now = performance.now();
    const s = this.stats;
    s.latencyMs = s.latencyMs ? s.latencyMs * 0.8 + (now - p.sentAt) * 0.2 : now - p.sentAt;
    if (msg.processing_ms != null) s.serverMs = s.serverMs ? s.serverMs * 0.8 + msg.processing_ms * 0.2 : msg.processing_ms;
    s._frames++;
    if (now - s._windowStart >= 1000) {
      s.fps = (s._frames * 1000) / (now - s._windowStart);
      s._frames = 0;
      s._windowStart = now;
    }
    this._schedule(Math.max(0, MIN_FRAME_INTERVAL_MS - (now - p.sentAt)));
  }
}
