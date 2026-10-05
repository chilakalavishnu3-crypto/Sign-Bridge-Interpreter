/** Webcam lifecycle with explicit, user-explainable failure states. */

const ERRORS = {
  NotAllowedError: {
    title: "Camera access is blocked",
    body: "Allow camera access in the browser's address bar, then try again. You can also tap signs instead.",
  },
  NotFoundError: {
    title: "No camera found",
    body: "Connect a camera to this counter, or tap signs in the sign guide instead.",
  },
  NotReadableError: {
    title: "Camera is busy",
    body: "Another app is using the camera. Close it and try again.",
  },
  insecure: {
    title: "Camera needs a secure page",
    body: "Open SignBridge from http://localhost or over HTTPS to use the camera.",
  },
};

export class Camera {
  constructor(video) {
    this.video = video;
    this.stream = null;
  }

  get ready() {
    return !!this.stream && this.video.readyState >= 2 && this.video.videoWidth > 0;
  }

  async start() {
    this.stop();
    if (!navigator.mediaDevices?.getUserMedia) {
      throw Object.assign(new Error("insecure"), { info: ERRORS.insecure });
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
    } catch (err) {
      throw Object.assign(err, { info: ERRORS[err.name] || { title: "Camera unavailable", body: err.message || "Unknown camera error." } });
    }
    this.video.srcObject = this.stream;
    await this.video.play().catch(() => {});
    if (this.video.readyState < 2) {
      await new Promise((resolve) => this.video.addEventListener("loadeddata", resolve, { once: true }));
    }
    this.stream.getVideoTracks()[0]?.addEventListener("ended", () => this.onEnded?.());
  }

  stop() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }
}
