/**
 * "Ask AI" — opt-in sign identification with an OpenAI vision model.
 *
 * Flow: consent (citizen must agree) -> 3-2-1 -> 4 frames over ~1 s,
 * cropped around the hands the server last detected (keeps faces out when
 * possible) -> POST /api/ai/identify-sign -> suggestion staff can accept.
 * Nothing is sent without the consent checkbox; nothing is added
 * automatically.
 */

import { $, el, icon, toast, wireDialog } from "./ui.js";

const FRAMES = 4;
const FRAME_GAP_MS = 300;
const OUT_SIZE = 384;

export function initAskAI({ camera, getHands, getPack, onAccept, onSpeakText, onTeach }) {
  const dialog = $("#ai-dialog");
  const ctl = wireDialog(dialog, { onClose: reset });
  const video = $("#ai-video");
  const consent = $("#ai-consent-check");
  const captureBtn = $("#btn-ai-capture");
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  let timer = 0;
  let aborted = false;

  function show(step) {
    for (const id of ["ai-consent", "ai-capture", "ai-waiting", "ai-result", "ai-error"]) {
      $(`#${id}`).hidden = id !== step;
    }
  }

  function reset() {
    aborted = true;
    clearInterval(timer);
    video.srcObject = null;
    consent.checked = false;
    captureBtn.disabled = true;
    show("ai-consent");
  }

  function open(trigger) {
    if (!camera.stream) return toast("Turn the camera on first.", "error");
    reset();
    aborted = false;
    ctl.open(trigger);
  }

  /** Crop box (in the mirrored frame) around detected hands, or the full frame. */
  function cropBox(vw, vh) {
    const hands = getHands();
    if (!hands?.length) return { x: 0, y: 0, w: vw, h: vh };
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    for (const h of hands) for (const p of h.landmarks) {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    }
    const cx = ((minX + maxX) / 2) * vw;
    const cy = ((minY + maxY) / 2) * vh;
    const side = Math.max((maxX - minX) * vw, (maxY - minY) * vh) * 1.8;
    const size = Math.min(Math.max(side, Math.min(vw, vh) * 0.45), Math.min(vw, vh));
    return {
      x: Math.max(0, Math.min(vw - size, cx - size / 2)),
      y: Math.max(0, Math.min(vh - size, cy - size / 2)),
      w: size,
      h: size,
    };
  }

  function grab() {
    const v = camera.video;
    const vw = v.videoWidth;
    const vh = v.videoHeight;
    const box = cropBox(vw, vh);
    const scale = OUT_SIZE / Math.max(box.w, box.h);
    canvas.width = Math.round(box.w * scale);
    canvas.height = Math.round(box.h * scale);
    // Mirror (selfie view) to match the landmark coordinates.
    ctx.setTransform(-scale, 0, 0, scale, canvas.width, 0);
    ctx.drawImage(v, vw - box.x - box.w, box.y, box.w, box.h, 0, 0, box.w, box.h);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return canvas.toDataURL("image/jpeg", 0.8);
  }

  async function capture() {
    show("ai-capture");
    video.srcObject = camera.stream;
    video.play().catch(() => {});
    const counter = $("#ai-countdown");
    for (let n = 3; n > 0; n--) {
      counter.textContent = String(n);
      await new Promise((r) => { timer = setTimeout(r, 700); });
      if (aborted) return;
    }
    counter.textContent = "";
    $("#ai-capture-hint").textContent = "Capturing… keep holding the sign.";
    const images = [];
    for (let i = 0; i < FRAMES; i++) {
      images.push(grab());
      await new Promise((r) => { timer = setTimeout(r, FRAME_GAP_MS); });
      if (aborted) return;
    }
    video.srcObject = null;
    $("#ai-capture-hint").textContent = "Citizen: make the sign and hold it.";
    ask(images);
  }

  async function ask(images) {
    show("ai-waiting");
    try {
      const res = await fetch("/api/ai/identify-sign", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images, consent: true, pack: getPack() }),
      });
      const data = await res.json().catch(() => ({}));
      if (aborted) return;
      if (!res.ok) throw new Error(data.detail || "The AI request failed.");
      showResult(data);
    } catch (err) {
      if (aborted) return;
      $("#ai-error-text").textContent = err.message;
      show("ai-error");
    }
  }

  function showResult(r) {
    const unclear = r.sign === "UNCLEAR";
    $("#ai-sign").textContent = unclear ? "No clear sign" : r.sign;
    const meta = $("#ai-meta");
    if (unclear) {
      meta.textContent = r.hands_seen === 0
        ? "No hand was visible in the pictures — the AI was not asked. Keep the hands inside the camera view and try again."
        : "The AI couldn't recognize a clear sign. Try again, holding the sign steady.";
    } else {
      // Model self-confidence is not calibrated: show it only as a rough level.
      const level = r.confidence >= 0.75 ? "high" : r.confidence >= 0.45 ? "mid" : "low";
      const label = { high: "Likely", mid: "Possible", low: "Weak guess" }[level];
      meta.replaceChildren(
        el("span.likelihood", { dataset: { level } }, label),
        r.in_vocabulary ? "Known sign" : "Not in this counter's vocabulary",
      );
    }
    $("#ai-reason").textContent = r.reason && !unclear ? `Why: ${r.reason}` : "";

    $("#ai-alternatives").replaceChildren(...(r.alternatives || []).map((alt) => el("button.phrase-btn", {
      type: "button",
      onclick: () => showResult({ ...r, sign: alt.sign, in_vocabulary: alt.in_vocabulary, alternatives: [], reason: "Chosen from the AI's alternatives." }),
    }, el("span.phrase-gloss", {}, "Or maybe"), el("span.phrase-text", {}, alt.sign))));

    const actions = [];
    if (!unclear && r.in_vocabulary) {
      actions.push(el("button.btn.btn-primary.btn-lg", {
        type: "button",
        onclick: () => { onAccept(r.sign); ctl.close(); },
      }, icon("i-plus"), `Add ${r.sign}`));
    } else if (!unclear) {
      actions.push(
        el("button.btn.btn-primary.btn-lg", { type: "button", onclick: () => { onSpeakText(r.sign); ctl.close(); } }, icon("i-volume"), "Show & speak"),
        el("button.btn.btn-outline.btn-lg", { type: "button", onclick: () => { ctl.close(); onTeach(r.sign); } }, "Teach this sign"),
      );
    }
    actions.push(el("button.btn.btn-ghost.btn-lg", { type: "button", onclick: () => { aborted = false; capture(); } }, icon("i-refresh"), "Try again"));
    $("#ai-actions").replaceChildren(...actions);
    show("ai-result");
  }

  consent.onchange = () => { captureBtn.disabled = !consent.checked; };
  captureBtn.onclick = () => { if (consent.checked) { aborted = false; capture(); } };
  $("#btn-ai-retry").onclick = () => { aborted = false; capture(); };

  return { open };
}
