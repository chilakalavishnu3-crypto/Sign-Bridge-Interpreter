/**
 * "Teach a sign": record a few seconds of any sign (from any sign language),
 * name it, write what it means — the server stores the landmark samples and
 * recognizes the sign from then on; the meaning is what gets spoken.
 *
 * Recording happens on the server over the existing live socket, so taught
 * signs use exactly the same detection pipeline as recognition.
 */

import { $, el, icon, toast, wireDialog } from "./ui.js";

const MIN_SAMPLES = 12;
const GOOD_SAMPLES = 60;

export function initTeach({ live, camera, getLanguages, getLang, onSignsChanged }) {
  const dialog = $("#teach-dialog");
  const ctl = wireDialog(dialog, { onClose: stopPreview });
  const video = $("#teach-video");
  const preview = $("#teach-preview");
  const nameInput = $("#teach-name");
  const textInput = $("#teach-text");
  const langSelect = $("#teach-lang");
  const recordBtn = $("#btn-teach-record");
  const saveBtn = $("#btn-teach-save");
  const status = $("#teach-status");

  let samples = 0;
  let recording = false;
  let countdownTimer = 0;
  let signs = [];

  function open(trigger, { name = "" } = {}) {
    langSelect.replaceChildren(...getLanguages().map((l) =>
      el("option", { value: l.code }, `${l.native_name}`)));
    langSelect.value = getLanguages().some((l) => l.code === getLang()) ? getLang() : "en";
    textInput.lang = langSelect.value;
    startPreview();
    renderList();
    updateButtons();
    if (name) nameInput.value = name;
    ctl.open(trigger);
    (name ? textInput : nameInput).focus();
  }

  function startPreview() {
    const hasCam = !!camera.stream;
    $("#teach-nocam").hidden = hasCam;
    if (hasCam) {
      video.srcObject = camera.stream;
      video.play().catch(() => {});
    }
  }

  function stopPreview() {
    clearInterval(countdownTimer);
    $("#teach-countdown").hidden = true;
    video.srcObject = null;
    if (recording) {
      recording = false;
      setRecordingUI(false);
    }
  }

  function updateButtons() {
    const ready = samples >= MIN_SAMPLES && nameInput.value.trim() && textInput.value.trim();
    saveBtn.disabled = !ready || recording;
    recordBtn.disabled = recording || !camera.stream || !live.open;
    $("#teach-meter-fill").style.width = `${Math.min(100, (samples / GOOD_SAMPLES) * 100)}%`;
  }

  function setRecordingUI(on) {
    preview.dataset.state = on ? "recording" : "idle";
    $("#teach-rec").hidden = !on;
    if (!on) $("#teach-progress-fill").style.width = "0";
    updateButtons();
  }

  function startRecording() {
    if (!live.open) return toast("Not connected to the recognizer.", "error");
    recording = true;
    updateButtons();
    const counter = $("#teach-countdown");
    let n = 3;
    counter.textContent = String(n);
    counter.hidden = false;
    status.textContent = "Get ready — show the sign to the camera.";
    countdownTimer = setInterval(() => {
      n -= 1;
      if (n > 0) {
        counter.textContent = String(n);
        return;
      }
      clearInterval(countdownTimer);
      counter.hidden = true;
      setRecordingUI(true);
      live.send({ action: "teach_start" });
    }, 800);
  }

  function describeSamples() {
    if (samples === 0) return "No hands captured yet. Make sure your hand is fully visible.";
    if (samples < MIN_SAMPLES) return `${samples} frames captured — record again (need at least ${MIN_SAMPLES}).`;
    if (samples < GOOD_SAMPLES) return `${samples} frames captured. Good — one or two more takes will make it more reliable.`;
    return `${samples} frames captured. Ready to save.`;
  }

  /** Called by main.js for every live message. Returns true if consumed. */
  function handleMessage(msg) {
    if (msg.type === "frame_result" && msg.teaching) {
      $("#teach-progress-fill").style.width = `${(msg.teach_progress || 0) * 100}%`;
      status.textContent = msg.hands_detected
        ? `Recording… ${msg.teach_samples} frames`
        : "Recording… no hand visible — move it into view";
      return true;
    }
    if (msg.type === "teach_recorded") {
      recording = false;
      samples = msg.samples;
      setRecordingUI(false);
      status.textContent = describeSamples();
      return true;
    }
    if (msg.type === "teach_saved") {
      signs = msg.signs;
      toast(`Learned “${msg.sign.name}”. Sign it any time to say it.`, "success");
      resetForm();
      renderList();
      onSignsChanged(signs);
      return true;
    }
    if (msg.type === "error" && (msg.code === "teach_invalid" || msg.code === "teach_unavailable")) {
      recording = false;
      setRecordingUI(false);
      status.textContent = msg.message;
      toast(msg.message, "error");
      return true;
    }
    return false;
  }

  function resetForm() {
    samples = 0;
    nameInput.value = "";
    textInput.value = "";
    status.textContent = "Record 2–3 times, moving your hand slightly each time.";
    updateButtons();
  }

  function renderList() {
    const list = $("#taught-list");
    if (!signs.length) {
      list.replaceChildren(el("li.taught-empty", {}, "No taught signs yet."));
      return;
    }
    list.replaceChildren(...signs.map((s) => el("li.taught-item", {},
      el("div.taught-main", {},
        el("span.taught-name", {}, s.name),
        el("span.taught-text", { lang: s.text_lang }, s.text),
        el("span.taught-meta", {}, `${s.samples} examples`)),
      el("button.btn.btn-ghost.btn-icon", {
        type: "button",
        "aria-label": `Delete ${s.name}`,
        onclick: () => remove(s),
      }, icon("i-x")))));
  }

  async function remove(sign) {
    if (!confirm(`Delete the taught sign “${sign.name}”?`)) return;
    try {
      const res = await fetch(`/api/custom-signs/${encodeURIComponent(sign.id)}`, { method: "DELETE" });
      if (!res.ok) throw new Error();
      signs = signs.filter((s) => s.id !== sign.id);
      renderList();
      onSignsChanged(signs);
      toast(`Deleted “${sign.name}”.`);
    } catch {
      toast("Couldn't delete that sign.", "error");
    }
  }

  recordBtn.onclick = startRecording;
  $("#btn-teach-reset").onclick = () => {
    live.send({ action: "teach_discard" });
    resetForm();
  };
  nameInput.oninput = textInput.oninput = updateButtons;
  langSelect.onchange = () => { textInput.lang = langSelect.value; };
  $("#teach-form").onsubmit = (e) => {
    e.preventDefault();
    if (saveBtn.disabled) return;
    live.send({
      action: "teach_save",
      name: nameInput.value.trim(),
      text: textInput.value.trim(),
      text_lang: langSelect.value,
    });
  };

  return {
    open,
    handleMessage,
    setSigns(list) {
      signs = list || [];
      if (dialog.open) renderList();
    },
  };
}
