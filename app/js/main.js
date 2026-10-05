/**
 * SignBridge counter controller.
 *
 * Citizen (left): camera -> server-side hand landmarks -> classifier ->
 *   hold-to-confirm words -> pause/“Send now” -> sentence in N languages.
 * Staff (right): reads/hears the sentence, answers with quick replies or
 *   free text, which the signing hand performs and the citizen reads.
 *
 * The server is the single source of truth for the citizen's words; the UI
 * only renders what it reports (frame_result / words_updated / sentence_*).
 */

import { api } from "./api.js";
import { Camera } from "./camera.js";
import { LiveClient } from "./live.js";
import { Speaker } from "./speech.js";
import { initTeach } from "./teach.js";
import { initAskAI } from "./ai.js";
import { signNote, signWord, t } from "./i18n.js";
import { $, el, icon, setStatus, timeLabel, toast, wireDialog } from "./ui.js";

const STORAGE_KEY = "signbridge.prefs";

const state = {
  vocab: null,
  pack: "railway",
  lang: "ta",
  region: "",
  words: [],
  request: null, // { words, translations, source }
  transcriptCount: 0,
  hold: { state: "idle", confirmedUntil: 0 },
  lastHands: [],
  visionEnabled: false,
  debug: new URLSearchParams(location.search).has("debug"),
};

let handScene = null;
let avatar = null;
let signable = new Set();

const camera = new Camera($("#webcam-video"));
const speaker = new Speaker({ onState: renderSpeakState });
const live = new LiveClient({ onMessage: handleLiveMessage, onStatus: handleLiveStatus });

const teach = initTeach({
  live,
  camera,
  getLanguages: () => state.vocab?.languages || [],
  getLang: () => state.lang,
  onSignsChanged: (list) => {
    if (state.vocab) state.vocab.custom_signs = list;
    renderGuide();
  },
});

const askAI = initAskAI({
  camera,
  getHands: () => state.lastHands,
  getPack: () => state.pack,
  onAccept: (sign) => addSign(sign) && toast(`Added ${sign}`, "success", 1600),
  onSpeakText: (text) => {
    const translations = Object.fromEntries((state.vocab?.languages || [{ code: "en" }]).map((l) => [l.code, text]));
    const speechLangs = Object.fromEntries(Object.keys(translations).map((c) => [c, "en"]));
    state.request = { words: [text], translations, source: "ai", speechLangs };
    renderRequest({ fresh: true });
    speakRequest();
    addTranscript({ role: "citizen", text, meta: "AI-identified sign (unconfirmed)" });
  },
  onTeach: (name) => teach.open(null, { name }),
});

const langDialog = wireDialog($("#lang-dialog"));
const guideDialog = wireDialog($("#guide-dialog"));

// ---------------------------------------------------------------- Bootstrap

loadPrefs();
init();

async function init() {
  wireStaticControls();
  initKeyboard();
  const threeReady = initThree();
  live.configure({ pack: state.pack, lang: state.lang });
  live.connect();
  startCamera();
  refreshHealth();
  await loadVocabulary();
  await threeReady;
  if (state.vocab) renderGuide(); // re-render once we know which signs the hand can show
}

function loadPrefs() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    Object.assign(state, { pack: saved.pack || state.pack, lang: saved.lang || state.lang, region: saved.region || "" });
  } catch { /* storage unavailable: use defaults */ }
}

function savePrefs() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ pack: state.pack, lang: state.lang, region: state.region }));
  } catch { /* ignore */ }
}

async function initThree() {
  if (!window.THREE) {
    console.warn("Three.js unavailable; 3D views disabled.");
    return;
  }
  try {
    const [{ HandOverlayScene }, { SigningAvatarScene, SIGNABLE, poseFromFeatures }] = await Promise.all([
      import("./three/hand-scene.js"),
      import("./three/avatar-scene.js"),
    ]);
    signable = SIGNABLE;
    avatarPoseFrom = poseFromFeatures;
    handScene = new HandOverlayScene($("#hand-canvas"), $("#webcam-video"));
    avatar = new SigningAvatarScene($("#avatar-canvas"));
  } catch (err) {
    console.warn("3D init failed:", err);
  }
}

async function loadVocabulary() {
  try {
    state.vocab = await api.vocabulary();
  } catch (err) {
    const grid = $("#reply-grid");
    grid.replaceChildren(
      el("div.note", {}, `Couldn't load counter phrases (${err.message}). `,
        el("button.btn.btn-outline", { type: "button", onclick: () => { grid.replaceChildren(); loadVocabulary(); } }, "Retry")),
    );
    return;
  }
  if (!state.vocab.packs[state.pack]) state.pack = Object.keys(state.vocab.packs)[0];
  if (!state.vocab.languages.some((l) => l.code === state.lang)) state.lang = state.vocab.languages[0]?.code || "en";
  renderPackSwitch();
  renderLanguageButton();
  renderReplies();
  renderGuide();
  renderLanguageDialog();
  renderWords();
  teach.setSigns(state.vocab.custom_signs);
}

async function refreshHealth() {
  try {
    const h = await api.health();
    const provider = { openai: "OpenAI", groq: "Groq" }[h.llm_provider];
    setStatus(
      "status-phrases",
      "ok",
      provider ? `AI phrasing · ${provider}` : "Offline phrases",
      provider ? "Verified templates first, AI for new combinations" : "Using verified phrase templates (no internet needed)",
    );
    state.visionEnabled = !!h.vision_enabled;
    updateAskAI();
    if (!h.model_loaded) setStatus("status-live", "error", "Model missing", "Sign classifier is not loaded on the server");
  } catch {
    setStatus("status-phrases", "error", "Server offline");
    setTimeout(refreshHealth, 5000);
  }
}

// ---------------------------------------------------------------- Camera

async function startCamera() {
  setCameraOverlay("starting");
  setStatus("status-camera", "pending", "Camera");
  try {
    await camera.start();
    setCameraOverlay("live");
    setStatus("status-camera", "ok", "Camera on");
    updateAskAI();
    camera.onEnded = () => {
      setCameraOverlay("error", { title: "Camera disconnected", body: "The camera stopped. Reconnect it and try again." });
      setStatus("status-camera", "error", "Camera off");
    };
    live.startFrames(camera);
  } catch (err) {
    setCameraOverlay("error", err.info);
    setStatus("status-camera", "error", "Camera off", err.info?.title);
  }
}

function updateAskAI() {
  $("#btn-ask-ai").hidden = !(state.visionEnabled && camera.stream);
}

// Draw attention to Ask AI after the recognizer has been unsure for a while.
let unsureSince = 0;
let lastUnsure = 0;
let lastNudge = 0;
function nudgeAskAI() {
  const now = performance.now();
  if (!state.visionEnabled) return;
  if (now - lastUnsure > 1000) unsureSince = now; // a new unsure stretch
  lastUnsure = now;
  if (now - unsureSince > 2500 && now - lastNudge > 15000) {
    lastNudge = now;
    const btn = $("#btn-ask-ai");
    btn.classList.remove("nudge");
    void btn.offsetWidth;
    btn.classList.add("nudge");
  }
}

function setCameraOverlay(mode, info = {}) {
  const stage = $("#camera-stage");
  stage.dataset.state = mode;
  const iconHost = $("#overlay-icon");
  const actions = $("#overlay-actions");
  if (mode === "starting") {
    iconHost.replaceChildren(el("span.spinner", { "aria-hidden": "true" }));
    $("#overlay-title").textContent = "Starting camera…";
    $("#overlay-body").textContent = "Allow camera access when your browser asks.";
    actions.hidden = true;
  } else if (mode === "error") {
    iconHost.replaceChildren(icon(info.icon || "i-camera-off"));
    $("#overlay-title").textContent = info.title || "Camera unavailable";
    $("#overlay-body").textContent = info.body || "";
    actions.hidden = false;
    $("#btn-camera-retry").hidden = info.retry === false;
  }
}

// ---------------------------------------------------------------- Live messages

function handleLiveStatus(status) {
  if (status === "open") {
    setStatus("status-live", "ok", "Recognizer live");
    setWordControlsEnabled(true);
  } else if (status === "connecting") {
    setStatus("status-live", "pending", "Connecting…");
  } else {
    setStatus("status-live", "error", "Reconnecting…", "Lost connection to the recognition server");
    setWordControlsEnabled(false);
    renderRecognition({ hands_detected: 0, hold_progress: 0, pause_progress: 0 });
  }
}

function handleLiveMessage(msg) {
  switch (msg.type) {
    case "hello":
      if (!msg.server_detection) {
        live.stopFrames();
        setCameraOverlay("error", {
          icon: "i-alert",
          title: "Hand detection is unavailable",
          body: "The server can't run hand tracking right now. Use the sign guide to tap signs instead.",
          retry: false,
        });
      }
      if (!msg.model_loaded) setStatus("status-live", "error", "Model missing");
      break;
    case "frame_result":
      handScene?.setHands(msg.extracted_landmarks || []);
      if (msg.extracted_landmarks?.length) state.lastHands = msg.extracted_landmarks;
      if (msg.teaching) {
        teach.handleMessage(msg);
        break;
      }
      renderRecognition(msg);
      if (msg.confirmed_word) onWordConfirmed(msg.confirmed_word);
      setWords(msg.words);
      if (state.debug) renderPerf();
      break;
    case "words_updated":
    case "word_removed":
    case "reset_ack":
      setWords(msg.words || []);
      break;
    case "sentence_pending":
      renderRequestPending(msg.words);
      break;
    case "sentence_ready":
      onSentenceReady(msg);
      break;
    case "teach_recorded":
    case "teach_saved":
      teach.handleMessage(msg);
      break;
    case "error":
      if (teach.handleMessage(msg)) break;
      if (msg.code !== "bad_frame") toast(msg.message, "error");
      break;
    default:
      break;
  }
}

function renderRecognition(msg) {
  const chip = $("#recog-chip");
  const label = $("#recog-label");
  const meta = $("#recog-meta");
  const fill = $("#hold-ring-fill");
  const now = performance.now();

  if (now < state.hold.confirmedUntil) return; // keep the confirmation visible briefly

  let mode = "idle";
  let progress = 0;
  if (!msg.hands_detected) {
    label.textContent = state.words.length ? "Lower hands to send" : "Show your hands";
    meta.textContent = state.words.length ? "Or sign the next word" : "Signing area is in front of you";
  } else if (msg.candidate && msg.hold_progress > 0) {
    mode = "locking";
    progress = msg.hold_progress;
    label.textContent = msg.candidate;
    meta.textContent = `Hold steady · ${Math.round(progress * 100)}%`;
  } else if (msg.top_prediction && msg.top_prediction !== "NONE" && msg.confidence >= 0.4) {
    mode = "tracking";
    label.textContent = `${msg.top_prediction}?`;
    meta.textContent = state.visionEnabled
      ? `Not sure yet (${Math.round(msg.confidence * 100)}%) — hold steady, or tap Ask AI`
      : `Not sure yet (${Math.round(msg.confidence * 100)}%) — hold the sign clearly`;
    nudgeAskAI();
  } else {
    mode = "tracking";
    label.textContent = "Hands detected";
    meta.textContent = "Make a sign and hold it";
  }
  chip.dataset.state = mode;
  fill.style.strokeDashoffset = String(100 - progress * 100);
  handScene?.setHold(progress, mode === "locking" ? "locking" : "tracking");

  const pause = msg.pause_progress || 0;
  $("#pause-meter").hidden = !(pause > 0 && state.words.length);
  $("#pause-fill").style.width = `${pause * 100}%`;
  if (pause > 0) {
    const secs = Math.max(0, (1 - pause) * (state.vocab?.timing?.pause_seconds || 1.5));
    $("#pause-text").textContent = `Sending in ${secs.toFixed(1)}s`;
  }
}

function onWordConfirmed(word) {
  state.hold.confirmedUntil = performance.now() + 700;
  const chip = $("#recog-chip");
  chip.dataset.state = "confirmed";
  $("#recog-label").textContent = `✓ ${word}`;
  $("#recog-meta").textContent = "Added to your request";
  $("#hold-ring-fill").style.strokeDashoffset = "0";
  handScene?.setHold(1, "confirmed");
  handScene?.confirm();
  chime();
}

function renderPerf() {
  const chip = $("#perf-chip");
  chip.hidden = false;
  const s = live.stats;
  chip.textContent = `${s.fps.toFixed(0)} fps · ${s.latencyMs.toFixed(0)} ms (server ${s.serverMs.toFixed(0)} ms)`;
}

// ---------------------------------------------------------------- Words

function setWords(words) {
  if (!Array.isArray(words)) return;
  const same = words.length === state.words.length && words.every((w, i) => w === state.words[i]);
  if (same) return;
  state.words = [...words];
  renderWords();
}

function renderWords() {
  const list = $("#word-chips");
  if (!state.words.length) {
    list.replaceChildren(el("li.placeholder", {}, "Signs you hold will appear here."));
  } else {
    list.replaceChildren(...state.words.map((w, i) =>
      el("li.word-chip", {},
        el("span.word-chip-index", { "aria-hidden": "true" }, i + 1),
        w,
        el("button", { type: "button", "aria-label": `Remove ${w}`, onclick: () => live.send({ action: "remove_word", index: i }) }, icon("i-x")),
      )));
  }
  const has = state.words.length > 0 && live.open;
  $("#btn-finalize").disabled = !has;
  $("#btn-clear-words").disabled = !has;
  if (!state.words.length) $("#pause-meter").hidden = true;
}

function setWordControlsEnabled(enabled) {
  $("#btn-finalize").disabled = !enabled || !state.words.length;
  $("#btn-clear-words").disabled = !enabled || !state.words.length;
}

function addSign(sign) {
  if (!live.send({ action: "add_word", sign })) {
    toast("Not connected to the recognizer yet.", "error");
    return false;
  }
  return true;
}

// ---------------------------------------------------------------- Citizen request

function renderRequestPending(words) {
  const card = $("#request-card");
  card.dataset.state = "pending";
  $("#request-empty").hidden = true;
  $("#request-body").hidden = true;
  $("#request-pending").hidden = false;
  $("#pending-words").textContent = words.join(" · ");
}

function onSentenceReady(msg) {
  state.request = {
    words: msg.words,
    translations: msg.translations || {},
    source: msg.source,
    speechLangs: msg.speech_langs || {},
  };
  renderRequest({ fresh: true });
  speakRequest();
  addTranscript({
    role: "citizen",
    text: requestText(),
    sub: state.lang !== "en" ? state.request.translations.en : "",
    meta: msg.words.join(" · "),
  });
}

function requestText(lang = state.lang) {
  const r = state.request;
  if (!r) return "";
  return r.translations[lang] || r.translations.en || r.words.join(" ");
}

function renderRequest({ fresh = false } = {}) {
  const card = $("#request-card");
  const r = state.request;
  $("#request-pending").hidden = true;
  if (!r) {
    card.dataset.state = "empty";
    $("#request-empty").hidden = false;
    $("#request-body").hidden = true;
    $("#source-badge").hidden = true;
    $("#btn-speak").disabled = true;
    $("#btn-dismiss").disabled = true;
    return;
  }
  $("#request-empty").hidden = true;
  $("#request-body").hidden = false;
  const text = $("#request-text");
  text.textContent = requestText();
  text.lang = state.lang;
  const en = r.translations.en || "";
  $("#request-english").textContent = state.lang !== "en" && en ? en : "";
  $("#request-gloss").textContent = `SIGNED: ${r.words.join(" · ")}`;

  const badge = $("#source-badge");
  badge.hidden = false;
  badge.dataset.source = r.source;
  badge.textContent = {
    template: "Verified phrase",
    llm: "AI phrased",
    taught: "Taught sign",
    ai: "AI guess — confirm with citizen",
    fallback: "Raw signs — confirm with citizen",
  }[r.source] || r.source;

  $("#btn-speak").disabled = false;
  $("#btn-dismiss").disabled = false;
  card.dataset.state = "ready";
  if (fresh) {
    void card.offsetWidth;
    card.dataset.state = "fresh";
  }
}

async function speakRequest() {
  if (!state.request) return;
  // A taught meaning without a translation is spoken in its own language.
  const voiceLang = state.request.speechLangs?.[state.lang] || state.lang;
  const ok = await speaker.speak(requestText(), voiceLang);
  if (!ok) toast("Audio unavailable for this language — please read the text.", "info");
}

function renderSpeakState(s) {
  const btn = $("#btn-speak");
  btn.classList.toggle("btn-speaking", s === "speaking" || s === "loading");
  $("#btn-speak-label").textContent = s === "loading" ? "Loading audio…" : s === "speaking" ? "Speaking…" : "Speak again";
}

function dismissRequest() {
  speaker.stop();
  state.request = null;
  renderRequest();
}

// ---------------------------------------------------------------- Clerk replies

function packData() {
  return state.vocab?.packs?.[state.pack];
}

function langName(code) {
  return state.vocab?.languages.find((l) => l.code === code)?.native_name || code;
}

function renderReplies() {
  const grid = $("#reply-grid");
  const replies = packData()?.quick_replies || [];
  if (!replies.length) {
    grid.replaceChildren(el("p.field-hint", {}, "No quick replies for this counter. Use a custom reply below."));
    return;
  }
  grid.replaceChildren(...replies.map((reply, i) => {
    const main = reply.text[state.lang] || reply.text.en;
    return el("button.reply-btn", {
      type: "button",
      dataset: { replyId: reply.id },
      "aria-keyshortcuts": i < 9 ? String(i + 1) : undefined,
      onclick: () => sendReply({ id: reply.id, text: main, english: reply.text.en, clips: reply.clips }),
    },
    i < 9 ? el("span.reply-key", { "aria-hidden": "true" }, i + 1) : null,
    el("span.reply-main", { lang: state.lang }, main),
    state.lang !== "en" ? el("span.reply-sub", {}, reply.text.en) : null);
  }));
}

/**
 * Turns typed text into an exact, word-by-word signing plan:
 *   built-in sign  -> signed        taught sign -> recorded hand shape
 *   number         -> digit signs   other Latin word -> fingerspelled
 *   other scripts  -> shown as text (fingerspelling uses the Latin alphabet)
 */
function buildSigningPlan(text) {
  const words = text.match(/[\p{L}\p{M}\p{N}']+/gu) || [];
  const taught = new Map((state.vocab?.custom_signs || []).map((t) => [t.name.toUpperCase(), t]));
  const plan = [];
  for (let i = 0; i < words.length; ) {
    let matched = false;
    for (let n = Math.min(3, words.length - i); n >= 1 && !matched; n--) {
      const phrase = words.slice(i, i + n).join(" ");
      const key = phrase.toUpperCase();
      if (/^\d+$/.test(key)) break;
      if (signable.has(key) && !/^\d$/.test(key)) {
        plan.push({ label: phrase, kind: "sign", items: [{ kind: "sign", sign: key }] });
      } else if (taught.has(key) && avatarPose(taught.get(key))) {
        plan.push({ label: phrase, kind: "taught", items: [{ kind: "pose", points: avatarPose(taught.get(key)) }] });
      } else continue;
      i += n;
      matched = true;
    }
    if (matched) continue;

    const word = words[i++];
    if (/^\d+$/.test(word)) {
      plan.push({ label: word, kind: "spell", items: [...word].map((d) => ({ kind: "sign", sign: d })) });
    } else if (/^[A-Za-z']+$/.test(word)) {
      const letters = [...word.toUpperCase()].filter((c) => c !== "'");
      plan.push({ label: word, kind: "spell", items: letters.map((letter) => ({ kind: "letter", letter })) });
    } else {
      plan.push({ label: word, kind: "text", items: [{ kind: "gap" }] });
    }
  }
  return plan;
}

const poseCache = new WeakMap();
function avatarPose(taughtSign) {
  if (!avatar || !taughtSign?.pose) return null;
  if (!poseCache.has(taughtSign)) poseCache.set(taughtSign, avatarPoseFrom(taughtSign.pose));
  return poseCache.get(taughtSign);
}
let avatarPoseFrom = () => null;

function planFromClips(clips) {
  return (clips || []).filter((c) => signable.has(c))
    .map((c) => ({ label: c, kind: "sign", items: [{ kind: "sign", sign: c }] }));
}

function renderPlanSteps(plan) {
  const steps = $("#clip-steps");
  steps.replaceChildren(...plan.map((w) => {
    const li = el("li", { dataset: { state: "pending", kind: w.kind } });
    if (w.kind === "spell") {
      w.items.forEach((it) => li.append(el("span.ltr", {}, it.letter || it.sign)));
      li.setAttribute("aria-label", `${w.label} (fingerspelled)`);
    } else {
      li.textContent = w.label;
      if (w.kind === "text") li.title = "Shown as text — this script cannot be fingerspelled";
    }
    return li;
  }));
  return steps;
}

function sendReply({ id = null, text, english = "", clips = null }) {
  // Quick replies carry curated ISL clips; typed text is signed exactly as written.
  const plan = clips ? planFromClips(clips) : buildSigningPlan(text);
  const caption = $("#reply-text");
  caption.classList.remove("is-idle");
  caption.textContent = text;
  caption.lang = id ? state.lang : "";

  const steps = renderPlanSteps(plan);
  document.querySelectorAll(".reply-btn[data-playing]").forEach((b) => b.removeAttribute("data-playing"));
  const btn = id && document.querySelector(`.reply-btn[data-reply-id="${CSS.escape(id)}"]`);
  if (btn) btn.dataset.playing = "true";

  // Flatten to avatar items, with a short rest between words.
  const items = [];
  const where = [];
  plan.forEach((w, wi) => {
    w.items.forEach((it, li) => { items.push(it); where.push([wi, li]); });
    if (wi < plan.length - 1) { items.push({ kind: "gap" }); where.push([wi, -1]); }
  });

  const lis = [...steps.children];
  if (avatar && items.length) {
    avatar.play(items, {
      onStep: (idx) => {
        const [wi, li] = where[idx];
        lis.forEach((node, i) => {
          node.dataset.state = i < wi ? "done" : i === wi ? (li < 0 ? "done" : "active") : "pending";
          node.querySelectorAll(".ltr").forEach((sp, j) => sp.classList.toggle("on", i === wi && j === li));
        });
      },
      onDone: (cancelled) => {
        btn?.removeAttribute("data-playing");
        if (!cancelled) lis.forEach((node) => {
          node.dataset.state = "done";
          node.querySelectorAll(".ltr.on").forEach((sp) => sp.classList.remove("on"));
        });
      },
    });
  } else {
    btn?.removeAttribute("data-playing");
  }

  const summary = plan.map((w) => (w.kind === "spell" ? `${w.label} (spelled)` : w.kind === "text" ? `${w.label} (text)` : w.label));
  addTranscript({
    role: "clerk",
    text,
    sub: english && english !== text ? english : "",
    meta: summary.length ? `Signed: ${summary.join(" · ")}` : "Shown as text",
  });
}

// ---------------------------------------------------------------- Transcript

function addTranscript({ role, text, sub, meta }) {
  const list = $("#transcript");
  if (!state.transcriptCount) list.replaceChildren();
  state.transcriptCount++;
  const item = el(`li.msg.msg-${role}`, {},
    el("div.msg-meta", {}, el("span", {}, role === "citizen" ? "Citizen" : "Counter"), el("span", {}, timeLabel())),
    el("p.msg-text", {}, text),
    sub ? el("p.msg-sub", {}, sub) : null,
    meta ? el("p.msg-sub", {}, meta) : null);
  list.append(item);
  while (list.children.length > 60) list.firstElementChild.remove();
  list.scrollTop = list.scrollHeight;
}

// ---------------------------------------------------------------- Pack & language

function renderPackSwitch() {
  const host = $("#pack-switch");
  const packs = Object.entries(state.vocab.packs);
  host.replaceChildren(...packs.map(([id, p]) => el("button", {
    type: "button",
    role: "radio",
    "aria-checked": String(id === state.pack),
    tabindex: id === state.pack ? "0" : "-1",
    title: p.description,
    dataset: { pack: id },
    onclick: () => selectPack(id),
  }, p.name.split(/\s+/)[0])));
  host.onkeydown = (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const ids = packs.map(([id]) => id);
    const next = ids[(ids.indexOf(state.pack) + (e.key === "ArrowRight" ? 1 : ids.length - 1)) % ids.length];
    selectPack(next);
    host.querySelector(`[data-pack="${next}"]`)?.focus();
  };
}

function selectPack(id) {
  if (id === state.pack) return;
  state.pack = id;
  savePrefs();
  live.configure({ pack: id });
  renderPackSwitch();
  renderReplies();
  renderGuide();
}

function renderLanguageButton() {
  $("#current-lang-name").textContent = langName(state.lang);
}

function renderLanguageDialog() {
  const regions = state.vocab.regions || {};
  const select = $("#region-select");
  if (select.options.length <= 1) {
    Object.keys(regions).forEach((r) => select.append(el("option", { value: r }, r)));
    select.onchange = () => {
      state.region = select.value;
      savePrefs();
      renderLanguageDialog();
    };
  }
  select.value = state.region;

  const common = regions[state.region]?.common || [];
  const langs = [...state.vocab.languages].sort((a, b) => {
    const ia = common.indexOf(a.code);
    const ib = common.indexOf(b.code);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  $("#lang-list").replaceChildren(...langs.map((l) => el("button.lang-option", {
    type: "button",
    role: "option",
    "aria-selected": String(l.code === state.lang),
    onclick: () => { selectLanguage(l.code); langDialog.close(); },
  },
  el("span", {}, el("span.lang-native", { lang: l.code }, l.native_name), el("span.lang-english", {}, l.english_name)),
  regions[state.region]?.primary === l.code ? el("span.lang-tag", {}, "Main language here")
    : common.includes(l.code) ? el("span.lang-tag", {}, "Common here") : null,
  l.code === state.lang ? icon("i-check") : null)));
}

function selectLanguage(code) {
  if (code === state.lang) return;
  state.lang = code;
  savePrefs();
  live.configure({ lang: code });
  renderLanguageButton();
  renderLanguageDialog();
  renderReplies();
  renderGuide();
  if (state.request) {
    renderRequest();
    speakRequest();
  }
}

// ---------------------------------------------------------------- Sign guide (tap-to-sign)

/** Citizen-facing sign guide, fully in the selected language. */
function renderGuide() {
  const pack = packData();
  if (!pack) return;
  const lang = state.lang;
  const lex = state.vocab.lexicon;
  const tr = (key) => t(lang, key);
  const word = (sign) => signWord(lex, sign, lang);

  const dialog = $("#guide-dialog");
  dialog.lang = lang;
  dialog.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = tr(node.dataset.i18n); });
  $("#guide-pack-name").textContent = `· ${pack.name}`;

  const addBtn = (sign, label) => el("button.btn.btn-primary", {
    type: "button",
    "aria-label": `${tr("add")} ${label}`,
    onclick: () => addSign(sign) && toast(`${tr("added")}: ${label}`, "success", 1400),
  }, icon("i-plus"), tr("add"));

  const taught = state.vocab.custom_signs || [];
  $("#taught-guide").hidden = !taught.length;
  $("#taught-grid").replaceChildren(...taught.map((ts) => el("div.sign-card", {},
    el("span.sign-name", {}, ts.name),
    el("p.sign-desc", { lang: ts.text_lang }, `${tr("says")} “${ts.text}”`),
    el("div.sign-actions", {}, addBtn(ts.name, ts.name)))));

  const packSigns = new Set(pack.signs);
  const phrases = Object.entries(state.vocab.templates || {})
    .filter(([key]) => signsForKey(key).every((s) => packSigns.has(s)));
  $("#phrase-list").replaceChildren(...phrases.map(([key, tpl]) => el("button.phrase-btn", {
    type: "button",
    title: key,
    onclick: () => askPhrase(key),
  },
  el("span.phrase-gloss", {}, signsForKey(key).map(word).join(" · ")),
  el("span.phrase-text", {}, tpl[lang] || tpl.en))));

  $("#sign-grid").replaceChildren(...pack.signs.map((sign) => {
    const name = word(sign);
    return el("div.sign-card", {},
      el("span.sign-name", {}, name),
      name.toUpperCase() !== sign ? el("span.sign-gloss", { lang: "en" }, sign) : null,
      el("p.sign-desc", {}, signNote(lex, sign, lang)),
      el("div.sign-actions", {},
        addBtn(sign, name),
        signable.has(sign) ? el("button.btn.btn-outline", {
          type: "button",
          "aria-label": `${tr("show")} ${name}`,
          onclick: () => { guideDialog.close(); avatar?.preview(sign); },
        }, icon("i-play"), tr("show")) : null));
  }));
}

/** Template keys are space-joined signs; resolve multi-word signs like "THANK YOU". */
function signsForKey(key) {
  const classes = new Set(state.vocab.classes);
  const parts = key.split(" ");
  const signs = [];
  for (let i = 0; i < parts.length; i++) {
    const pair = `${parts[i]} ${parts[i + 1]}`;
    if (classes.has(pair)) { signs.push(pair); i++; } else signs.push(parts[i]);
  }
  return signs;
}

function askPhrase(key) {
  const signs = signsForKey(key);
  if (!live.open) return toast("Not connected to the recognizer yet.", "error");
  live.send({ action: "reset" });
  signs.forEach((sign) => live.send({ action: "add_word", sign }));
  live.send({ action: "finalize" });
  guideDialog.close();
}

// ---------------------------------------------------------------- Controls & keyboard

function wireStaticControls() {
  $("#btn-language").onclick = (e) => langDialog.open(e.currentTarget);
  $("#btn-guide").onclick = (e) => guideDialog.open(e.currentTarget);
  $("#btn-teach").onclick = (e) => teach.open(e.currentTarget);
  $("#btn-ask-ai").onclick = (e) => askAI.open(e.currentTarget);
  document.querySelectorAll("[data-open-guide]").forEach((b) => { b.onclick = () => guideDialog.open(b); });
  $("#btn-camera-retry").onclick = startCamera;
  $("#btn-finalize").onclick = () => live.send({ action: "finalize" });
  $("#btn-clear-words").onclick = () => live.send({ action: "reset" });
  $("#btn-speak").onclick = speakRequest;
  $("#btn-dismiss").onclick = dismissRequest;
  $("#custom-form").onsubmit = (e) => {
    e.preventDefault();
    const input = $("#custom-input");
    const text = input.value.trim();
    if (!text) return input.focus();
    sendReply({ text });
    input.value = "";
  };
}

function isTyping(target) {
  return target.closest?.("input, textarea, select, [contenteditable]");
}

function initKeyboard() {
  window.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (document.querySelector("dialog[open]")) return; // dialogs handle their own keys (Esc closes)
    if (isTyping(e.target)) {
      if (e.key === "Escape") e.target.blur();
      return;
    }
    if (e.code === "Space" && !e.target.closest("button, a, [role=radio]")) {
      e.preventDefault();
      if (state.request) speakRequest();
    } else if (e.key === "Escape") {
      dismissRequest();
      if (state.words.length) live.send({ action: "reset" });
    } else if (/^[1-9]$/.test(e.key)) {
      document.querySelectorAll(".reply-btn")[Number(e.key) - 1]?.click();
    }
  });
}

let audioCtx = null;
function chime() {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.frequency.value = 880;
    osc.connect(gain).connect(audioCtx.destination);
    gain.gain.setValueAtTime(0.06, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.12);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.12);
  } catch { /* audio not available */ }
}
