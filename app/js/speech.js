/**
 * Speech output.
 *  1. Server audio: pre-generated clips for every template sentence (offline,
 *     natural voices), else gTTS when online.
 *  2. Browser speechSynthesis — only if a voice for that language exists,
 *     so Tamil text is never read out by an English voice.
 */

import { api, getBlob } from "./api.js";

const LOCALES = { en: "en-IN", ta: "ta-IN", hi: "hi-IN", te: "te-IN", kn: "kn-IN", ml: "ml-IN" };

export class Speaker {
  constructor({ onState } = {}) {
    this.onState = onState || (() => {});
    this.audio = new Audio();
    this.objectUrl = null;
    this.token = 0;
    this.audio.addEventListener("ended", () => this.onState("idle"));
    this.audio.addEventListener("pause", () => this.onState("idle"));
    window.speechSynthesis?.getVoices(); // warm the voice list
  }

  stop() {
    this.token++;
    this.audio.pause();
    window.speechSynthesis?.cancel();
    this.onState("idle");
  }

  async speak(text, lang) {
    if (!text) return false;
    this.stop();
    const token = this.token;
    this.onState("loading");

    try {
      const blob = await getBlob(api.ttsUrl(text, lang), { timeout: 6000 });
      if (token !== this.token) return false;
      if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = URL.createObjectURL(blob);
      this.audio.src = this.objectUrl;
      await this.audio.play();
      this.onState("speaking");
      return true;
    } catch {
      if (token !== this.token) return false;
    }

    const voice = this._voiceFor(lang);
    if (voice) {
      const u = new SpeechSynthesisUtterance(text);
      u.lang = voice.lang;
      u.voice = voice;
      u.rate = 0.95;
      u.onend = u.onerror = () => token === this.token && this.onState("idle");
      window.speechSynthesis.speak(u);
      this.onState("speaking");
      return true;
    }
    this.onState("idle");
    return false;
  }

  _voiceFor(lang) {
    const voices = window.speechSynthesis?.getVoices() || [];
    const locale = LOCALES[lang] || lang;
    return voices.find((v) => v.lang === locale) || voices.find((v) => v.lang?.toLowerCase().startsWith(lang));
  }
}
