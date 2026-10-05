"""
SignBridge Offline Audio Generator
Pre-generates gTTS MP3 files for template sentences, quick replies, and vocabulary
in all six launch languages (en, ta, hi, te, kn, ml) with hash indexing.
Matches B4 in Prompt Library.
"""

import os
import sys
import json
import hashlib
from gtts import gTTS

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

AUDIO_DIR = os.path.join("assets", "audio")
INDEX_PATH = os.path.join(AUDIO_DIR, "index.json")


def get_text_hash(text: str, lang: str) -> str:
    key = f"{lang}:{text.strip().lower()}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def generate_audio_for_phrase(text: str, lang: str, index: dict) -> str:
    h = get_text_hash(text, lang)
    filename = f"{lang}_{h}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    if not os.path.exists(filepath):
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(filepath)
            print(f"  [+] Generated audio: {lang} -> {filename}")
        except Exception as e:
            print(f"  [!] Failed for {lang}: {e}")

    index[f"{lang}:{text}"] = filename
    return filename


def generate_all_audio():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    index = {}
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            index = {}

    # Load templates
    templates_path = os.path.join("assets", "packs", "templates.json")
    with open(templates_path, "r", encoding="utf-8") as f:
        templates = json.load(f)

    # Load packs for quick replies
    packs_path = os.path.join("assets", "packs", "packs.json")
    with open(packs_path, "r", encoding="utf-8") as f:
        packs = json.load(f)

    print("Generating offline audio for templates and quick replies across 6 languages...")

    # Generate for primary demo templates first
    demo_keys = ["TRAIN WHEN", "HELLO TICKET", "TICKET", "PLATFORM WHERE", "HELP", "WATER WHERE", "MONEY"]
    for key in demo_keys:
        if key in templates:
            translations = templates[key]
            for lang, text in translations.items():
                generate_audio_for_phrase(text, lang, index)

    # Generate for quick replies
    for pack_id, pack_data in packs.items():
        for reply in pack_data.get("quick_replies", []):
            for lang, text in reply.get("text", {}).items():
                generate_audio_for_phrase(text, lang, index)

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"\nAudio generation finished! Index contains {len(index)} audio mappings at {INDEX_PATH}")


if __name__ == "__main__":
    generate_all_audio()
