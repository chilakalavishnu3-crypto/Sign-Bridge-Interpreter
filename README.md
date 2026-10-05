# SignBridge — ISL counter interpreter

Two-way Indian Sign Language communication for public service counters
(railway ticket counters, hospital help desks).

- **Citizen → staff:** the citizen signs in front of the camera. Each sign is
  confirmed by holding it steady; lowering the hands sends the request. Staff
  see and hear it as a sentence in Tamil, Hindi, Telugu, Kannada, Malayalam or
  English.
- **Staff → citizen:** quick replies or free text are shown as large captions
  and performed by a 3D signing hand.
- **Fallback:** if the camera or recognition fails, citizens can tap signs or
  common requests in the **Sign guide**.

## Teach a sign (any sign language)

The built-in model knows 25 Indian Sign Language signs, trained on generated
data. To add **any** sign from any sign language, click **Teach a sign**:

1. Type a name and what it should say, and pick the language that text is in.
2. Press **Record 3 seconds** and hold the sign. Repeat 2–3 times, moving
   slightly each time.
3. Press **Save sign**.

From then on, signing it adds it to the request and speaks its meaning.
Taught signs are matched by nearest neighbour against your real recorded hand
landmarks (`backend/custom_signs.py`). They are stored in
`data/custom_signs.json` as landmark numbers only; no images are kept.
Teaching a built-in name is refused. Teaching an existing taught name adds
more examples. If matching is too strict or too loose, adjust
`SIGNBRIDGE_CUSTOM_MATCH_SCALE`.

## OpenAI (optional)

Set `OPENAI_API_KEY` in the environment before starting the server
(never in code or in this repo):

- **Sentence phrasing:** recognized signs that have no verified template
  are turned into natural sentences in all six languages. Only words are sent.
- **Ask AI** (button appears next to "Send now"): for when the local
  recognizer is unsure. After the citizen consents, 4 frames cropped around
  the hands are sent to `OPENAI_VISION_MODEL`. The answer is a suggestion that
  staff can add, speak, or turn into a taught sign. Images are never stored or
  logged. General vision models are not sign-language recognizers, so treat
  answers as guesses.

Model names are configurable (`OPENAI_MODEL`, `OPENAI_VISION_MODEL`).

## Run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8080
```

Open <http://localhost:8080> in Chrome or Edge and **allow camera access**.
The camera only works on `localhost` or HTTPS.

Add `?debug=1` to the URL to show live fps and latency.

Desktop (OpenCV window, no browser): `python desktop_runner.py`

## Architecture

```
browser (app/)                               server (backend/)
camera ─► 320px mirrored JPEG ─ WS /ws/live ─► vision.HandTracker (MediaPipe, per connection)
                                               └► vision.Classifier (126-feature model)
                                               └► session.SignSession (hold / pause rules)
3D overlay ◄─ landmarks, hold progress ◄─────── frame_result
request card ◄─ all languages ◄──────────────── sentences.SentenceService
                                                  template → Groq LLM (optional) → raw signs
speech ◄── GET /api/tts (pre-generated mp3 cache → gTTS)
```

- `backend/server.py`: routes, middleware (request IDs, security headers,
  CSP), rate limiting, and the WebSocket protocol (documented at the top).
- `backend/session.py`: time-based state machine. It is pure and unit-tested,
  and the desktop runner uses it too.
- `backend/sentences.py`: sentence composition with LLM validation and
  telemetry (model, prompt version, latency, tokens, outcome; no user text).
- `backend/vision.py`: frame decoding with size limits, landmark detection
  and classification. Detection runs in a worker thread.
- `app/js/`: ES modules, no build step. `main.js` (controller), `live.js`
  (frame pump with back-pressure and reconnect), `camera.js`, `speech.js`,
  `three/` (shared `HandRig`, live overlay, signing avatar).

Design system: "SignBridge Lite Terminal System" (from
`stitch_design_system_generator.zip`): bright tonal surfaces, Atkinson
Hyperlegible Next, 48px touch targets, and a 3px focus ring.

## Tests

```bash
python -m pytest -q
```

Covers normalization parity, model accuracy and latency, the REST API, the
session state machine, the WebSocket protocol (including malformed input),
LLM response validation, the TTS cache, and security headers and CORS.

## Training

`training/` holds the dataset generator, recorder, trainer, evaluator and
audio pre-generation. Artifacts go to `assets/model/` and `assets/audio/`.
