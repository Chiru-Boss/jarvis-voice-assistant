# JARVIS Voice Assistant – Codebase Analysis Report

> **Scope:** Analysis of the existing codebase as-of the current `main` branch.
> File and function references are exact; no assumptions or suggestions are included.

---

## Table of Contents

1. [Implemented Features](#1-implemented-features)
2. [Architecture](#2-architecture)
3. [What Is Working Correctly](#3-what-is-working-correctly)
4. [What Is Not Working Properly / Bugs](#4-what-is-not-working-properly--bugs)
5. [What Is Missing for Production-Level Readiness](#5-what-is-missing-for-production-level-readiness)
6. [Performance Observations](#6-performance-observations)
7. [Final Summary](#7-final-summary)

---

## 1. Implemented Features

### 1.1 Voice Input

**Files:** `core/audio_input.py`

`AudioInput` captures microphone audio using `sounddevice.RawInputStream` at:

| Parameter | Value |
|---|---|
| Sample rate | 16 000 Hz |
| Channels | 1 (mono) |
| Bit depth | 16-bit signed PCM |
| Frame size | 320 samples = 20 ms |

`webrtcvad.Vad` (`aggressiveness=2`) performs frame-by-frame Voice Activity Detection.
The recording loop in `AudioInput.listen()` (line 29) works as follows:

1. Blocks until the first speech frame is detected → prints `🗣️ Speech detected, recording…`
2. Continues recording until `silence_timeout` (default 1.5 s) of consecutive silence is
   detected **and** at least `min_duration` (default 2.0 s) of speech has been captured.
3. Enforces an absolute `max_duration` (default 10.0 s) guard to prevent infinite loops.
4. Returns raw PCM bytes.

`AudioInput.close()` (line 111) is a no-op (`pass`); `sounddevice` resources are released
by the `with sd.RawInputStream(…)` context manager at the end of each `listen()` call.

---

### 1.2 Speech-to-Text (STT)

**Files:** `core/speech_recognition.py`

| Property | Value |
|---|---|
| Library | `SpeechRecognition` (PyPI: `SpeechRecognition`) |
| Back-end | **Google Web Speech API** (`recognizer.recognize_google()`, line 50) |
| Model | Cloud (Google's servers) — **no local model** |
| CPU/GPU | N/A – cloud-based; local process is CPU-only |
| Offline | No – requires internet |

Workflow inside `recognize_speech()` (line 17):

1. Wraps raw PCM in a WAV container via `pcm_to_wav_bytes()` (line 6).
2. Creates `sr.Recognizer` and loads audio from an in-memory `io.BytesIO` buffer.
3. Calls `recognize_google(audio)` — a free-tier, unauthenticated HTTP request.
4. On `UnknownValueError` or `RequestError` retries up to `retries=2` additional times.
5. Returns `None` when all attempts fail.

> **Note:** The `README.md` (around line 140) describes "Whisper / Google STT" but no Whisper
> integration exists in the codebase. The root-level `speech_recognition.py` (line 3)
> contains only a stub: `def recognizing_speech(): pass`.

---

### 1.3 Wake Word Detection

**Files:** `core/wake_word.py`

`listen_for_wake_word(recognized_text, wake_word='jarvis')` (line 1):

- Performs a **case-insensitive substring search** (`wake_word.lower() in recognized_text.lower()`).
- Default wake word: `"jarvis"` (overridable via `WAKE_WORD` env var in `config/config.py`).
- Matches partial strings, e.g. `"hey jarvis"`, `"JARVIS!"`, `"my jarvis please"`.

`strip_wake_word(text, wake_word)` (line 23):

- Uses `re.sub` to remove the wake word from the text, enabling the remainder to be
  treated as a command (e.g. `"Jarvis, what time is it?"` → `"what time is it?"`).

**Important:** This is **not** a dedicated hotword engine. Detection happens **after** a
full STT transcription (a cloud round-trip). There is no audio-level wake word detection
that would allow JARVIS to skip the STT call for non-wake-word speech.

The two-step flow in `main.py` (lines 248–279):

```
1. listen() → full audio capture
2. recognize_speech() → cloud STT (network call)
3. listen_for_wake_word() → string search in result
```

Every ambient sound, background conversation, or noise that triggers the VAD incurs an
unnecessary STT API call.

---

### 1.4 AI Response System

**Files:** `core/llm_brain.py`, `core/system_prompt.py`

| Property | Value |
|---|---|
| Type | LLM (cloud) |
| Provider | NVIDIA NGC API |
| Endpoint | `https://integrate.api.nvidia.com/v1/chat/completions` |
| Default model | `meta/llama-3.1-8b-instruct` (configurable via `NVIDIA_LLM_MODEL`) |
| Protocol | OpenAI-compatible chat completions |
| Tool calling | OpenAI function-calling (parallel tool calls) |

`process_input()` (line 51) in `llm_brain.py`:

1. Builds a `messages` list: system prompt + conversation history + current user input.
2. Attaches available MCP tool schemas if an `mcp_client` is provided.
3. Enters a tool-calling loop (up to `max_tool_iterations=5` rounds):
   - On `finish_reason == 'stop'` or no `tool_calls` → returns the text response.
   - On tool calls → executes each via `mcp_client.call_tool()`, appends results, loops.
4. Returns a fallback message if the max-iteration limit is reached without a text reply.

`SYSTEM_PROMPT` in `core/system_prompt.py` is a static string that lists all 13 registered
tools with usage instructions and a step-by-step computer-control workflow.

Pattern-based personalisation: `AdaptiveAgent.get_pattern_summary()` (line 215 of
`adaptive_agent.py`) produces a short string (e.g. `"Most-used apps: brave, vscode; Common searches: python"`)
which is appended to the system prompt at inference time (`llm_brain.py`, line 103).

---

### 1.5 Text-to-Speech (TTS)

**Files:** `core/text_to_speech.py`

| Property | Value |
|---|---|
| Primary engine | ElevenLabs REST API (v1) |
| Fallback engine | `pyttsx3` (local, offline, OS TTS) |
| ElevenLabs model | `eleven_monolingual_v1` (configurable) |
| Character limit | 500 chars (`ELEVENLABS_MAX_TEXT_LENGTH`) |
| Audio playback | `winsound` on Windows; `ffplay` subprocess on Linux/macOS |

`speak()` (line 17) flow:

1. If `elevenlabs_api_key` and `elevenlabs_voice_id` are set → POST to ElevenLabs API.
2. On HTTP 200 → saves audio to `jarvis_response.mp3` in the current working directory.
3. Plays the file with `winsound` (Windows) or `ffplay` (Linux/macOS).
4. On any ElevenLabs failure (non-200, timeout, exception) → falls back to `pyttsx3`.
5. If no ElevenLabs credentials → calls `_speak_pyttsx3(text)` directly.

`_speak_pyttsx3(text)` (line 8):
- Creates a new `pyttsx3` engine instance on **every call** (`pyttsx3.init()`).
- Sets rate = 175 WPM, volume = 0.9.
- This pattern is stable but slightly wasteful (re-initialises the engine each time).

**Stability:**
- pyttsx3 fallback: stable for short responses; some OS TTS engines stutter on long text.
- ElevenLabs: depends on network; 15 s timeout is generous but not robust under flaky connectivity.

---

### 1.6 Continuous Listening Behaviour

**Files:** `main.py` (lines 238–342)

The main loop (`while True:`) is **always running** — JARVIS never truly "sleeps":

```
Loop iteration:
  ① AudioInput.listen()          ← blocks until speech + silence
  ② recognize_speech()           ← STT (cloud)
  ③ listen_for_wake_word()       ← text check
  ④ (optional) listen() again    ← if wake word given alone
  ⑤ handle_special_commands()    ← built-in commands
  ⑥ AdaptiveAgent.process_command()  ← intent detection + pattern learning
  ⑦ process_input()              ← LLM + tool calling
  ⑧ speak()                      ← TTS
  ⑨ memory.add_conversation()    ← persist to JSON
```

All steps are **sequential and synchronous** — no concurrent processing occurs.
The hand tracking loop, when enabled, runs in a separate daemon thread
(`core/hand_voice_integration.py`, line 107), which is the only true concurrency.

---

## 2. Architecture

### 2.1 Structure: Modular

The project is **highly modular**, not single-file. Component breakdown:

```
jarvis-voice-assistant/
├── main.py                      ← entry point + main event loop
├── config/
│   ├── config.py                ← env-backed CONFIG dict
│   ├── tools_config.py          ← MCP / approval mode settings
│   └── hand_tracking_config.py  ← hand tracking tuning params
├── core/
│   ├── audio_input.py           ← microphone capture + VAD
│   ├── speech_recognition.py    ← STT (Google cloud)
│   ├── wake_word.py             ← wake-word text matching
│   ├── llm_brain.py             ← NVIDIA LLM + tool-calling loop
│   ├── text_to_speech.py        ← ElevenLabs / pyttsx3 TTS
│   ├── system_prompt.py         ← static SYSTEM_PROMPT string
│   ├── mcp_server.py            ← in-process + HTTP tool server
│   ├── mcp_client.py            ← thin client for MCPServer
│   ├── tool_registry.py         ← tool schema + function store
│   ├── adaptive_agent.py        ← orchestrator for all subsystems
│   ├── app_controller.py        ← launch / close / focus apps
│   ├── system_executor.py       ← shell commands + file ops
│   ├── screen_vision.py         ← screenshot + OCR + UI detection
│   ├── behavior_learner.py      ← frequency/sequence analysis
│   ├── pattern_memory.py        ← persistent pattern JSON store
│   ├── prediction_engine.py     ← next-action prediction
│   ├── chat_session.py          ← session deduplication guard
│   ├── hand_voice_integration.py← hand tracking thread manager
│   ├── hand_tracking.py         ← MediaPipe hand landmark model
│   ├── gesture_recognition.py   ← gesture → name mapping
│   ├── hand_mouse_controller.py ← EMA-smoothed virtual mouse
│   ├── hand_ui_overlay.py       ← OpenCV overlay rendering
│   ├── swipe_keyboard.py        ← air-swipe text input
│   ├── browser_automation.py    ← Selenium browser control
│   ├── input_handler.py
│   ├── ui_detector.py
│   └── system_health.py
├── tools/
│   ├── __init__.py              ← build_registry() factory
│   ├── system_tools.py          ← 13 AI agent tools
│   ├── web_apis.py              ← weather, search, news, crypto
│   ├── laptop_control.py        ← brightness, volume, wifi, etc.
│   ├── knowledge_base.py        ← Q&A knowledge tool
│   └── home_automation.py       ← smart home stub
├── utils/
│   ├── memory.py                ← ConversationMemory (JSON)
│   ├── knowledge_store.py       ← vector-like knowledge storage
│   ├── app_finder.py            ← executable path resolver
│   ├── window_manager.py        ← platform window focus
│   ├── calibration.py           ← hand-tracking calibration I/O
│   ├── helpers.py               ← text truncation utilities
│   └── logger.py
├── data/
│   ├── user_patterns.json       ← persistent pattern DB
│   └── jarvis_memory.json       ← conversation history
└── tests/
    ├── test_adaptive_agent.py
    ├── test_swipe_keyboard.py
    └── test_chat_session.py
```

### 2.2 Component Interaction

```
main.py
  │
  ├─► AudioInput.listen()         → raw PCM bytes
  ├─► recognize_speech()          → text string
  ├─► listen_for_wake_word()      → bool
  │
  ├─► AdaptiveAgent.process_command()
  │     ├─► AppController          (open/close apps, click, type)
  │     ├─► BrowserAutomation      (Selenium search)
  │     ├─► SystemExecutor         (shell + file ops)
  │     ├─► ScreenVision           (screenshot + OCR)
  │     ├─► BehaviorLearner        (pattern analysis)
  │     ├─► PatternMemory          (data/user_patterns.json)
  │     └─► PredictionEngine       (next-action prediction)
  │
  ├─► process_input()              (llm_brain.py)
  │     ├─► NVIDIA LLM API         (cloud)
  │     └─► MCPClient.call_tool()
  │           └─► MCPServer.execute_tool()
  │                 └─► ToolRegistry → tool function
  │
  ├─► speak()                      (text_to_speech.py)
  │     ├─► ElevenLabs API         (cloud, primary)
  │     └─► pyttsx3                (local, fallback)
  │
  ├─► ConversationMemory           (utils/memory.py → JSON)
  └─► ChatSessionManager           (deduplication guard)

[background daemon thread, when HAND_TRACKING_ENABLED=true]
  HandVoiceIntegration._tracking_loop()
    ├─► HandTracker     (MediaPipe)
    ├─► GestureRecognizer
    ├─► HandMouseController  (pyautogui)
    ├─► SwipeKeyboard
    └─► HandUIOverlay   (OpenCV window)
```

**MCP Architecture detail:**
- `ToolRegistry` stores tool definitions (name, description, OpenAI schema, callable).
- `MCPServer` wraps the registry and provides (a) in-process `execute_tool()`, and
  (b) an optional HTTP server on `127.0.0.1:8765` (daemon thread) for external clients.
- `MCPClient` is a thin adapter — `call_tool()` delegates to `MCPServer.execute_tool()`
  and formats the result as a string for the LLM message thread.
- The LLM receives tool schemas via `mcp_client.get_available_tools()` (line 115 of
  `llm_brain.py`) — these are the OpenAI-format function definitions embedded in the API
  request payload.

---

## 3. What Is Working Correctly

| # | Feature | Evidence |
|---|---|---|
| 1 | **VAD-based audio capture** | `AudioInput.listen()` correctly uses webrtcvad 20ms frames and enforces min/max durations. |
| 2 | **Google STT with retry** | `recognize_speech()` catches `UnknownValueError` and `RequestError` and retries up to `retries+1` times. |
| 3 | **Wake word text matching** | Case-insensitive substring detection correctly handles `"Hey Jarvis"`, `"JARVIS!"`, etc. |
| 4 | **NVIDIA LLM integration** | `_call_api()` constructs a valid OpenAI-format payload; timeout and non-200 errors raise `RuntimeError`. |
| 5 | **Tool-calling loop** | Up to `MAX_TOOL_ITERATIONS=5` rounds; exits cleanly on `finish_reason='stop'` or no tool calls. |
| 6 | **ElevenLabs TTS with fallback** | HTTP errors and exceptions are caught; `_speak_pyttsx3` is called reliably as fallback. |
| 7 | **Persistent conversation memory** | `ConversationMemory` loads on startup, enforces `max_history`, saves on every `add_conversation()`. |
| 8 | **Pattern learning pipeline** | `BehaviorLearner.learn_from_interaction()` records commands, apps, searches, and workflows correctly. |
| 9 | **Prediction engine** | `PredictionEngine` uses frequency analysis and sequence detection to suggest next actions. |
| 10 | **Session deduplication** | `ChatSessionManager.should_suppress()` correctly blocks repeated confirmation phrases within the dedup window. |
| 11 | **Safety command blocking** | `SystemExecutor.execute_command()` checks `BLOCKED_COMMANDS` (fork bomb, `rm -rf /`, etc.) before execution. |
| 12 | **File operation undo stack** | Write, delete, and move operations push undo lambdas to a bounded `deque(maxlen=10)`. |
| 13 | **Cross-platform app launch** | `AppController.open_app()` uses the correct launch strategy for Windows / macOS / Linux. |
| 14 | **MCP HTTP server** | `MCPServer.start_http_server()` launches a daemon `HTTPServer` with `/tools`, `/execute`, `/sse` endpoints. |
| 15 | **Hand tracking (when enabled)** | `HandVoiceIntegration` correctly isolates the camera loop in a daemon thread; gesture dispatch works for pinch, fist, swipe, thumbs-up, peace. |
| 16 | **Special built-in commands** | `exit`, `memory`, `clear`, `tools`, `patterns`, `predict` are all handled before the LLM is consulted. |
| 17 | **Approval mode** | `MCPServer.execute_tool()` blocks tools with `requires_approval=True` when `APPROVAL_MODE` is enabled. |

---

## 4. What Is Not Working Properly / Bugs

### Bug 1 – Root `speech_recognition.py` is an empty stub

**File:** `speech_recognition.py` (project root, lines 1–4)

```python
# Speech Recognition Module
def recognizing_speech():
    pass  # Implement speech recognition logic here
```

This file shadows the name `speech_recognition` at the project root level. If any code
were to import from the wrong path, it would silently use the no-op stub instead of
`core/speech_recognition.py`. The stub also suggests incomplete earlier development.

---

### Bug 2 – README claims Whisper integration; none exists

**Files:** `README.md` (architecture section), `core/speech_recognition.py`

The README architecture diagram shows `core/speech_recognition.py – Whisper / Google STT`
but the actual implementation is Google STT **only**. No Whisper model, no `openai-whisper`
package in `requirements.txt`, and no local inference code exist anywhere in the codebase.

---

### Bug 3 – Wake word detection requires a full cloud STT round-trip for all audio

**File:** `main.py` lines 248–265; `core/wake_word.py`

Every VAD-triggered recording (including background noise and non-wake speech) is sent to
Google's STT API before the wake word is even checked. This means:

- Wasted API calls for any ambient audio the VAD picks up.
- Added latency (STT round-trip) even when the user did not say "Jarvis".
- No way to operate offline between wake-word moments.

---

### Bug 4 – `click_element()` by description is non-functional

**File:** `core/app_controller.py` lines 155–172

```python
def click_element(self, description: str) -> str:
    …
    return (
        f"⚠️ Could not locate '{description}' on screen automatically. "
        "Use click_at with exact coordinates instead."
    )
```

The function **always** returns a warning. The LLM's system prompt (`system_prompt.py`,
line 18) instructs it to call `click_element(description, x, y)`, but when no coordinates
are supplied, the tool silently fails with a non-fatal warning that the LLM may misinterpret
as success.

---

### Bug 5 – ElevenLabs audio file is hardcoded and never cleaned up

**File:** `core/text_to_speech.py` line 51

```python
audio_file = 'jarvis_response.mp3'
```

The file is always written to the current working directory. It is never deleted. On rapid
successive calls the file is overwritten mid-playback, which can cause audio corruption or
playback errors. It also leaves artefacts on disk indefinitely.

---

### Bug 6 – ElevenLabs TTS silently truncates responses at 500 characters

**File:** `core/text_to_speech.py` line 45

```python
'text': text[:ELEVENLABS_MAX_TEXT_LENGTH],  # 500 chars
```

Long LLM responses are truncated without any user notification. The user hears an
incomplete sentence while reading the full response in the console.

---

### Bug 7 – `pyttsx3.init()` is called on every TTS request

**File:** `core/text_to_speech.py` line 10

```python
def _speak_pyttsx3(text):
    engine = pyttsx3.init()
    …
    engine.runAndWait()
```

A new TTS engine is initialised and destroyed on each call. `pyttsx3.init()` spawns an
OS-level TTS driver process. On some platforms (especially Linux with `espeak`) this causes
a brief stutter or delay on every utterance. The engine instance is never reused.

---

### Bug 8 – `easyocr.Reader` is re-instantiated on every OCR fallback call

**File:** `core/screen_vision.py` lines 112–122

```python
reader = easyocr.Reader(['en'], verbose=False)
```

Creating an `easyocr.Reader` loads a deep learning model (hundreds of MB) from disk on
each call. In the fallback path this happens inside `get_ocr_text()` which is called every
time `get_screen_content()` is invoked. This makes the easyocr fallback path unusably slow.

---

### Bug 9 – `_session_start` internal timestamp leaks into the persisted pattern JSON

**File:** `core/pattern_memory.py` lines 122–125

```python
entry['_session_start'] = time.time()
```

The private `_session_start` key (prefixed `_` by convention for "internal") is stored
directly in the persistent `data/user_patterns.json` under each app entry. The key holds
a Unix timestamp float that has no meaning outside the current process and should not be
persisted.

---

### Bug 10 – Pattern memory performs a disk write on every single interaction

**File:** `core/pattern_memory.py` – `save()` called from `record_command()` (line 223),
`record_app_open()` (line 135), `record_app_close()` (line 153), `record_search()` (line 171),
`record_workflow()` (lines 193, 200)

Every user interaction triggers at least 2–3 full JSON serialisations to disk. As
`user_patterns.json` grows (up to 500 command history entries), this becomes a measurable
synchronous I/O delay on each command.

---

### Bug 11 – `AdaptiveAgent._is_app_running` / `is_app_running` duplication

**File:** `core/app_controller.py` line 255

```python
_is_app_running = is_app_running
```

There are two distinct calls: the public `is_app_running` used by `adaptive_agent.py`
(line 158 of `adaptive_agent.py`) and the private alias `_is_app_running`. The `open_app`
method calls `self._is_app_running(app_name)` (line 60), but the class also exposes
`is_app_running` as a public method. The dual names cause confusion but do not break
functionality.

---

## 5. What Is Missing for Production-Level Readiness

### 5.1 Offline / Local Speech-to-Text

The system is entirely **dependent on Google's cloud STT** for every voice interaction.
This means:

- No internet = no voice commands.
- Every spoken word is sent to a third-party server.
- STT latency is network-bound.

A production deployment would need a local STT model (e.g. OpenAI Whisper via
`openai-whisper` or `faster-whisper`) running on the CPU or GPU.

---

### 5.2 Dedicated Wake-Word Engine

The current text-match approach (Bug 3 above) is not suitable for production. A dedicated
hotword library (e.g. Porcupine, openWakeWord) would detect the wake word **at the audio
level** without sending audio to the cloud, eliminating wasted STT calls.

---

### 5.3 API Key Validation at Startup

`config/config.py` silently returns an empty string when `NVIDIA_API_KEY` is not set.
The first LLM call then fails with a 401 error at runtime. There is no startup check that
validates required keys are present and reachable before entering the main loop.

---

### 5.4 Structured Logging

The entire codebase uses `print()` for user-facing output and `logging.getLogger()` for
internal events, but the logging system is never configured (no `basicConfig`, no handlers,
no log level set). All `logger.info/debug/warning` calls produce no output by default,
and `print()` output has no timestamps, levels, or correlation IDs.

---

### 5.5 Error Monitoring and Alerting

There is no integration with any error-tracking system (Sentry, Datadog, etc.). A silent
exception in a tool function returns a string like `"❌ Command failed: <exc>"` back to the
LLM but is never reported to an operator.

---

### 5.6 Authentication and Multi-User Support

There is no authentication layer. Anyone with physical access to the microphone can
control the system. All state (memory, patterns) is single-user with no isolation.

---

### 5.7 Rate Limiting

There is no rate limiting for:

- NVIDIA LLM API calls (pay-per-token or rate-limited).
- ElevenLabs TTS API calls (monthly character quota).
- Google STT (daily free-tier limit of 60 minutes).

A runaway loop or adversarial input could exhaust quotas.

---

### 5.8 TTS Audio File Management

`jarvis_response.mp3` (Bug 5) is written to the working directory and never cleaned up.
For production use this file should be written to a secure temporary path
(`tempfile.mkstemp` or `tempfile.NamedTemporaryFile`) and deleted after playback.

---

### 5.9 Test Coverage

Only three test files exist:

| File | Coverage |
|---|---|
| `tests/test_adaptive_agent.py` | `AdaptiveAgent` |
| `tests/test_swipe_keyboard.py` | `SwipeKeyboard` |
| `tests/test_chat_session.py` | `ChatSessionManager` |

The following have **no tests**:
`AudioInput`, `recognize_speech`, `speak`, `listen_for_wake_word`,
`process_input`, `MCPServer`, `MCPClient`, `ToolRegistry`, `ConversationMemory`,
`PatternMemory`, `BehaviorLearner`, `AppController`, `SystemExecutor`, `ScreenVision`,
all tool modules.

---

### 5.10 Graceful Degradation and Health Checks

There is no health-check endpoint, no readiness probe, and no self-diagnostic system.
If the NVIDIA API is unreachable, JARVIS returns a timeout error but does not retry with
a local fallback LLM. If pyttsx3 also fails, TTS is silently lost.

---

### 5.11 Configuration Schema Validation

`config/config.py` accepts arbitrary string values from environment variables with
minimal validation (`_safe_int` for integer fields only). Boolean values are parsed with
`== 'true'` (line 26), which will silently default to `False` for `'True'` or `'TRUE'`.
No schema validation (e.g. `pydantic`, `cerberus`) is applied.

---

### 5.12 Thread Safety for Pattern Memory

`PatternMemory.save()` is called synchronously on the main thread during each interaction.
If hand tracking were to trigger pattern recording from its daemon thread simultaneously,
there would be a race condition on `self._data` and the JSON file.

---

## 6. Performance Observations

### 6.1 Latency Breakdown (per voice interaction, estimated)

| Stage | Typical Latency |
|---|---|
| `AudioInput.listen()` | 2–12 s (user-dependent speech duration) |
| `recognize_speech()` (Google STT) | 0.5–3 s (network round-trip) |
| Wake-word check (`listen_for_wake_word`) | < 1 ms |
| `AdaptiveAgent.process_command()` | 50–500 ms (psutil + regex) |
| `process_input()` (NVIDIA LLM, no tools) | 2–8 s (network round-trip) |
| Tool call round-trip (per tool) | 0.1–2 s each (up to 5 rounds) |
| `speak()` – ElevenLabs | 1–3 s (network + playback) |
| `speak()` – pyttsx3 | 0.5–2 s (local synthesis, blocking) |
| Memory + pattern save (disk I/O) | 10–100 ms |
| **Total (best case, no tools)** | **~6–16 s end-to-end** |
| **Total (with 2 tool calls)** | **~10–24 s end-to-end** |

---

### 6.2 GPU vs CPU Usage

| Component | Compute location |
|---|---|
| Audio capture | CPU (sounddevice, low load) |
| VAD (webrtcvad) | CPU (very lightweight) |
| STT (Google) | **Cloud (Google GPU)** |
| LLM inference (NVIDIA meta/llama-3.1-8b) | **Cloud (NVIDIA GPU)** |
| TTS (ElevenLabs) | **Cloud (ElevenLabs GPU)** |
| TTS (pyttsx3) | CPU (OS TTS, negligible) |
| Pattern analysis, prediction | CPU (pure Python) |
| OCR (pytesseract) | CPU only (single-threaded) |
| OCR (easyocr fallback) | CPU or GPU (if CUDA present) |
| Hand tracking (MediaPipe) | CPU (or GPU if TensorFlow backend available) |

**Local GPU is not used by any core JARVIS component.** All AI inference runs in the
cloud. A machine with a GPU will see no benefit to voice pipeline latency.

---

### 6.3 Key Inefficiencies

| Inefficiency | Location | Impact |
|---|---|---|
| Every VAD-triggered audio → cloud STT | `main.py` lines 248–257 | Wasted API calls, added latency |
| `pyttsx3.init()` on every TTS call | `text_to_speech.py` line 10 | Re-creates OS TTS driver each time |
| Pattern memory disk write per interaction | `pattern_memory.py` | 2–5 JSON serialisations per command |
| `easyocr.Reader` re-created on each OCR fallback | `screen_vision.py` line 115 | Multi-second model load on each call |
| `AppController.open_app()` calls `time.sleep(2.0)` | `app_controller.py` line 86 | Unconditional 2 s blocking delay after every launch |
| Main loop is fully synchronous | `main.py` | No pipelining; LLM waits for audio; TTS waits for LLM |
| Conversation history re-sent to LLM every request | `llm_brain.py` line 111 | Token cost grows linearly with history length |
| `PatternMemory._session_start` persisted to disk | `pattern_memory.py` line 125 | Stale float values accumulate in JSON |

---

## 7. Final Summary

### Current system level

```
┌─────────────────────────────────────────────────────────┐
│           INTERMEDIATE                                  │
│                                                         │
│  Beyond beginner:                                       │
│  ✔ Fully modular architecture (25+ modules)            │
│  ✔ MCP tool server/client pattern                      │
│  ✔ LLM with multi-round tool-calling                   │
│  ✔ Persistent memory + pattern learning                 │
│  ✔ Hand tracking integration (MediaPipe)               │
│  ✔ Session deduplication guard                         │
│  ✔ Safety command blocking + undo stack                │
│  ✔ Cross-platform app control                          │
│                                                         │
│  Not yet advanced:                                      │
│  ✘ No local / offline STT (Whisper not implemented)    │
│  ✘ Wake word requires full cloud STT on every frame    │
│  ✘ No production monitoring, auth, or rate limiting    │
│  ✘ Critical tools broken (click_element by description)│
│  ✘ No structured logging or startup validation         │
│  ✘ Minimal test coverage                               │
│  ✘ Fully synchronous; no latency pipelining            │
└─────────────────────────────────────────────────────────┘
```

The codebase demonstrates sound software engineering practices — clear separation of
concerns, well-named modules, docstrings, typed function signatures, and a coherent
data flow — placing it solidly at the **Intermediate** level. The architecture is
significantly more sophisticated than a single-file script, and advanced features
(adaptive learning, MCP tool calling, hand tracking) are genuinely implemented. However,
core production requirements (offline STT, true hotword detection, authentication,
observability, and comprehensive test coverage) are absent or incomplete.
