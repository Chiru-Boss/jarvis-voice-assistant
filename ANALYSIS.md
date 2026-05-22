# JARVIS Voice Assistant – Codebase Analysis Report

> **Scope:** Analysis of the `main` branch as of May 2026.  
> **Method:** Direct file and function inspection — no assumptions, no rebuilding.  
> All section references include exact file paths and function/class names.

---

## 1. Implemented Features

### 1.1 Voice Input

**File:** `core/audio_input.py` — class `AudioInput`

The assistant captures audio with `sounddevice` (`sd.RawInputStream`) at
**16 kHz, mono, 16-bit PCM** using 20 ms frames (320 samples each, as required by
WebRTC VAD).

`AudioInput.listen()` implements a smart silence-detection loop:

1. Reads frames continuously from the microphone.
2. Passes each frame to `webrtcvad.Vad` (aggressiveness level 2 by default).
3. Starts accumulating once speech is detected.
4. Stops after `silence_timeout` (default 1.5 s) of consecutive silence **and** after at
   least `min_duration` (default 2.0 s) of audio has been captured, preventing early
   cut-off for short words.
5. Enforces a hard `max_duration` timeout (default 10 s) to prevent infinite loops.

Returns raw PCM bytes to the caller.

### 1.2 Speech-to-Text (STT)

**File:** `core/speech_recognition.py` — function `recognize_speech`

| Aspect | Detail |
|---|---|
| **Library** | `SpeechRecognition` (`speech_recognition` PyPI package) |
| **Backend** | `recognizer.recognize_google()` — Google Cloud STT free tier |
| **GPU usage** | None — entirely cloud-based, no local GPU or CPU inference |
| **Input format** | Raw PCM is wrapped in a WAV container via `pcm_to_wav_bytes()` |
| **Retry logic** | Up to `retries` (default 2) additional attempts on failure |
| **Offline** | No — requires internet connection for every utterance |

> **README vs Code discrepancy:** The README architecture diagram lists
> "Whisper / Google STT" but **Whisper is not implemented anywhere in the
> codebase**.  Only Google STT is used.

### 1.3 Wake Word Detection

**File:** `core/wake_word.py` — functions `listen_for_wake_word`, `strip_wake_word`

The wake-word system is a **post-transcription text search**, not a dedicated
acoustic model:

1. The full audio clip is sent to Google STT (`recognize_speech`).
2. The transcribed text is checked with a case-insensitive `str.lower() in` search
   (`listen_for_wake_word`).
3. If the wake word is absent the loop continues; if present,
   `strip_wake_word` removes it via `re.sub` and the remainder becomes the
   command.
4. If the transcription contains *only* the wake word (no command), a second
   `audio_input.listen()` + `recognize_speech` pass is triggered for the
   actual question.

**Configurable:** The wake word is read from `CONFIG['WAKE_WORD']`
(`config/config.py`), defaulting to `"jarvis"`.

There is no dedicated always-on acoustic wake-word engine (e.g. Porcupine,
OpenWakeWord, Snowboy).  Every audio clip — including clips that contain no
wake word — incurs a full Google STT network round-trip.

### 1.4 AI Response System

**File:** `core/llm_brain.py` — function `process_input`  
**File:** `core/system_prompt.py` — constant `SYSTEM_PROMPT`

| Aspect | Detail |
|---|---|
| **Type** | Full LLM (not rule-based) |
| **Provider** | NVIDIA NIM API (`https://integrate.api.nvidia.com/v1/chat/completions`) |
| **Default model** | `meta/llama-3.1-8b-instruct` (configurable via `NVIDIA_LLM_MODEL`) |
| **Protocol** | OpenAI-compatible `chat/completions` with function-calling (`tool_choice: "auto"`) |
| **Tool loop** | Up to `MAX_TOOL_ITERATIONS` (default 5) rounds of tool calls per request |
| **Context** | Recent conversation history (`ConversationMemory.get_recent_messages()`) |
| **Personalisation** | Learned user patterns injected into the system prompt via `AdaptiveAgent.get_pattern_summary()` |

`process_input` builds a message list (system prompt → conversation history →
user message), calls `_call_api`, and loops while the model returns `tool_calls`
objects, executing each via `MCPClient.call_tool` and appending results until
the model produces a final text response or the iteration cap is reached.

### 1.5 Text-to-Speech (TTS)

**File:** `core/text_to_speech.py` — functions `speak`, `_speak_pyttsx3`

Two-tier fallback design:

| Tier | Library / Service | Condition |
|---|---|---|
| Primary | ElevenLabs REST API (`/v1/text-to-speech/{voice_id}`) | `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` are set |
| Fallback | `pyttsx3` (local, offline) | ElevenLabs unavailable or returns non-200 |

**ElevenLabs path:**
- Text truncated to 500 characters (`ELEVENLABS_MAX_TEXT_LENGTH`).
- Voice settings: `stability=0.5`, `similarity_boost=0.75`.
- Audio saved to `jarvis_response.mp3` in the **current working directory**.
- Playback: `ffplay -nodisp -autoexit` (Linux/macOS) or `winsound` (Windows).
- Falls back to pyttsx3 on any exception.

**pyttsx3 path (`_speak_pyttsx3`):**
- Rate: 175 wpm, volume: 0.9.
- A new `pyttsx3` engine instance is created on **every call** (`pyttsx3.init()`).

**Stability concern:** Controlled — `requests.post` timeout is set to 15 s;
all exceptions fall through to pyttsx3.  The `winsound`/`ffplay` subprocess path
can silently fail if `ffplay` is not installed.

### 1.6 Continuous Listening Behaviour

**File:** `main.py` — function `main`

The assistant runs in an **infinite `while True` loop**:

```
capture audio  →  STT  →  wake word check  →  (optional 2nd capture for command)
→  special command handler  →  AdaptiveAgent pre-processing  →  LLM  →  TTS  →  memory save
→  repeat
```

It is **always listening** — there is no idle/sleep state between cycles.
Each iteration starts a new `audio_input.listen()` call immediately after the
previous response is spoken.

The `ChatSessionManager` (`core/chat_session.py`) adds a deduplication guard
that suppresses bare acknowledgement phrases (e.g., "yes", "confirmed") from
triggering a new LLM call within a 120-second window after the previous session.

---

## 2. Architecture

### 2.1 Single-File vs Modular

The project is **fully modular**.  There is no single monolithic file; logic is
distributed across well-separated packages:

```
jarvis-voice-assistant/
├── main.py                   – entry point, main loop, wires everything together
├── config/
│   ├── config.py             – core env-driven configuration (CONFIG dict)
│   ├── tools_config.py       – MCP / tool server settings (TOOLS_CONFIG dict)
│   └── hand_tracking_config.py – hand tracking parameters
├── core/
│   ├── audio_input.py        – sounddevice + WebRTC VAD capture
│   ├── speech_recognition.py – Google STT wrapper
│   ├── wake_word.py          – wake word text detection
│   ├── llm_brain.py          – NVIDIA LLM API + MCP tool-call loop
│   ├── system_prompt.py      – static LLM system prompt
│   ├── text_to_speech.py     – ElevenLabs / pyttsx3 TTS
│   ├── mcp_server.py         – in-process + optional HTTP tool server
│   ├── mcp_client.py         – thin client over MCPServer
│   ├── tool_registry.py      – ToolDefinition / ToolRegistry
│   ├── adaptive_agent.py     – orchestrator: intent detection + automation
│   ├── app_controller.py     – launch / close / click / type / key press
│   ├── system_executor.py    – shell commands + file ops + undo stack
│   ├── behavior_learner.py   – frequency / sequence / time pattern analysis
│   ├── pattern_memory.py     – JSON-backed pattern persistence
│   ├── prediction_engine.py  – next-action prediction
│   ├── screen_vision.py      – PIL screenshot + pytesseract/easyocr OCR
│   ├── browser_automation.py – Selenium browser search
│   ├── chat_session.py       – session deduplication guard
│   ├── hand_tracking.py      – MediaPipe hand landmark detection
│   ├── hand_mouse_controller.py – EMA-smoothed hand-to-mouse mapping
│   ├── hand_voice_integration.py – background thread for hand tracking
│   ├── gesture_recognition.py – gesture classification from landmarks
│   ├── swipe_keyboard.py     – mid-air swipe typing overlay
│   ├── hand_ui_overlay.py    – OpenCV heads-up display
│   ├── ui_detector.py        – OpenCV UI element finder
│   ├── input_handler.py      – unified input routing
│   └── system_health.py      – health check utilities
├── tools/
│   ├── __init__.py           – build_registry() factory
│   ├── system_tools.py       – CPU/RAM/screenshot + AI agent MCP tools
│   ├── laptop_control.py     – open_application, execute_command, file_operations
│   ├── web_apis.py           – weather, web search, news, crypto
│   ├── knowledge_base.py     – personal knowledge store CRUD
│   └── home_automation.py    – smart-home placeholder (Hub API)
└── utils/
    ├── memory.py             – ConversationMemory (JSON persistence)
    ├── knowledge_store.py    – KnowledgeStore (JSON persistence)
    ├── app_finder.py         – cross-platform executable path lookup
    ├── window_manager.py     – window focus utilities
    ├── helpers.py            – shared helpers (truncate, etc.)
    ├── logger.py             – logging setup
    └── calibration.py        – hand-tracking calibration load/save
```

### 2.2 Component Interaction

```
main.py
  │
  ├─► AudioInput.listen()          [core/audio_input.py]
  │       └── sounddevice + webrtcvad
  │
  ├─► recognize_speech()           [core/speech_recognition.py]
  │       └── SpeechRecognition → Google STT API
  │
  ├─► listen_for_wake_word()       [core/wake_word.py]
  │       └── text-based substring check
  │
  ├─► handle_special_commands()    [main.py]
  │       └── ChatSessionManager  [core/chat_session.py]
  │
  ├─► AdaptiveAgent.process_command()  [core/adaptive_agent.py]
  │       ├── AppController        [core/app_controller.py]
  │       │       └── psutil + pyautogui + subprocess
  │       ├── BrowserAutomation    [core/browser_automation.py]
  │       │       └── Selenium
  │       ├── SystemExecutor       [core/system_executor.py]
  │       │       └── subprocess + shutil
  │       ├── BehaviorLearner      [core/behavior_learner.py]
  │       │       └── PatternMemory [core/pattern_memory.py]
  │       └── PredictionEngine     [core/prediction_engine.py]
  │
  ├─► process_input()              [core/llm_brain.py]
  │       ├── NVIDIA NIM API (Llama 3.1 8B)
  │       └── MCPClient.call_tool() [core/mcp_client.py]
  │               └── MCPServer.execute_tool() [core/mcp_server.py]
  │                       └── ToolRegistry [core/tool_registry.py]
  │                               ├── tools/system_tools.py
  │                               ├── tools/laptop_control.py
  │                               ├── tools/web_apis.py
  │                               ├── tools/knowledge_base.py
  │                               └── tools/home_automation.py
  │
  ├─► speak()                      [core/text_to_speech.py]
  │       ├── ElevenLabs API + ffplay/winsound
  │       └── pyttsx3 (fallback)
  │
  └─► ConversationMemory           [utils/memory.py]
          └── jarvis_memory.json
```

---

## 3. What Is Working Correctly

1. **Audio capture pipeline** (`core/audio_input.py`): The VAD-gated recording
   loop is well-implemented.  Frame-size validation, min-duration guard, and
   max-duration timeout all function as designed.

2. **STT with retry** (`core/speech_recognition.py`): PCM→WAV wrapping is
   correct; retry logic handles transient Google STT failures gracefully.

3. **Wake word detection** (`core/wake_word.py`): The text-match approach works
   reliably once STT produces a transcript.  `strip_wake_word` uses a compiled
   regex and correctly handles punctuation (`lstrip(',')`) and case variants.

4. **LLM tool-call loop** (`core/llm_brain.py`): The multi-round tool-calling
   loop with `max_tool_iterations` guard is solid.  Conversation history is
   threaded in correctly.  Timeout and general exceptions are caught and returned
   as user-facing strings.

5. **TTS fallback chain** (`core/text_to_speech.py`): All exceptions from
   ElevenLabs path fall through to pyttsx3, so TTS always produces output.

6. **MCP architecture** (`core/tool_registry.py`, `core/mcp_server.py`,
   `core/mcp_client.py`): Tool registration, schema generation, in-process
   execution, and HTTP/SSE server are correctly structured.  `approval_mode`
   gate functions as expected for destructive tools.

7. **Persistent conversation memory** (`utils/memory.py`): JSON load/save with
   `max_history` bound, `get_recent_messages()` returning OpenAI-style dicts —
   all correct.

8. **Adaptive agent intent routing** (`core/adaptive_agent.py`): Regex-based
   intent detection for open/close/search/command is functional; pattern
   learning is accumulated per session and passed to the LLM.

9. **System executor safety list** (`core/system_executor.py`): `BLOCKED_COMMANDS`
   prevents the most common destructive shell commands.  Undo stack with backup
   files is implemented correctly.

10. **Session deduplication** (`core/chat_session.py`): `ChatSessionManager`
    correctly suppresses redundant re-triggers from bare confirmation phrases
    within the configurable window.

11. **Cross-platform app launching** (`core/app_controller.py`): Windows
    (`os.startfile`), macOS (`open -a`), and Linux (`subprocess.Popen`) paths
    are all present.

12. **Hand tracking (optional)** (`core/hand_voice_integration.py`): Runs in a
    daemon background thread; failure to import `mediapipe`/`opencv-python` is
    caught and hand tracking is silently disabled without crashing the main loop.

---

## 4. What Is NOT Working Properly / Bugs

### 4.1 Wake word forces a full STT round-trip for every audio clip
**File:** `main.py` lines 249–265; `core/speech_recognition.py`

Every audio clip — even one that contains no wake word at all — is sent to
Google STT before the wake word is checked.  There is no acoustic pre-filter.
If the user is silent or says something random, a full network request is still
made and the result discarded.  This is a functional correctness problem when
there is no internet connection: the assistant cannot detect the wake word at all.

### 4.2 `click_element` by description always fails
**File:** `core/app_controller.py` lines 155–172 — `AppController.click_element`

The method unconditionally returns:
```
"⚠️ Could not locate '{description}' on screen automatically.
 Use click_at with exact coordinates instead."
```
despite the LLM system prompt (`core/system_prompt.py`) instructing the model to
call `click_element(description, x, y)` as the primary click mechanism.  The LLM
will call this tool expecting it to work; the user receives a warning instead of
an actual click.

### 4.3 Duplicate tool names registered in the registry
**File:** `tools/__init__.py` — `build_registry()`

`build_registry()` calls both `laptop_control.register_tools(registry)` and
`system_tools.register_tools(registry)`.  Both modules register tools with
overlapping names:

| Tool name | Registered by `laptop_control` | Registered by `system_tools` |
|---|---|---|
| `file_operations` | `create/read/move/delete` ops | `read/write/delete/move/copy/list` ops |
| `execute_command` | basic `subprocess.run` wrapper | `SystemExecutor.execute_command` with blocked-command list |

Because `ToolRegistry._tools` is a plain `dict`, the second registration
silently **overwrites** the first.  The `laptop_control` versions are
unreachable after startup.  The discrepancy in supported operations between the
two `file_operations` implementations (e.g., `create` is only in
`laptop_control`) means `create` is permanently unavailable to the LLM.

### 4.4 `pyttsx3` engine re-initialised on every TTS call
**File:** `core/text_to_speech.py` lines 8–14 — `_speak_pyttsx3`

```python
def _speak_pyttsx3(text):
    engine = pyttsx3.init()   # ← new engine every call
    ...
    engine.runAndWait()
```

`pyttsx3.init()` is not safe to call repeatedly in the same process.  On Linux
(espeak backend) and Windows (SAPI) this creates a new driver thread each time,
which can leak handles and eventually raise
`RuntimeError: run loop already started` after enough calls.

### 4.5 `jarvis_response.mp3` written to the current working directory
**File:** `core/text_to_speech.py` line 51

```python
audio_file = 'jarvis_response.mp3'
```

The file is written to wherever `os.getcwd()` happens to be at runtime —
not a predictable temp directory.  Each response overwrites the previous file
without cleanup, and it is included/polluted in the project root if the assistant
is run from there.

### 4.6 Missing guard when `NVIDIA_API_KEY` is empty
**File:** `core/llm_brain.py`; `config/config.py` line 17

`CONFIG['NVIDIA_API_KEY']` defaults to `''` if the env var is unset.  `process_input`
passes this empty string to `_call_api`, which constructs
`Authorization: Bearer ` (empty bearer token).  The API returns a 401 that is
then caught by the generic `except Exception` in `process_input` and surfaced to
the user as a raw error string rather than a clear "please set your API key"
message.

### 4.7 `ffplay` not available by default on Windows or macOS
**File:** `core/text_to_speech.py` lines 61–65

`ffplay` is part of FFmpeg and is not installed on most machines by default.
On failure the code catches the exception and falls through to pyttsx3, so TTS
still works — but the ElevenLabs audio that was downloaded and paid for is
silently discarded.  There is no warning to the user that ElevenLabs playback
failed despite `ffplay` not being found.

### 4.8 Root-level duplicate files (`audio_input.py`, `speech_recognition.py`)
**Root directory** contains `audio_input.py` and `speech_recognition.py` that
appear to be earlier standalone versions.  They are not imported by `main.py`
(which uses `core/audio_input.py` and `core/speech_recognition.py`), but they
create confusion and may be accidentally imported if the package path resolution
changes.

### 4.9 `web_search` returns limited results from DuckDuckGo Instant Answer API
**File:** `tools/web_apis.py` — `web_search`

The DuckDuckGo Instant Answer API (`api.duckduckgo.com`) only returns
pre-computed "instant answers" and related topic snippets — it is **not** a full
web search returning ranked pages.  For the majority of real-world queries it
returns zero or very few results (`[{'snippet': 'No results found for the
query.'}]`), making the `web_search` tool unreliable in practice.

### 4.10 `home_automation` tools are placeholder-only
**File:** `tools/home_automation.py`

All home automation tools return placeholder messages unless `HOME_AUTOMATION_URL`
is configured.  They are registered in the MCP registry and exposed to the LLM,
which may confidently tell the user it toggled a light when the hub URL is empty
and no real action occurred.

---

## 5. What Is Missing for Production-Level Readiness

### 5.1 Always-on acoustic wake-word engine
The current STT-first, text-search wake word approach requires a Google STT
round-trip for every ambient audio clip.  Production systems use a dedicated
always-on, low-power acoustic model (e.g., Porcupine, OpenWakeWord, Snowboy)
that runs locally and only wakes the full pipeline on a positive detection.

### 5.2 Offline / local STT option
The Google STT dependency means the assistant cannot function without internet.
For privacy and reliability a local STT engine (e.g., OpenAI Whisper via
`faster-whisper`, Vosk) should be the default or primary path, with Google STT
as an optional cloud upgrade.

### 5.3 Structured logging and log levels
The codebase uses bare `print()` statements throughout (80+ occurrences across
`main.py`, `core/`, and `tools/`).  Production systems require structured logging
(e.g., Python `logging` with configurable levels) so that debug output can be
suppressed in production without code changes.  `utils/logger.py` exists but is
not actually used by any module.

### 5.4 Authentication and API key validation at startup
No startup validation of required API keys (`NVIDIA_API_KEY`, etc.).  The user
only discovers a missing key when the first LLM call fails mid-conversation.
A startup health check that validates all required credentials and clearly prints
what is missing and how to fix it is essential for a production-grade entry point.

### 5.5 `pyttsx3` engine lifecycle management
`_speak_pyttsx3` must initialise the engine once at module/class level and reuse
it, or use a lock-protected singleton, to avoid thread-safety issues and resource
leaks on long-running sessions.

### 5.6 Reliable `click_element` by description
`AppController.click_element` always returns a warning.  For the LLM tool-calling
UI automation workflow described in `SYSTEM_PROMPT` to actually work, this
method needs a working implementation (e.g., using pytesseract OCR coordinates
from `ScreenVision`, or a pyautogui image-search fallback).

### 5.7 Duplicate tool name deduplication in `ToolRegistry`
`ToolRegistry.register` silently overwrites existing entries with the same name.
At minimum it should raise a `ValueError` or emit a warning when a duplicate name
is registered.  Longer-term, the overlapping `file_operations` and
`execute_command` registrations in `laptop_control` and `system_tools` should be
consolidated into a single canonical implementation.

### 5.8 Audio playback abstraction
ElevenLabs playback depends on `ffplay` being on `$PATH`.  A cross-platform audio
playback abstraction (e.g., `playsound`, `pygame.mixer`, or `sounddevice` for
reading the mp3) that does not require a system-level binary would make the
assistant deployable without manual FFmpeg installation.

### 5.9 Unit and integration test coverage
Only a subset of modules have tests (`tests/test_adaptive_agent.py`,
`tests/test_swipe_keyboard.py`, `tests/test_chat_session.py`).  Core voice
pipeline modules (`audio_input`, `speech_recognition`, `llm_brain`,
`text_to_speech`) have no automated tests.

### 5.10 Docker / containerisation gap
A `Dockerfile` and `docker-compose.yml` are present, but audio input inside a
container requires device passthrough (`/dev/snd` on Linux, or PulseAudio socket)
and is not documented or configured.  Hand tracking requires a video device as
well.  The container is effectively non-functional for interactive voice use
without additional host-side configuration that is not documented.

### 5.11 Secrets management
The `.env.example` shows all keys in a flat `.env` file.  For production use,
secrets should be sourced from a secrets manager (AWS Secrets Manager, HashiCorp
Vault, etc.) rather than a plain-text dotenv file, or at minimum the README
should include a warning about not committing `.env` to source control.

### 5.12 Error recovery and reconnect
There is no automatic recovery from prolonged failures (e.g., Google STT returning
errors for 10 consecutive cycles, or NVIDIA API returning 429 rate-limit
responses).  The loop simply prints the error and retries immediately, which
could result in rapid-fire API calls during an outage.  Exponential back-off
and a circuit breaker would be needed for production stability.

---

## 6. Performance Observations

### 6.1 Latency breakdown (approximate, per cycle)

| Stage | Typical duration | Notes |
|---|---|---|
| Audio capture | 2–10 s | Depends on utterance length + 1.5 s silence timeout |
| Google STT | 0.5–2 s | Network round-trip; happens **twice** when wake word is standalone |
| Wake word check | < 1 ms | In-memory string operation |
| AdaptiveAgent pre-processing | 50–500 ms | Regex match + optional app launch (2 s wait) |
| NVIDIA LLM API | 3–15 s | Depends on model load, token count, and tool iterations |
| Tool calls (if any) | 0.1–5 s each | Up to 5 iterations × tool latency |
| ElevenLabs TTS generation | 1–4 s | Network round-trip; pyttsx3 fallback ≈ 0.1 s |
| TTS playback | Duration of speech | Blocking — no async playback |
| Memory save | < 10 ms | JSON file write |

**Minimum end-to-end latency** (short question, no tool calls, pyttsx3 TTS):
≈ **5–18 seconds** from end of speaking to start of JARVIS reply.

**With ElevenLabs and tool calls:** easily **15–30 seconds** or more.

### 6.2 GPU vs CPU

| Component | Compute |
|---|---|
| Audio VAD (`webrtcvad`) | CPU — negligible |
| Google STT | Cloud GPU (not local) |
| NVIDIA LLM | Cloud GPU (not local) |
| ElevenLabs TTS | Cloud GPU (not local) |
| pyttsx3 TTS | CPU — fast |
| pytesseract OCR | CPU — slow (100–500 ms per frame) |
| OpenCV UI detection | CPU — fast |
| MediaPipe hand tracking | CPU (or GPU if CUDA available, but not configured) |

**There is zero local GPU usage.**  All heavy inference runs in the cloud.
On a machine without internet the assistant is completely non-functional
(wake word detection, STT, and LLM all require network).

### 6.3 Key Inefficiencies

1. **Double STT cost for bare wake-word utterances** (`main.py` lines 270–278):
   When the user says only "Jarvis" (no inline command), the system calls
   `recognize_speech` twice — once for the wake word check and again for the
   subsequent command capture.

2. **Blocking TTS playback:** `_speak_pyttsx3` and the `ffplay` subprocess call
   are both synchronous and blocking.  The main loop is frozen for the entire
   duration of the spoken response.  No audio streaming or async playback.

3. **`pyttsx3.init()` per call:** Creates a new engine (and its underlying
   COM/SAPI/espeak instance) on every TTS invocation.  Engine startup overhead
   is 50–200 ms and increases over a long session.

4. **ScreenVision called on every command** (`core/adaptive_agent.py` line 197):
   `_get_screen_summary()` calls `ScreenVision.get_screen_content()` for every
   user command, triggering a screenshot + OCR.  pytesseract is CPU-intensive
   (100–500 ms) and is invoked even for purely conversational questions.

5. **All tool results are string-serialised and sent back to the LLM:**
   Large tool outputs (e.g., `get_screen_content` returning thousands of OCR
   characters) are injected verbatim into the message thread, consuming LLM
   context tokens and increasing API cost and latency.

---

## 7. Final Summary

### Current System Level: **Intermediate**

#### Justification

**Above beginner because:**
- Fully modular architecture with clearly separated concerns across 25+ modules.
- Real LLM integration (NVIDIA NIM / Llama 3.1) with multi-round tool calling,
  not a simple API wrapper.
- MCP tool architecture with a registry, approval mode, and HTTP/SSE server —
  a professional design pattern.
- VAD-gated audio capture (not just a fixed `time.sleep` recording).
- Persistent conversation memory and per-session pattern learning.
- Adaptive agent with behaviour learning, sequence detection, and prediction
  engine.
- Optional hand-tracking mouse controller with gesture-to-voice integration.
- Cross-platform support (Windows / macOS / Linux) in AppController and
  SystemExecutor.
- Undo stack and blocked-command safety list in SystemExecutor.
- Session deduplication guard for bare confirmations.

**Not yet advanced because:**
- No local/offline STT or wake-word engine; full internet dependency for core
  functionality.
- Wake word detection requires a complete cloud STT call — not an always-on
  low-power model.
- `click_element` by description is unimplemented (always returns a warning).
- Duplicate tool name registrations silently overwrite each other.
- `pyttsx3` initialised per call — a known stability hazard.
- No startup API key validation; silent failures on missing credentials.
- Structured logging not wired up (`utils/logger.py` unused).
- Core voice pipeline modules have no automated tests.
- Docker image is not functional for interactive voice use without undocumented
  host configuration.
- End-to-end latency (5–30 s) is too high for a responsive voice assistant
  experience.
