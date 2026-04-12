# JARVIS Voice Assistant

A professional AI voice assistant with an MCP tool architecture, hand-gesture
control, air-swipe typing, and an adaptive autonomous AI agent that learns your
patterns over time.

---

## Feature Overview

### 🖐️ Hand Tracking – Hologram Cursor (legacy, undisturbed)

Control your mouse entirely hands-free using a webcam and MediaPipe hand
tracking. The overlay renders a live "hologram" cursor that mirrors your index
finger.

| Gesture | Action |
|---|---|
| Point (index finger extended) | Move mouse cursor |
| Pinch (thumb + index) | Left click |
| Open palm | Right click |
| Fist | Pause / freeze pointer |
| Swipe right | Scroll right |
| Swipe left | Scroll left |
| Thumbs up | Re-activate voice listening |
| Peace sign | Take a screenshot |

Enable via `.env`:
```
HAND_TRACKING_ENABLED=true
```

---

### ✍️ Air Swipe Typing Keyboard

Type words in mid-air by swiping your finger over a QWERTY overlay rendered in
the lower portion of the camera frame.

**How it works:**
1. **Pinch** your thumb and index finger while your fingertip is over the
   keyboard overlay to enter swipe mode.
2. **Swipe** your fingertip across the letters of the word you want to type.
3. **Release** the pinch – JARVIS auto-corrects the raw swipe path to the
   closest English word using Levenshtein distance and types it into the
   focused window.

**Features:**
- Full QWERTY layout (26 letters + `SPACE`, `DEL`, `ENTER`)
- Live word suggestion displayed on the overlay as you swipe
- Hold any key for ≥ 1 s to repeat it (e.g. double-`L` in *hello*)
- `DEL` removes the last letter from the swipe path
- Bundled English word list (`data/english_words.txt`) with fallback
  built-in vocabulary
- Zero interference with the hand tracking mouse – swipe mode activates only
  when the finger is inside the keyboard zone

**Module:** `core/swipe_keyboard.py`  
**Tests:** `tests/test_swipe_keyboard.py`

---

### 🤖 JARVIS Adaptive Autonomous Agent

A multi-subsystem AI agent that sees your screen, controls apps, executes
system commands, and learns your habits over time.

#### Subsystems

| Module | Responsibility |
|---|---|
| `core/adaptive_agent.py` | Central orchestrator – routes commands to subsystems |
| `core/screen_vision.py` | On-demand / cached screenshot + OCR text extraction |
| `core/app_controller.py` | Launch, close, focus desktop applications |
| `core/system_executor.py` | Execute shell commands safely |
| `core/browser_automation.py` | Browser search via Selenium or keyboard shortcut |
| `core/pattern_memory.py` | Persist usage patterns to `data/user_patterns.json` |
| `core/behavior_learner.py` | Analyse patterns: frequency, sequences, time-of-day |
| `core/prediction_engine.py` | Predict next action and rank suggestions |
| `utils/app_finder.py` | Resolve friendly app names to executable paths |
| `utils/window_manager.py` | Cross-platform window focus |

#### Pattern Learning

JARVIS records every command, app launch/close, and search to build a personal
profile:

```
data/user_patterns.json
├── apps          – frequency, session duration, time-of-day usage
├── searches      – search term frequency
├── workflows     – multi-step action sequences with success rate
├── time_patterns – apps used at morning / afternoon / evening / night
├── context       – most recently active app + last commands
└── command_history – chronological log (capped at 500 entries)
```

The pattern database is **not** committed to the repository (`.gitignore`); it
lives locally on each user's machine.

#### Smart Browser Control

JARVIS uses screen context to avoid opening duplicate browser windows:

```
You: "open Brave and search for Python tutorials"
→ JARVIS checks: is Brave already running?
  YES → focuses existing window, types search in address bar
  NO  → launches Brave, waits for it, then performs the search
```

#### Predictions

After each command the prediction engine ranks likely next actions:

```
🔮 Prediction: open VSCode (72%) – frequently used at this time of day
```

#### Voice Commands (after wake word)

| Command | Effect |
|---|---|
| `patterns` / `show patterns` | Display learned app / search patterns |
| `predict` / `what next` | Show ranked next-action predictions |
| `memory` / `status` | Show conversation memory summary |
| `tools` / `capabilities` | List all MCP tools |
| `clear` / `forget` | Erase conversation memory |
| `exit` / `quit` / `bye` | Shut down JARVIS |

**Tests:** `tests/test_adaptive_agent.py`

---

## Architecture

```
JARVIS 2.0
├── 🗣️  Voice Pipeline
│   ├── core/audio_input.py       – sounddevice VAD capture
│   ├── core/speech_recognition.py – Whisper / Google STT
│   ├── core/wake_word.py          – wake-word detection
│   ├── core/llm_brain.py          – LLM + tool-calling loop
│   └── core/text_to_speech.py     – ElevenLabs / pyttsx3 TTS
│
├── 🔧 MCP Tool Server
│   ├── core/mcp_server.py         – HTTP tool server
│   ├── core/mcp_client.py         – client (call tools from LLM)
│   ├── core/tool_registry.py      – tool registration
│   └── tools/                     – tool implementations
│       ├── system_tools.py        – CPU/RAM/screenshot/agent tools
│       ├── web_apis.py            – web search, weather
│       ├── laptop_control.py      – volume, brightness, battery
│       ├── knowledge_base.py      – local knowledge store
│       └── home_automation.py     – smart-home integrations
│
├── 🖐️  Hand Tracking (optional)
│   ├── core/hand_voice_integration.py – background thread + gesture dispatch
│   ├── core/hand_tracking.py          – MediaPipe hand landmark processing
│   ├── core/gesture_recognition.py    – classify gestures
│   ├── core/hand_mouse_controller.py  – smoothed mouse control
│   ├── core/hand_ui_overlay.py        – OpenCV overlay rendering
│   └── core/swipe_keyboard.py         – air swipe typing + auto-correction
│
└── 🤖 Adaptive AI Agent
    ├── core/adaptive_agent.py      – orchestrator
    ├── core/screen_vision.py       – screenshot + OCR
    ├── core/app_controller.py      – launch / close / focus apps
    ├── core/system_executor.py     – shell commands
    ├── core/browser_automation.py  – browser search
    ├── core/pattern_memory.py      – persist patterns
    ├── core/behavior_learner.py    – analyse patterns
    └── core/prediction_engine.py   – predict next action
```

---

## Setup

### Requirements

- Python 3.9+
- Docker (optional, for containerised deployment)

### Installation

```bash
git clone https://github.com/Chiru-Boss/jarvis-voice-assistant.git
cd jarvis-voice-assistant
pip install -r requirements.txt
```

**Optional – OCR support (screen vision):**
```bash
# Install Tesseract binary first: https://github.com/tesseract-ocr/tesseract
pip install pytesseract
```

**Optional – hand tracking:**
```bash
pip install mediapipe opencv-python pyautogui
```

### Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```env
NVIDIA_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here   # optional
HAND_TRACKING_ENABLED=true         # enable webcam hand control
WAKE_WORD=jarvis
```

### Running

```bash
python main.py
```

---

## Testing

```bash
pip install pytest
pytest tests/ -v
```

All 126 tests cover:
- Swipe keyboard layout, geometry, lifecycle, auto-correction, hold-repeat
- PatternMemory persistence and retrieval
- BehaviorLearner frequency, sequence, and time-pattern analysis
- PredictionEngine ranked suggestions
- AdaptiveAgent command routing and pattern recording
- Hand-tracking module import guard (no regressions to hologram cursor)

---

## Next Enhancement Suggestion

### 🔔 Proactive Contextual Alerts

Now that JARVIS learns your patterns and can see the screen, the logical next
step is **proactive suggestions delivered before you ask**:

1. **Time-triggered reminders** – if every weekday at 09:00 you open VSCode,
   JARVIS greets you: *"Good morning! Ready to code? Opening VSCode for you."*
2. **Anomaly detection** – if you deviate from your usual workflow, JARVIS
   notices: *"You usually search Python docs at this time – want me to open
   that?"*
3. **App-state awareness** – using screen OCR, JARVIS detects when a build
   fails in the terminal and automatically suggests the fix or searches the
   error message.
4. **Reliability** – add a lightweight background scheduler
   (`core/proactive_scheduler.py`) that runs predictions on a configurable
   interval (e.g. every 5 minutes) and speaks a single non-intrusive hint when
   confidence exceeds a threshold (e.g. 80 %).

This builds directly on the already-implemented `PredictionEngine`,
`BehaviorLearner`, and `ScreenVision` components with no architectural
changes required.

---

## Performance Optimization

See [`performance_optimization.md`](performance_optimization.md) for tips on
reducing CPU usage, improving OCR latency, and tuning the hand tracking
confidence thresholds.

## Example Configurations

See [`example_configurations.md`](example_configurations.md) for sample `.env`
files for different use cases.
