# Whisperflowactions → JARVIS Merge Plan (Phase 0/1)

## Objective
Adopt Whisperflowactions strengths (modern HUD UX, streaming interaction model, and orchestrator-style state handling) **without replacing** current JARVIS voice, MCP tools, adaptive memory, or hand-tracking features.

## 1) Whisperflowactions analysis summary

### Front-end / UX paradigm
- Electron HUD (`ui/hud/main.js`) + Python overlay bridge (`ui/overlay.py`).
- State-driven UI (`IDLE/LISTENING/THINKING/EXECUTING/SUCCESS/ERROR/SPEAKING`) with live updates over local WebSocket.
- Visual/audio feedback loops (audio-energy streaming and step/status updates).

### Backend / orchestration model
- Central orchestrator (`core/orchestrator.py`) with thread-based non-blocking pipeline.
- Parallelized stages: context collection + STT.
- Streaming/iterative planning and route/execution loop.
- Built-in continuous wake handling + dictation mode.

### Tool architecture
- MCP server entrypoint (`mcp_server/server.py`) using FastMCP-style grouped tool registration.
- Emphasis on robust tool/resource/prompt registration.

## 2) Compatibility map with current JARVIS

### Already strong in JARVIS
- Mature modular voice pipeline and command loop (`main.py` + `core/*`).
- MCP tool architecture already present and integrated.
- Adaptive agent + pattern memory + hand gestures.

### Gap/opportunity to merge
- JARVIS currently has CLI-centric status flow; Whisperflowactions has richer UI state semantics.
- JARVIS has tool-calling loop but no dedicated UI event bridge abstraction for external HUDs.

## 3) Initial integration architecture (implemented in this patch)

### New merge point
- `core/ui_bridge.py` added as a **non-breaking bridge abstraction**:
  - Buffers state/stream events from the core voice loop.
  - Provides `transition()`, `stream_update()`, `snapshot()`, `drain_events()`.
  - Keeps all current JARVIS behavior unchanged when disabled.

### Wiring added
- `main.py` now emits UI lifecycle events around:
  - listening/transcribe/wake detection
  - LLM planning
  - response generation
  - speaking
  - completion/error transitions

### Config hooks
- `UI_BRIDGE_ENABLED` and `UI_BRIDGE_MAX_EVENTS` added to config/.env example.
- Default is disabled to preserve current runtime behavior.

## 4) Remaining roadmap (next patches)

- [ ] Add adapter module to push `UIBridge.drain_events()` to WebSocket/SSE.
- [ ] Implement Whisperflowactions-compatible HUD client contract (state payload schema).
- [ ] Add incremental token/tool-step streaming from `core/llm_brain.py`.
- [ ] Add optional continuous wake listener mode behind config flag.
- [ ] Add typed API endpoint(s) for UI state snapshot + event stream.
- [ ] Add integration tests for UI bridge transport layer.

## 5) Potential breaking points / risks

- Introducing always-on UI transports (WebSocket/Electron) can add optional runtime deps.
- Over-eager event emission may impact latency if not buffered/throttled.
- Streaming LLM/tool callbacks require careful threading around existing MCP/tool loop.

## 6) Why this phased approach

This keeps JARVIS stable while creating clear seam(s) for Whisperflowactions-style UI and streaming behavior. Current functionality is retained and future integration can be layered without refactoring the entire assistant core.
