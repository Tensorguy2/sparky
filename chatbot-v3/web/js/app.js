/**
 * Voice Chatbot — Browser client
 *
 * Handles:
 *  - WebSocket connection to the chatbot server
 *  - Microphone capture via AudioWorklet (PCM16 at 24 kHz)
 *  - LLM streaming display
 *  - TTS audio playback via Web Audio API
 *  - Session configuration (model, voice, instructions, context)
 */

import { AudioStreamPlayer } from "./audio-stream.js";

const $ = (id) => document.getElementById(id);

// -- DOM refs ------------------------------------------------------------------
const healthBadge     = $("health-badge");
const modelSelect     = $("model-select");
const sttSelect       = $("stt-select");
const voiceSelect     = $("voice-select");
const instructSelect  = $("instruction-select");
const contextSelect   = $("context-select");
const ttsToggle       = $("tts-toggle");
const handsFreeToggle = $("hands-free-toggle");
const btnClear        = $("btn-clear");
const sessionSelect   = $("session-select");
const btnLoadSession  = $("btn-load-session");
const btnSaveSession  = $("btn-save-session");
const btnSaveAs       = $("btn-save-as");
const btnRestartChatbot = $("btn-restart-chatbot");
const btnRestartTts = $("btn-restart-tts");
const sessionStatus   = $("session-status");
const chatMessages    = $("chat-messages");
const textInput       = $("text-input");
const btnSend         = $("btn-send");
const btnExcuseMe     = $("btn-excuse-me");
const btnMic          = $("btn-mic");
const sttStatus       = $("stt-status");
const llmStatus       = $("llm-status");
const contextModal    = $("context-modal");
const contextEditor   = $("context-editor");
const btnEditContext   = $("btn-edit-context");
const btnSaveContext   = $("btn-save-context");
const btnCancelContext = $("btn-cancel-context");
const btnCloseContext  = $("btn-close-context");
const instructionsModal = $("instructions-modal");
const instructionsModalTitle = $("instructions-modal-title");
const instructionIdInput = $("instruction-id-input");
const instructionNameInput = $("instruction-name-input");
const instructionPromptInput = $("instruction-prompt-input");
const instructionModelsInput = $("instruction-models-input");
const instructionEditorStatus = $("instruction-editor-status");
const btnEditInstructions = $("btn-edit-instructions");
const btnNewInstructions = $("btn-new-instructions");
const btnDeleteInstructions = $("btn-delete-instructions");
const btnSaveInstructions = $("btn-save-instructions");
const btnCancelInstructions = $("btn-cancel-instructions");
const btnCloseInstructions = $("btn-close-instructions");

let instructionsEditorMode = "edit"; // "edit" | "new"

// -- State ---------------------------------------------------------------------
/** @type {WebSocket | null} */
let ws = null;
let connected = false;
let listenMode = false;
let inUtterance = false;
let sttFinishing = false;
let sttPartialText = "";
let sttPartialBubble = null;
let conversationBusy = false;
let sessionId = localStorage.getItem("chatbot_session_id") || "last";
let sessionLoaded = false;

/** @type {AudioStreamPlayer | null} */
let ttsPlayer = null;

/** @type {MediaStream | null} */
let micStream = null;
/** @type {AudioContext | null} */
let micCtx = null;
/** @type {ScriptProcessorNode | null} */
let micProcessor = null;

// Voice activity detection (energy-based, tuned for 24 kHz / 4096-sample frames)
const VAD = {
  speechThreshold: 0.012,
  minSpeechFrames: 2,
  // 2 frames × 4096 @ 24 kHz ≈ 341 ms end-of-speech wait (was 4 ≈ 683 ms).
  silenceFrames: 2,
  // Echo guard: while the assistant is speaking (conversationBusy), require a
  // stronger and longer signal to barge in, so speaker bleed of the bot's own
  // voice doesn't self-trigger an interrupt loop.
  bargeInThresholdMult: 1.5,
  bargeInMinSpeechFrames: 4,
};
let vadSpeechFrames = 0;
let vadSilenceFrames = 0;

let currentAssistantEl = null;
let currentAssistantText = "";

// -- WebSocket -----------------------------------------------------------------

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/chat`;
}

function connectWS() {
  if (ws) return;
  ws = new WebSocket(wsUrl());
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    connected = true;
    // After chatbot restart / reconnect, re-arm hands-free mic if the toggle is on.
    if (handsFreeToggle?.checked && !listenMode) {
      enableListenMode().catch((err) => {
        console.warn("Failed to re-enable mic after reconnect:", err);
      });
    }
    setHealth(true);
    sessionLoaded = false;
    sendJSON({ type: "load_session", session_id: sessionId });
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      handleServerEvent(JSON.parse(ev.data));
    }
  };

  ws.onerror = () => setHealth(false);

  ws.onclose = () => {
    connected = false;
    disableListenMode();
    ws = null;
    setHealth(false);
    setTimeout(connectWS, 3000);
  };
}

function sendJSON(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function sendConfig() {
  sendJSON({
    type: "config",
    model: modelSelect.value,
    stt_model: sttSelect?.value,
    voice_id: voiceSelect.value,
    instruction_set: instructSelect.value,
    context: contextSelect.value,
    tts_enabled: ttsToggle.checked,
    language: "English",
  });
}

// -- Server events -------------------------------------------------------------

function handleServerEvent(evt) {
  switch (evt.type) {
    case "config_ack":
      break;

    case "session_loaded":
      applyLoadedSession(evt);
      break;

    case "session_saved":
      sessionId = evt.session_id || sessionId;
      localStorage.setItem("chatbot_session_id", sessionId);
      updateSessionStatus(evt.turn_count, sessionId);
      refreshSessionList().then(() => {
        if (sessionSelect) sessionSelect.value = sessionId;
      });
      break;

    case "stt_ready":
      break;

    case "speech_started":
      onBargeIn();
      sttPartialText = "";
      sttStatus.textContent = "Hearing speech…";
      break;

    case "turn_cancelled":
      onTurnCancelled();
      break;

    case "speech_stopped":
      sttStatus.textContent = "Processing…";
      break;

    case "vad_speech_stopped":
      if (inUtterance && !sttFinishing) {
        sttFinishing = true;
        inUtterance = false;
        sttStatus.textContent = "Processing…";
        sendJSON({ type: "stop_stt" });
        btnMic.classList.remove("recording");
      }
      break;

    case "transcript_delta":
      if (evt.delta === "Transcribing…") {
        sttStatus.textContent = evt.delta;
      } else {
        sttPartialText += evt.delta;
        if (!sttPartialBubble) {
          sttPartialBubble = appendMessage("user", sttPartialText, true);
        } else {
          const textSpan = sttPartialBubble.querySelector(".message-text");
          if (textSpan) textSpan.textContent = sttPartialText;
          scrollToBottom();
        }
        sttStatus.textContent = "Listening…";
      }
      break;

    case "transcript_done":
      if (sttPartialBubble) {
        sttPartialBubble.classList.remove("message-streaming");
        const textSpan = sttPartialBubble.querySelector(".message-text");
        if (textSpan) textSpan.textContent = evt.text;
        sttPartialBubble = null;
      } else {
        appendMessage("user", evt.text);
      }
      sttPartialText = "";
      sttStatus.textContent = "";
      break;

    case "stt_stopped":
      inUtterance = false;
      sttFinishing = false;
      sttPartialText = "";
      if (sttPartialBubble) {
        const textSpan = sttPartialBubble.querySelector(".message-text");
        if (textSpan && !textSpan.textContent.trim()) {
          sttPartialBubble.remove();
        } else if (sttPartialBubble) {
          sttPartialBubble.classList.remove("message-streaming");
        }
        sttPartialBubble = null;
      }
      btnMic.classList.remove("recording");
      updateListenStatus();
      break;

    case "llm_start":
      conversationBusy = true;
      updateExcuseMeButton();
      llmStatus.textContent = `Generating (${evt.model})…`;
      currentAssistantText = "";
      currentAssistantEl = appendMessage("assistant", "", true);
      stopTtsPlayback();
      break;

    case "excuse_me_start":
      onExcuseMeStart(evt);
      break;

    case "tts_instability":
      onTtsInstability(evt);
      break;

    case "tts_start":
      ensureTtsPlayer().then((p) => {
        if (!evt.is_filler) {
          p.startSmooth(evt.sample_rate || 24000, evt.num_sentences || 1);
        }
      });
      break;

    case "llm_delta":
      currentAssistantText += evt.delta;
      if (currentAssistantEl) {
        currentAssistantEl.querySelector(".message-text").textContent = currentAssistantText;
        scrollToBottom();
      }
      break;

    case "llm_done":
      llmStatus.textContent = "";
      if (currentAssistantEl) {
        currentAssistantEl.classList.remove("message-streaming");
        const meta = currentAssistantEl.querySelector(".message-meta");
        if (meta) meta.textContent = modelSelect.value;
      }
      currentAssistantEl = null;
      break;

    case "tts_audio":
      playTtsChunk(evt);
      break;

    case "tts_sentence_done":
      break;

    case "tts_done":
      if (ttsPlayer) ttsPlayer.flush();
      break;

    case "turn_done":
      conversationBusy = false;
      updateExcuseMeButton();
      updateListenStatus();
      break;

    case "history_cleared":
      chatMessages.innerHTML = '<div class="chat-empty">New conversation — say something or type below.</div>';
      updateSessionStatus(0);
      break;

    case "error":
      llmStatus.textContent = "";
      sttStatus.textContent = "";
      appendSystem(`Error: ${evt.message}`);
      break;
  }
}

// -- Chat UI -------------------------------------------------------------------

async function pollChatbotHealth(maxMs = 90_000) {
  const start = Date.now();
  const tick = async () => {
    try {
      const r = await fetch("/api/health");
      if (r.ok) {
        setHealth(true);
        if (sessionStatus) sessionStatus.textContent = "Reconnected";
        appendSystem("Chatbot server is back online.");
        if (!ws || ws.readyState !== WebSocket.OPEN) {
          connectWS();
        }
        return;
      }
    } catch { /* server still down */ }
    if (Date.now() - start < maxMs) {
      setTimeout(tick, 2000);
    } else {
      appendSystem("Chatbot still starting — refresh the page or wait a bit longer.");
    }
  };
  setTimeout(tick, 3000);
}

async function restartService(endpoint, label) {
  if (!confirm(`Restart ${label}? Active generation will stop.`)) {
    return;
  }
  const isTts = endpoint.includes("restart-tts");
  const isChatbot = !isTts;
  try {
    const r = await fetch(endpoint, { method: "POST" });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      appendSystem(`Restart failed: ${data.error || r.statusText}`);
      return;
    }
    appendSystem(data.message || `Restarting ${label}…`);
    if (isTts) {
      pollTtsHealth(90_000);
    } else if (isChatbot) {
      setHealth(false);
      if (sessionStatus) sessionStatus.textContent = "Restarting…";
      pollChatbotHealth(90_000);
    }
  } catch (err) {
    // Chatbot restart kills the connection; treat as success and poll for recovery.
    if (isChatbot) {
      appendSystem("Chatbot is restarting… reconnecting shortly.");
      setHealth(false);
      if (sessionStatus) sessionStatus.textContent = "Restarting…";
      pollChatbotHealth(90_000);
      return;
    }
    appendSystem(`Restart failed: ${err.message}`);
  }
}

async function pollTtsHealth(maxMs = 60_000) {
  const start = Date.now();
  const tick = async () => {
    try {
      const r = await fetch("/api/tts-health");
      const d = await r.json();
      if (d.online) {
        appendSystem("TTS server is back online.");
        await loadVoices();
        return;
      }
    } catch { /* ignore */ }
    if (Date.now() - start < maxMs) {
      setTimeout(tick, 3000);
    } else {
      appendSystem("TTS server still starting — try again in a moment.");
    }
  };
  setTimeout(tick, 5000);
}

async function refreshSessionList() {
  if (!sessionSelect) return;
  const current = sessionId;
  sessionSelect.innerHTML = "";
  const seen = new Set();
  const add = (id, label) => {
    if (!id || seen.has(id)) return;
    seen.add(id);
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = label;
    sessionSelect.appendChild(opt);
  };
  try {
    const r = await fetch("/api/sessions");
    const list = await r.json();
    add("last", "Last conversation");
    for (const s of list) {
      const n = s.turn_count || 0;
      const title = s.title ? `${s.title.slice(0, 36)}` : s.session_id;
      add(s.session_id, `${title} (${n})`);
    }
  } catch {
    add("last", "Last conversation");
  }
  if ([...sessionSelect.options].some((o) => o.value === current)) {
    sessionSelect.value = current;
  }
}

function requestLoadSession(id) {
  sessionId = id || sessionSelect?.value || "last";
  localStorage.setItem("chatbot_session_id", sessionId);
  if (sessionSelect) sessionSelect.value = sessionId;
  conversationBusy = false;
  currentAssistantEl = null;
  sendJSON({ type: "load_session", session_id: sessionId });
}

function updateSessionStatus(turnCount, sid = sessionId) {
  if (!sessionStatus) return;
  if (turnCount > 0) {
    sessionStatus.textContent = `Saved · ${turnCount} message(s)`;
    sessionStatus.title = `Session: ${sid}`;
  } else {
    sessionStatus.textContent = "New conversation";
    sessionStatus.title = "";
  }
}

function applyLoadedSession(evt) {
  sessionLoaded = true;
  sessionId = evt.session_id || "last";
  localStorage.setItem("chatbot_session_id", sessionId);
  if (sessionSelect) sessionSelect.value = sessionId;
  refreshSessionList();

  if (evt.model && [...modelSelect.options].some((o) => o.value === evt.model)) {
    modelSelect.value = evt.model;
  }
  if (evt.voice_id && [...voiceSelect.options].some((o) => o.value === evt.voice_id)) {
    voiceSelect.value = evt.voice_id;
  }
  if (evt.instruction_set) instructSelect.value = evt.instruction_set;
  if (evt.context) contextSelect.value = evt.context;
  if (typeof evt.tts_enabled === "boolean") ttsToggle.checked = evt.tts_enabled;
  if (evt.stt_model && sttSelect && [...sttSelect.options].some((o) => o.value === evt.stt_model)) {
    sttSelect.value = evt.stt_model;
  }

  renderHistory(evt.turns || []);
  updateSessionStatus(evt.turn_count || 0, sessionId);

  if (evt.resumed && evt.turn_count > 0) {
    const label = evt.title ? `"${evt.title.slice(0, 40)}…"` : `${evt.turn_count} messages`;
    appendSystem(`Continued your last conversation (${label}).`);
  }

  sendConfig();
  if (handsFreeToggle.checked) {
    enableListenMode();
  }
}

function renderHistory(turns) {
  chatMessages.innerHTML = "";
  if (!turns.length) {
    chatMessages.innerHTML = '<div class="chat-empty">Start a conversation by typing or using voice input.</div>';
    return;
  }
  for (const t of turns) {
    const el = appendMessage(t.role, t.content || "");
    if (t.role === "assistant") {
      const meta = el.querySelector(".message-meta");
      if (meta) meta.textContent = t.model || modelSelect.value;
    }
  }
}

function appendMessage(role, text, streaming = false) {
  removeEmpty();
  const div = document.createElement("div");
  div.className = `message message-${role}${streaming ? " message-streaming" : ""}`;

  const textSpan = document.createElement("span");
  textSpan.className = "message-text";
  textSpan.textContent = text;
  div.appendChild(textSpan);

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = role === "user" ? "You" : "";
  div.appendChild(meta);

  chatMessages.appendChild(div);
  scrollToBottom();
  return div;
}

function appendSystem(text) {
  removeEmpty();
  const div = document.createElement("div");
  div.className = "message message-assistant";
  div.style.color = "var(--danger)";
  div.textContent = text;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function removeEmpty() {
  const empty = chatMessages.querySelector(".chat-empty");
  if (empty) empty.remove();
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// -- Audio playback (TTS) ------------------------------------------------------

async function ensureTtsPlayer() {
  if (!ttsPlayer) {
    ttsPlayer = new AudioStreamPlayer();
  }
  await ttsPlayer.init(24000);
  return ttsPlayer;
}

function stopTtsPlayback() {
  if (ttsPlayer) {
    ttsPlayer.stopPlayback();
  }
}

function playTtsChunk(evt) {
  if (!ttsPlayer?.available) return;
  const raw = Uint8Array.from(atob(evt.data), (c) => c.charCodeAt(0));
  const pcm = new Float32Array(raw.buffer);
  if (evt.sample_rate) {
    ttsPlayer.correctSampleRate(evt.sample_rate);
  }
  if (evt.is_filler) {
    ttsPlayer.flush();
    ttsPlayer.write(pcm);
    ttsPlayer.flush();
  } else {
    ttsPlayer.write(pcm);
  }
}

// -- Microphone + hands-free VAD -----------------------------------------------

function pcmRms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    sum += samples[i] * samples[i];
  }
  return Math.sqrt(sum / samples.length);
}

function float32ToPcm16Buffer(float32) {
  const pcm16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm16.buffer;
}

function updateExcuseMeButton(highlight = false) {
  if (!btnExcuseMe) return;
  btnExcuseMe.disabled = !conversationBusy;
  btnExcuseMe.classList.toggle("highlight", highlight && conversationBusy);
}

function sendExcuseMe() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  stopTtsPlayback();
  btnExcuseMe.classList.remove("highlight");
  llmStatus.textContent = "Recovering…";
  sendJSON({ type: "excuse_me" });
}

function onTtsInstability(evt) {
  // Drop any already-queued PCM so a blast/partial sentence cannot keep playing.
  if (ttsPlayer) ttsPlayer.stopPlayback();
  const retrying = evt.will_retry ? " · server retrying" : " · muted";
  llmStatus.textContent = `Audio instability (${evt.reason})${retrying}`;
  updateExcuseMeButton(true);
}

function onExcuseMeStart(evt) {
  conversationBusy = true;
  updateExcuseMeButton();
  stopTtsPlayback();
  llmStatus.textContent = "Excuse me — replaying…";
  if (evt.text) {
    currentAssistantText = evt.text;
    if (!currentAssistantEl) {
      currentAssistantEl = appendMessage("assistant", "", true);
    }
    const textEl = currentAssistantEl.querySelector(".message-text");
    if (textEl) textEl.textContent = currentAssistantText;
    currentAssistantEl.classList.add("message-streaming");
  }
}

function onTurnCancelled() {
  stopTtsPlayback();
  conversationBusy = false;
  updateExcuseMeButton();
  llmStatus.textContent = "";
  if (currentAssistantEl) {
    currentAssistantEl.classList.remove("message-streaming");
    const textEl = currentAssistantEl.querySelector(".message-text");
    if (textEl && !textEl.textContent.trim()) {
      currentAssistantEl.remove();
    } else if (textEl) {
      textEl.textContent += " …";
    }
    const meta = currentAssistantEl.querySelector(".message-meta");
    if (meta) meta.textContent = "interrupted";
  }
  currentAssistantEl = null;
  updateListenStatus();
}

function onBargeIn() {
  stopTtsPlayback();
  if (conversationBusy) {
    conversationBusy = false;
    llmStatus.textContent = "";
  }
}

function updateListenStatus() {
  if (!listenMode) {
    sttStatus.textContent = "";
    return;
  }
  if (inUtterance) {
    sttStatus.textContent = conversationBusy ? "Interrupting…" : "Hearing you…";
  } else if (conversationBusy) {
    sttStatus.textContent = "Speaking — interrupt anytime";
  } else {
    sttStatus.textContent = "Listening — just speak";
  }
}

function beginUtterance() {
  if (inUtterance || sttFinishing || !listenMode) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (conversationBusy) {
    stopTtsPlayback();
    // Explicit interrupt: with non-blocking turns the server processes this
    // immediately, cancelling LLM/TTS before the start_stt below.
    sendJSON({ type: "interrupt" });
  }
  inUtterance = true;
  sttFinishing = false;
  vadSpeechFrames = 0;
  vadSilenceFrames = 0;
  sendJSON({ type: "start_stt" });
  btnMic.classList.add("recording");
  updateListenStatus();
}

function endUtterance() {
  if (!inUtterance || sttFinishing) return;
  sttFinishing = true;
  inUtterance = false;
  sttStatus.textContent = "Transcribing…";
  sendJSON({ type: "stop_stt" });
}

function processVad(float32) {
  if (!listenMode) return;

  // Echo guard: barge-in (speaking while the assistant talks) needs a louder,
  // longer signal than normal turn-taking.
  const bargeIn = conversationBusy;
  const threshold = bargeIn
    ? VAD.speechThreshold * VAD.bargeInThresholdMult
    : VAD.speechThreshold;
  const minFrames = bargeIn ? VAD.bargeInMinSpeechFrames : VAD.minSpeechFrames;
  const loud = pcmRms(float32) >= threshold;

  if (!inUtterance) {
    if (loud) {
      vadSpeechFrames += 1;
      if (vadSpeechFrames >= minFrames) {
        beginUtterance();
      }
    } else {
      vadSpeechFrames = 0;
    }
    return;
  }

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(float32ToPcm16Buffer(float32));
  }

  if (loud) {
    vadSilenceFrames = 0;
  } else {
    vadSilenceFrames += 1;
    if (vadSilenceFrames >= VAD.silenceFrames) {
      endUtterance();
      vadSpeechFrames = 0;
      vadSilenceFrames = 0;
    }
  }
}

async function enableListenMode() {
  if (listenMode) return;

  // TTS player init must not block mic capture (and can leave AudioContext suspended).
  try {
    await ensureTtsPlayer();
  } catch (err) {
    console.warn("TTS player init failed; mic will still try to start.", err);
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    appendSystem("Microphone unavailable (needs a secure context or browser permission).");
    handsFreeToggle.checked = false;
    return;
  }

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 24000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (err) {
    appendSystem(`Microphone error: ${err.message}`);
    handsFreeToggle.checked = false;
    return;
  }

  micCtx = new AudioContext({ sampleRate: 24000 });
  // Browsers often start AudioContext suspended until a user gesture / resume().
  if (micCtx.state === "suspended") {
    try {
      await micCtx.resume();
    } catch (err) {
      console.warn("AudioContext resume failed:", err);
    }
  }

  const source = micCtx.createMediaStreamSource(micStream);
  micProcessor = micCtx.createScriptProcessor(4096, 1, 1);
  micProcessor.onaudioprocess = (e) => {
    if (!listenMode) return;
    processVad(e.inputBuffer.getChannelData(0));
  };
  source.connect(micProcessor);
  // Do NOT connect the mic graph to destination. That plays the mic into the
  // speakers and echoCancellation then nulls the capture — mic looks "dead".
  const silent = micCtx.createGain();
  silent.gain.value = 0;
  micProcessor.connect(silent);
  silent.connect(micCtx.destination);

  listenMode = true;
  inUtterance = false;
  vadSpeechFrames = 0;
  vadSilenceFrames = 0;
  btnMic.classList.add("listening");
  btnMic.classList.remove("recording");
  updateListenStatus();
  sttStatus.textContent = micCtx.state === "running" ? "Mic on — listening" : "Mic on — tap page if silent";
}

function disableListenMode() {
  if (inUtterance) {
    sendJSON({ type: "stop_stt" });
  }
  listenMode = false;
  inUtterance = false;
  vadSpeechFrames = 0;
  vadSilenceFrames = 0;
  btnMic.classList.remove("listening", "recording");

  if (micProcessor) {
    micProcessor.disconnect();
    micProcessor.onaudioprocess = null;
    micProcessor = null;
  }
  if (micCtx) {
    micCtx.close().catch(() => {});
    micCtx = null;
  }
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
  }
  sttStatus.textContent = "";
}

function toggleListenMode() {
  if (listenMode) {
    disableListenMode();
    handsFreeToggle.checked = false;
  } else {
    handsFreeToggle.checked = true;
    enableListenMode();
  }
}

// -- Sending text --------------------------------------------------------------

function sendText() {
  ensureTtsPlayer();
  const text = textInput.value.trim();
  if (!text) return;
  appendMessage("user", text);
  sendJSON({ type: "text_input", text });
  textInput.value = "";
  autoResize();
}

// -- Health check --------------------------------------------------------------

function setHealth(ok) {
  healthBadge.className = `badge ${ok ? "badge-ok" : "badge-bad"}`;
  healthBadge.textContent = ok ? "Connected" : "Disconnected";
}

// -- Load sidebar data ---------------------------------------------------------

async function loadSttModels() {
  if (!sttSelect) return;
  try {
    const r = await fetch("/api/stt-models");
    const data = await r.json();
    sttSelect.innerHTML = "";
    const groups = {};
    for (const m of data.models || []) {
      const provider = m.provider || "other";
      if (!groups[provider]) groups[provider] = [];
      groups[provider].push(m);
    }
    for (const [provider, models] of Object.entries(groups)) {
      const group = document.createElement("optgroup");
      group.label = provider.charAt(0).toUpperCase() + provider.slice(1);
      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.label || m.id;
        group.appendChild(opt);
      }
      sttSelect.appendChild(group);
    }
    const defaultStt = data.default_model || "gpt-realtime-whisper";
    if (sttSelect.querySelector(`option[value="${defaultStt}"]`)) {
      sttSelect.value = defaultStt;
    }
  } catch {}
}

async function loadModels() {
  try {
    const r = await fetch("/api/models");
    const data = await r.json();
    modelSelect.innerHTML = "";
    for (const [provider, models] of Object.entries(data)) {
      const group = document.createElement("optgroup");
      group.label = provider.charAt(0).toUpperCase() + provider.slice(1);
      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        group.appendChild(opt);
      }
      modelSelect.appendChild(group);
    }
    // Try to select the default
    const defaultModel = healthBadge.dataset.defaultModel || "gpt-4o";
    if (modelSelect.querySelector(`option[value="${defaultModel}"]`)) {
      modelSelect.value = defaultModel;
    }
  } catch {}
}

async function loadVoices() {
  try {
    const r = await fetch("/api/voices");
    const voices = await r.json();
    if (!Array.isArray(voices) || !voices.length) return;
    voiceSelect.innerHTML = "";
    for (const v of voices) {
      const opt = document.createElement("option");
      opt.value = v.voice_id;
      opt.textContent = `${v.name} (${v.voice_id})`;
      voiceSelect.appendChild(opt);
    }
  } catch {}
}

async function loadInstructions(selectId) {
  try {
    const r = await fetch("/api/instructions");
    const sets = await r.json();
    if (!Array.isArray(sets) || !sets.length) return;
    const prev = selectId || instructSelect.value;
    instructSelect.innerHTML = "";
    for (const s of sets) {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = s.name;
      instructSelect.appendChild(opt);
    }
    if (prev && instructSelect.querySelector(`option[value="${prev}"]`)) {
      instructSelect.value = prev;
    }
    updateDeleteInstructionsButton();
  } catch {}
}

function updateDeleteInstructionsButton() {
  const id = instructSelect.value;
  btnDeleteInstructions.disabled = !id || id === "default";
}

function defaultModelsTemplate() {
  return {
    "gpt-4.1-mini": { temperature: 0.7, max_tokens: 2048 },
    "gpt-4o": { temperature: 0.7, max_tokens: 2048 },
    "claude-sonnet-4-6": { temperature: 0.7, max_tokens: 2048 },
  };
}

function setInstructionEditorStatus(msg, isError = false) {
  instructionEditorStatus.textContent = msg || "";
  instructionEditorStatus.classList.toggle("error", !!isError);
}

function closeInstructionsEditor() {
  instructionsModal.hidden = true;
  setInstructionEditorStatus("");
}

async function openInstructionsEditor(mode = "edit") {
  instructionsEditorMode = mode;
  setInstructionEditorStatus("");

  if (mode === "new") {
    instructionsModalTitle.textContent = "New instruction set";
    instructionIdInput.disabled = false;
    instructionIdInput.value = "";
    instructionNameInput.value = "";
    instructionPromptInput.value =
      "You are a helpful assistant. Keep responses conversational since they will be spoken aloud.";
    instructionModelsInput.value = JSON.stringify(defaultModelsTemplate(), null, 2);
    instructionsModal.hidden = false;
    return;
  }

  const id = instructSelect.value;
  if (!id) return;
  instructionsModalTitle.textContent = `Edit: ${id}`;
  instructionIdInput.disabled = true;
  instructionIdInput.value = id;

  try {
    const r = await fetch(`/api/instructions/${encodeURIComponent(id)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    instructionNameInput.value = data.name || "";
    instructionPromptInput.value = data.system_prompt || "";
    instructionModelsInput.value = JSON.stringify(data.models || {}, null, 2);
  } catch (e) {
    setInstructionEditorStatus(`Failed to load: ${e.message}`, true);
  }
  instructionsModal.hidden = false;
}

function buildInstructionPayload() {
  const id = instructionIdInput.value.trim().replace(/[^a-zA-Z0-9_-]/g, "");
  const name = instructionNameInput.value.trim();
  const system_prompt = instructionPromptInput.value.trim();
  if (!id) throw new Error("ID is required.");
  if (!name) throw new Error("Display name is required.");
  if (!system_prompt) throw new Error("System prompt is required.");

  let models;
  try {
    models = JSON.parse(instructionModelsInput.value || "{}");
  } catch {
    throw new Error("Per-model settings must be valid JSON.");
  }
  if (typeof models !== "object" || models === null || Array.isArray(models)) {
    throw new Error("Per-model settings must be a JSON object.");
  }

  return { id, name, system_prompt, models };
}

async function saveInstructions() {
  setInstructionEditorStatus("Saving…");
  let payload;
  try {
    payload = buildInstructionPayload();
  } catch (e) {
    setInstructionEditorStatus(e.message, true);
    return;
  }

  const isNew = instructionsEditorMode === "new";
  const url = isNew
    ? "/api/instructions"
    : `/api/instructions/${encodeURIComponent(payload.id)}`;
  const method = isNew ? "POST" : "PUT";

  try {
    const r = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      setInstructionEditorStatus(data.error || `Save failed (${r.status})`, true);
      return;
    }
    await loadInstructions(payload.id);
    instructSelect.value = payload.id;
    sendConfig();
    closeInstructionsEditor();
  } catch (e) {
    setInstructionEditorStatus(`Error: ${e.message}`, true);
  }
}

async function deleteInstructions() {
  const id = instructSelect.value;
  if (!id || id === "default") return;
  if (!confirm(`Delete instruction set "${id}"? This cannot be undone.`)) return;

  try {
    const r = await fetch(`/api/instructions/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      appendSystem(data.error || `Delete failed (${r.status})`);
      return;
    }
    await loadInstructions("default");
    instructSelect.value = "default";
    sendConfig();
  } catch (e) {
    appendSystem(`Delete error: ${e.message}`);
  }
}

async function loadContextFiles() {
  try {
    const r = await fetch("/api/context");
    const names = await r.json();
    if (!Array.isArray(names) || !names.length) return;
    contextSelect.innerHTML = "";
    for (const name of names) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      contextSelect.appendChild(opt);
    }
  } catch {}
}

// -- Context editor ------------------------------------------------------------

async function openContextEditor() {
  const name = contextSelect.value;
  try {
    const r = await fetch(`/api/context/${encodeURIComponent(name)}`);
    const data = await r.json();
    contextEditor.value = data.content || "";
  } catch {
    contextEditor.value = "";
  }
  contextModal.hidden = false;
}

async function saveContext() {
  const name = contextSelect.value;
  await fetch(`/api/context/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: contextEditor.value }),
  });
  contextModal.hidden = true;
}

// -- Auto-resize textarea ------------------------------------------------------

function autoResize() {
  textInput.style.height = "auto";
  textInput.style.height = Math.min(textInput.scrollHeight, 120) + "px";
}

// -- Event listeners -----------------------------------------------------------

btnSend.addEventListener("click", sendText);

if (btnExcuseMe) {
  btnExcuseMe.addEventListener("click", sendExcuseMe);
}

textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendText();
  }
});

textInput.addEventListener("input", autoResize);

btnMic.addEventListener("click", toggleListenMode);

handsFreeToggle.addEventListener("change", () => {
  if (handsFreeToggle.checked) {
    if (connected) enableListenMode();
  } else {
    disableListenMode();
  }
});

btnClear.addEventListener("click", () => {
  if (!confirm("Start a new conversation? The current one stays saved until you replace it.")) {
    return;
  }
  sendJSON({ type: "clear_history" });
});

if (btnRestartChatbot) {
  btnRestartChatbot.addEventListener("click", () => {
    restartService("/api/admin/restart", "chatbot");
  });
}

if (btnRestartTts) {
  btnRestartTts.addEventListener("click", () => {
    restartService("/api/admin/restart-tts", "TTS voice server");
  });
}

if (btnLoadSession) {
  btnLoadSession.addEventListener("click", () => {
    requestLoadSession(sessionSelect?.value || "last");
  });
}

if (btnSaveSession) {
  btnSaveSession.addEventListener("click", () => {
    sendJSON({ type: "save_session" });
    btnSaveSession.textContent = "Saving…";
    setTimeout(() => { btnSaveSession.textContent = "Save"; }, 800);
  });
}

if (btnSaveAs) {
  btnSaveAs.addEventListener("click", () => {
    const name = prompt("Session name:", sessionId === "last" ? "" : sessionId);
    if (name === null) return;
    const id = name.trim().replace(/[^a-zA-Z0-9_-]/g, "_") || "last";
    sendJSON({ type: "save_session", session_id: id });
    sessionId = id;
    localStorage.setItem("chatbot_session_id", id);
    if (sessionSelect) sessionSelect.value = id;
    btnSaveAs.textContent = "Saving…";
    setTimeout(() => { btnSaveAs.textContent = "Save as…"; }, 800);
  });
}

// Config changes -> send to server
modelSelect.addEventListener("change", sendConfig);
if (sttSelect) sttSelect.addEventListener("change", sendConfig);
voiceSelect.addEventListener("change", sendConfig);
instructSelect.addEventListener("change", () => {
  updateDeleteInstructionsButton();
  sendConfig();
});
contextSelect.addEventListener("change", sendConfig);
ttsToggle.addEventListener("change", sendConfig);

// Instructions editor
btnEditInstructions.addEventListener("click", () => openInstructionsEditor("edit"));
btnNewInstructions.addEventListener("click", () => openInstructionsEditor("new"));
btnDeleteInstructions.addEventListener("click", deleteInstructions);
btnSaveInstructions.addEventListener("click", saveInstructions);
btnCancelInstructions.addEventListener("click", closeInstructionsEditor);
btnCloseInstructions.addEventListener("click", closeInstructionsEditor);
instructionsModal.addEventListener("click", (e) => {
  if (e.target === instructionsModal) closeInstructionsEditor();
});

// Context editor
btnEditContext.addEventListener("click", openContextEditor);
btnSaveContext.addEventListener("click", saveContext);
btnCancelContext.addEventListener("click", () => { contextModal.hidden = true; });
btnCloseContext.addEventListener("click", () => { contextModal.hidden = true; });
contextModal.addEventListener("click", (e) => {
  if (e.target === contextModal) contextModal.hidden = true;
});

// -- Init ----------------------------------------------------------------------

async function init() {
  await Promise.all([
    loadModels(),
    loadSttModels(),
    loadVoices(),
    loadInstructions(),
    loadContextFiles(),
    refreshSessionList(),
  ]);
  connectWS();
}

init();
