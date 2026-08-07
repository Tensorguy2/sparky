"use strict";

// --- Tunables (same spirit as the chatbot VAD, tweak via window.VAD) ---
const VAD = (window.VAD = {
  onsetMs: 96,        // sustained speech before we consider the utterance started
  endpointMs: 340,    // silence that ends an utterance
  noiseFloorMult: 2.5,
  noiseFloorAlpha: 0.05,
  minRms: 0.006,
});

const SR = 16000;
const FRAME_MS = 32; // 512 samples @ 16 kHz

const els = {
  micBtn: document.getElementById("micBtn"),
  model: document.getElementById("model"),
  state: document.getElementById("state"),
  level: document.getElementById("levelBar"),
  transcripts: document.getElementById("transcripts"),
  status: document.getElementById("status"),
};

let ws = null;
let micCtx = null;
let micStream = null;
let listening = false;

let inSpeech = false;
let speechMs = 0;
let silenceMs = 0;
let noiseFloor = 0.005;
let preroll = []; // recent frames kept so onset ramp isn't clipped
let utteranceStart = 0;

function setState(text, cls) {
  els.state.textContent = text;
  els.state.className = "state " + (cls || "");
}

async function loadStatus() {
  const res = await fetch("/api/status");
  const st = await res.json();
  els.status.textContent =
    `parakeet: ${st.parakeet.onnx_model} on ${st.parakeet.device}` +
    (st.whisper.available ? ` | whisper: ${st.whisper.model} available` : "");
  if (!st.whisper.available) {
    [...els.model.options].forEach((o) => {
      if (o.value === "whisper") o.disabled = true;
    });
  }
}

function connectWS() {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "ready") resolve();
      else if (msg.type === "transcript") onTranscript(msg);
      else if (msg.type === "error") addRow(`error: ${msg.message}`, null);
    };
    ws.onerror = () => reject(new Error("WebSocket failed"));
    ws.onclose = () => { if (listening) stopMic(); };
  });
}

function floatToPCM16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out.buffer;
}

function onFrame(f32) {
  let sum = 0;
  for (let i = 0; i < f32.length; i++) sum += f32[i] * f32[i];
  const rms = Math.sqrt(sum / f32.length);
  els.level.style.width = Math.min(100, rms * 900) + "%";

  const threshold = Math.max(VAD.minRms, noiseFloor * VAD.noiseFloorMult);
  const voiced = rms > threshold;

  if (!inSpeech) {
    if (!voiced) noiseFloor += VAD.noiseFloorAlpha * (rms - noiseFloor);
    preroll.push(f32.slice(0));
    if (preroll.length > 8) preroll.shift();

    if (voiced) {
      speechMs += FRAME_MS;
      if (speechMs >= VAD.onsetMs) {
        inSpeech = true;
        silenceMs = 0;
        utteranceStart = performance.now();
        setState("speaking", "speaking");
        ws.send(JSON.stringify({ type: "start", sampleRate: SR, model: els.model.value }));
        for (const fr of preroll) ws.send(floatToPCM16(fr));
        preroll = [];
      }
    } else {
      speechMs = Math.max(0, speechMs - FRAME_MS);
    }
    return;
  }

  ws.send(floatToPCM16(f32));
  if (voiced) {
    silenceMs = 0;
  } else {
    silenceMs += FRAME_MS;
    if (silenceMs >= VAD.endpointMs) {
      inSpeech = false;
      speechMs = 0;
      silenceMs = 0;
      setState("transcribing…", "busy");
      ws.send(JSON.stringify({ type: "stop" }));
    }
  }
}

function onTranscript(msg) {
  const roundtrip = utteranceStart ? performance.now() - utteranceStart : 0;
  addRow(msg.text || "(empty)", msg, roundtrip);
  if (listening) setState("listening", "listening");
}

function addRow(text, msg, roundtripMs) {
  const row = document.createElement("div");
  row.className = "row";
  const t = document.createElement("div");
  t.className = "text";
  t.textContent = text;
  row.appendChild(t);
  if (msg && msg.stt_ms !== undefined) {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent =
      `${msg.model} · audio ${msg.audio_s}s · stt ${msg.stt_ms} ms` +
      (roundtripMs ? ` · end-to-end ${Math.round(roundtripMs)} ms` : "");
    row.appendChild(meta);
  }
  els.transcripts.prepend(row);
}

async function startMic() {
  await connectWS();
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  micCtx = new AudioContext({ sampleRate: SR });
  await micCtx.audioWorklet.addModule("js/pcm-worklet.js");
  const src = micCtx.createMediaStreamSource(micStream);
  const node = new AudioWorkletNode(micCtx, "pcm-worklet");
  node.port.onmessage = (ev) => onFrame(ev.data);
  src.connect(node);
  listening = true;
  els.micBtn.textContent = "Stop microphone";
  els.micBtn.classList.add("active");
  setState("listening", "listening");
}

function stopMic() {
  listening = false;
  inSpeech = false;
  speechMs = 0;
  silenceMs = 0;
  preroll = [];
  if (micCtx) { micCtx.close(); micCtx = null; }
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();
  ws = null;
  els.micBtn.textContent = "Start microphone";
  els.micBtn.classList.remove("active");
  els.level.style.width = "0%";
  setState("idle", "");
}

els.micBtn.addEventListener("click", () => {
  if (listening) stopMic();
  else startMic().catch((e) => { setState("error: " + e.message, "error"); stopMic(); });
});

loadStatus().catch(() => (els.status.textContent = "status unavailable"));
