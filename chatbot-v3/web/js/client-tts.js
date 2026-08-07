/**
 * Client-side TTS streaming — bypasses the chatbot server's sentence buffer
 * by calling the TTS server's WebSocket directly from the browser.
 *
 * Usage:
 *   import { ClientTTS } from "./client-tts.js";
 *   const ctts = new ClientTTS({ ttsPort: 25568, audioPlayer: ttsPlayer });
 *   // During LLM streaming:
 *   ctts.feed(token);
 *   // When LLM is done:
 *   ctts.flush();
 *   // On barge-in:
 *   ctts.cancel();
 *
 * Kill switch: set `window.__clientTtsEnabled = false` in the browser console
 * to disable at runtime without restarting.
 */

const CLAUSE_RE = /(?<=[,;:\u2014\u2013\-])\s+|(?<=[.!?…])\s+/;

// App-level control markers (e.g. [[GOODBYE]]) must never reach the voice.
const CONTROL_TOKEN_RE = /\[\[[^\]]*\]\]/g;

export class ClientTTS {
  constructor({ ttsPort = 25568, getAudioPlayer, getVoiceId, getLanguage }) {
    this._ttsPort = ttsPort;
    this._getAudioPlayer = getAudioPlayer;
    this._getVoiceId = getVoiceId;
    this._getLanguage = getLanguage || (() => "English");

    this._buffer = "";
    this._firstDispatched = false;
    this._queue = [];         // Array of text fragments to synthesize
    this._activeWs = null;    // Current TTS WebSocket
    this._consuming = false;
    this._cancelled = false;
    this._sentenceCount = 0;
    this._turnId = null;

    // Callbacks
    this.onTtsStart = null;   // () => void — first audio dispatched
    this.onTtsDone = null;    // () => void — all queued audio finished
    this.onError = null;      // (err) => void
  }

  get enabled() {
    return window.__clientTtsEnabled !== false;
  }

  get active() {
    return this._queue.length > 0 || this._consuming || this._buffer.length > 0;
  }

  /**
   * Feed a token from the LLM stream. Accumulates and dispatches at clause
   * boundaries.
   */
  feed(token) {
    if (!this.enabled) return;
    this._buffer += token;
    if (this._buffer.includes("]]")) {
      this._buffer = this._buffer.replace(CONTROL_TOKEN_RE, "");
    }

    // Very short fragments are the ones the clone truncates. The first one stays
    // small because it sets time-to-first-audio; later ones can be longer at no
    // perceived cost, since audio is already playing by then. Each TTS request
    // pays a ~570 ms fixed startup (~1.4 s when queued behind a prior one), so
    // fewer, larger fragments after the first are strictly cheaper.
    const minChars = this._firstDispatched ? 60 : 8;
    const parts = this._splitBuffer(this._buffer, minChars);

    if (parts.length > 1) {
      for (let i = 0; i < parts.length - 1; i++) {
        this._enqueue(parts[i]);
        this._firstDispatched = true;
      }
      this._buffer = parts[parts.length - 1];
    }
  }

  /**
   * Flush remaining buffer (call when LLM stream ends).
   */
  flush() {
    if (!this.enabled) return;
    const rest = this._buffer.replace(CONTROL_TOKEN_RE, "").trim();
    if (rest) {
      this._enqueue(rest);
    }
    this._buffer = "";
    this._enqueue(null); // sentinel: end of stream
  }

  /**
   * Cancel all pending and active TTS (barge-in).
   */
  cancel() {
    this._cancelled = true;
    this._buffer = "";
    this._queue = [];
    this._firstDispatched = false;
    this._sentenceCount = 0;
    this._turnId = null;
    if (this._activeWs) {
      try { this._activeWs.close(); } catch {}
      this._activeWs = null;
    }
  }

  /**
   * Reset state for a new turn.
   */
  reset() {
    this.cancel();
    this._cancelled = false;
  }

  // -- Internal ----------------------------------------------------------------

  _splitBuffer(text, minChars) {
    const raw = text.split(CLAUSE_RE);
    const merged = [];
    let buf = "";
    for (const part of raw) {
      buf += (buf ? " " : "") + part;
      if (buf.length >= minChars) {
        merged.push(buf);
        buf = "";
      }
    }
    if (buf) merged.push(buf);
    return merged;
  }

  _enqueue(fragment) {
    this._queue.push(fragment);
    if (!this._consuming) {
      this._consume();
    }
  }

  async _consume() {
    this._consuming = true;
    this._cancelled = false;
    if (!this._turnId) {
      this._turnId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    }

    const player = this._getAudioPlayer();
    if (player && this._sentenceCount === 0) {
      player.startSmooth(24000, 1);
      if (this.onTtsStart) this.onTtsStart();
    }

    let sawSentinel = false;
    while (this._queue.length > 0) {
      if (this._cancelled) break;

      const fragment = this._queue.shift();
      if (fragment === null) { // end sentinel
        sawSentinel = true;
        break;
      }

      this._sentenceCount++;
      try {
        await this._synthesize(fragment);
      } catch (err) {
        console.warn("[client-tts] synthesis error:", err);
        if (this.onError) this.onError(err);
      }
    }

    this._consuming = false;

    // Queue drained but the LLM is still streaming: keep the turn id and
    // sentence state so the next fragment of this reply inherits the same
    // speaker lock and no premature done event fires.
    if (!sawSentinel && !this._cancelled) return;

    if (!this._cancelled && player) {
      player.flush();
    }

    this._firstDispatched = false;
    this._sentenceCount = 0;
    this._turnId = null;

    if (!this._cancelled && this.onTtsDone) {
      this.onTtsDone();
    }
  }

  async _synthesize(text) {
    const player = this._getAudioPlayer();
    if (!player?.available) return;

    const host = location.hostname || "localhost";
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${host}:${this._ttsPort}/ws/tts`;

    return new Promise((resolve, reject) => {
      if (this._cancelled) { resolve(); return; }

      const ws = new WebSocket(url);
      this._activeWs = ws;
      let resolved = false;

      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        if (this._cancelled) { ws.close(); resolve(); return; }
        ws.send(JSON.stringify({
          text,
          voice_id: this._getVoiceId(),
          language: this._getLanguage(),
          // Groups this reply's fragments so the server can hold them all to
          // the speaker the first one rendered. Each fragment is an independent
          // generation, and without this they drift into different voices.
          turn_id: this._turnId,
        }));
      };

      ws.onmessage = (ev) => {
        if (this._cancelled) { ws.close(); return; }

        if (typeof ev.data === "string") {
          const pkt = JSON.parse(ev.data);
          if (pkt.type === "done") {
            this._activeWs = null;
            resolved = true;
            resolve();
          } else if (pkt.type === "error") {
            this._activeWs = null;
            resolved = true;
            reject(new Error(pkt.message || "TTS error"));
          } else if (pkt.type === "sample_rate_correction") {
            player.correctSampleRate(pkt.sample_rate);
          }
        } else if (ev.data instanceof ArrayBuffer) {
          const pcm = new Float32Array(ev.data);
          player.write(pcm);
        }
      };

      ws.onerror = (err) => {
        this._activeWs = null;
        if (!resolved) { resolved = true; reject(err); }
      };

      ws.onclose = () => {
        this._activeWs = null;
        if (!resolved) { resolved = true; resolve(); }
      };
    });
  }
}
