/**
 * Streams pcm_f32le chunks via Web Audio API.
 *
 * Strategy: continuous scheduling with an adaptive pre-roll jitter buffer.
 *
 * Why a pre-roll is required (measured on this server, see
 * playground/probe_ws_timing.py):
 *   - The model has ~3 s first-byte latency, then emits one decoded chunk of
 *     ~960 ms of audio every ~1000-1050 ms (RTF ≈ 1.04). That alone causes a
 *     ~40-60 ms underrun per chunk if we play the very first chunk
 *     immediately.
 *   - Between sentences the GPU re-primes, producing a ~1.0-1.3 s wall-time
 *     gap for ~0.24 s of trailing audio. The audio chain depletes by ~1.0 s
 *     at every sentence boundary.
 *
 * If we scheduled each chunk at `ctx.currentTime` (or with no headroom over
 * `_nextStart`), the chain underruns on the first late chunk and every
 * subsequent chunk plays late → audible clicks and gaps. That's the
 * "choppy/glitchy" symptom reported by users.
 *
 * Fix: accumulate incoming samples until the chain has a healthy lead over
 * the audio clock, then drain into a chain of AudioBufferSourceNodes whose
 * `start()` times are tracked via `_nextStart`. The pre-roll size is chosen
 * adaptively from `num_sentences` so single-sentence utterances stay
 * low-latency while multi-sentence text gets enough buffer to survive the
 * inter-sentence stall.
 *
 * Tunables (all in seconds):
 *   prerollSingleSentence – pre-roll used when only one sentence is coming.
 *                           Small (200 ms) so short answers feel responsive.
 *                           Single-sentence chunks usually only need to absorb
 *                           ~50 ms of jitter.
 *   prerollMultiSentence  – pre-roll used as soon as we know >1 sentence is
 *                           coming. Needs to exceed the worst-case
 *                           inter-sentence gap minus the tail chunk's
 *                           duration (~1.1 s on this server). 1.5 s gives
 *                           ~0.4 s of safety margin.
 *   reprimeSeconds        – pre-roll re-applied if the chain ever underruns
 *                           mid-stream. Larger = fewer subsequent clicks,
 *                           smaller = less audible silence on the underrun.
 *   scheduleAhead         – tiny safety margin so the audio thread has time
 *                           to pick up freshly scheduled sources.
 */
export class AudioStreamPlayer {
  constructor({
    prerollSingleSentence = 0.2,
    prerollMultiSentence = 1.2,
    // 0.5 s (was 0.3): with the v3 server generating faster than real time
    // (RTF < 0.8) underruns should be rare; when one does happen, a larger
    // re-prime prevents the repeated click/gap cascade seen with v2.
    reprimeSeconds = 0.5,
    scheduleAhead = 0.02,
  } = {}) {
    /** @type {AudioContext | null} */
    this._ctx = null;
    /** Sample rate reported by the server (used as the AudioBuffer's rate). */
    this._serverSampleRate = 24000;
    /** Audio-context time at which the next source should start. */
    this._nextStart = 0;
    /** @type {AudioBufferSourceNode[]} */
    this._sources = [];
    /** @type {Float32Array[]} */
    this._pending = [];
    this._pendingSamples = 0;
    this._started = false;
    this._closed = false;
    /** Effective pre-roll for the current request, in seconds. */
    this._prerollSeconds = prerollSingleSentence;

    this.prerollSingleSentence = prerollSingleSentence;
    this.prerollMultiSentence = prerollMultiSentence;
    this.reprimeSeconds = reprimeSeconds;
    this.scheduleAhead = scheduleAhead;

    /** Mid-stream underruns since construction (proof metric for smoothness). */
    this.underrunCount = 0;
    /** True once the first chunk of the current stream has been scheduled. */
    this._streamPrimed = false;

    this.available = true;
  }

  /**
   * Create the underlying AudioContext. **Must be called from a user gesture
   * handler** (e.g. the Generate button's click listener) so the context is
   * not born suspended on Chrome/Safari.
   *
   * Safe to call multiple times: subsequent calls reuse the existing context
   * and just resume it if it has gone suspended.
   *
   * @param {number} [sampleRate] – preferred context rate (24000 matches the
   *   server and avoids per-buffer resampling on most browsers). If the
   *   browser refuses, we fall back to its default; AudioBuffer-level
   *   resampling will still produce correct output.
   */
  async init(sampleRate = 24000) {
    if (this._ctx) {
      if (this._ctx.state === "suspended") {
        try { await this._ctx.resume(); } catch { /* ignore */ }
      }
      return;
    }
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) {
      this.available = false;
      return;
    }
    // `latencyHint: "playback"` asks the browser for a larger internal
    // buffer, which gives us extra slack on top of the JS-level pre-roll.
    // Interactive latency isn't needed for one-way TTS playback.
    try {
      this._ctx = new Ctx({ sampleRate, latencyHint: "playback" });
    } catch {
      try {
        this._ctx = new Ctx({ latencyHint: "playback" });
      } catch {
        this.available = false;
        return;
      }
    }
    if (this._ctx.state === "suspended") {
      try { await this._ctx.resume(); } catch { /* ignore */ }
    }
    this._nextStart = this._ctx.currentTime;
  }

  /**
   * Reset scheduling state for a new synthesis request. Does NOT recreate the
   * AudioContext (that must happen in a user gesture via `init()`).
   *
   * @param {number} sampleRate – server-reported sample rate for incoming PCM.
   * @param {number} [numSentences=1] – used to pick the pre-roll size.
   */
  start(sampleRate, numSentences = 1) {
    this._serverSampleRate = sampleRate;
    this._prerollSeconds =
      numSentences > 1 ? this.prerollMultiSentence : this.prerollSingleSentence;
    this._stopAllSources();
    this._pending = [];
    this._pendingSamples = 0;
    this._started = false;
    this._streamPrimed = false;
    if (this._ctx) {
      this._nextStart = this._ctx.currentTime;
    }
  }

  /**
   * Like start(), but doesn't interrupt currently playing audio.
   * Schedules new audio to begin after whatever is already queued.
   */
  startSmooth(sampleRate, numSentences = 1) {
    this._serverSampleRate = sampleRate;
    this._prerollSeconds =
      numSentences > 1 ? this.prerollMultiSentence : this.prerollSingleSentence;
    this._pending = [];
    this._pendingSamples = 0;
    this._started = false;
    // Note: keep _streamPrimed as-is — startSmooth continues an active chain,
    // so a stall right after it is still a real underrun.
  }

  /**
   * Buffer a chunk and (once the pre-roll threshold is met) drain into the
   * Web Audio scheduling chain.
   *
   * @param {Float32Array} samples
   */
  write(samples) {
    if (!this.available || !this._ctx || this._closed) return;
    if (samples.length === 0) return;

    this._pending.push(samples);
    this._pendingSamples += samples.length;

    if (!this._started) {
      const queuedSec = this._pendingSamples / this._serverSampleRate;
      if (queuedSec < this._prerollSeconds) return;
      this._started = true;
    }
    this._drainPending();
  }

  /**
   * Flush any remaining buffered samples (call when the stream ends so the
   * tail of a short utterance — shorter than the pre-roll target — still
   * plays).
   */
  flush() {
    if (!this.available || !this._ctx || this._closed) return;
    if (this._pending.length === 0) return;
    this._started = true;
    this._drainPending();
  }

  /**
   * Server told us its actual sample rate differs from 24000. Don't tear the
   * context down — just update the server rate so subsequent AudioBuffers
   * are tagged correctly and the browser resamples them on playback.
   *
   * @param {number} sampleRate
   */
  correctSampleRate(sampleRate) {
    this._serverSampleRate = sampleRate;
  }

  stopPlayback() {
    this._stopAllSources();
    this._pending = [];
    this._pendingSamples = 0;
    this._started = false;
    this._streamPrimed = false;
    if (this._ctx) {
      this._nextStart = this._ctx.currentTime;
    }
  }

  async close() {
    this._closed = true;
    this.stopPlayback();
    if (this._ctx) {
      try { await this._ctx.close(); } catch { /* ignore */ }
      this._ctx = null;
    }
  }

  // ── Internals ────────────────────────────────────────────────────────────

  _drainPending() {
    for (const samples of this._pending) {
      this._scheduleChunk(samples);
    }
    this._pending = [];
    this._pendingSamples = 0;
  }

  _scheduleChunk(samples) {
    const ctx = this._ctx;
    if (!ctx) return;

    // Tag the buffer with the server's sample rate. If it differs from the
    // context rate the browser will resample on playback — gapless and
    // transparent for typical voice content at 24 kHz.
    const buf = ctx.createBuffer(1, samples.length, this._serverSampleRate);
    buf.copyToChannel(samples, 0);

    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);

    const now = ctx.currentTime;
    let startAt;
    if (this._nextStart > now) {
      // Chain is still ahead of the clock: schedule immediately after the
      // currently queued tail for sample-accurate continuity.
      startAt = this._nextStart;
    } else {
      // The chain has stalled (underrun) or this is the first source after
      // reset. Re-prime with a small pre-roll so this chunk (and any that
      // follow in this burst) have headroom against further jitter.
      if (this._streamPrimed) {
        // Mid-stream stall, not a fresh start: count it as an underrun.
        this.underrunCount += 1;
        console.warn(
          `[audio-stream] underrun #${this.underrunCount} ` +
          `(gap ${(now - this._nextStart).toFixed(3)}s, repriming ${this.reprimeSeconds}s)`,
        );
      }
      startAt = now + this.reprimeSeconds + this.scheduleAhead;
    }
    this._streamPrimed = true;
    src.start(startAt);
    this._nextStart = startAt + buf.duration;

    this._sources.push(src);
    src.onended = () => {
      const i = this._sources.indexOf(src);
      if (i >= 0) this._sources.splice(i, 1);
    };
  }

  _stopAllSources() {
    for (const src of this._sources) {
      try { src.onended = null; } catch { /* ignore */ }
      try { src.stop(); } catch { /* already stopped */ }
      try { src.disconnect(); } catch { /* ignore */ }
    }
    this._sources = [];
  }
}
