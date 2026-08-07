/**
 * AudioWorkletProcessor for low-latency VAD and PCM capture.
 *
 * Runs in the audio rendering thread at 24 kHz. Produces 128-sample frames
 * (~5.3 ms each). Accumulates into configurable "VAD frames" (default 480
 * samples = 20 ms) before posting RMS + PCM to the main thread.
 *
 * Messages TO main thread:
 *   { type: "vad", rms: number, pcm: Int16Array.buffer }
 *
 * Messages FROM main thread:
 *   { type: "pause" }  — stop emitting (mic muted logically)
 *   { type: "resume" } — start emitting
 */

class VadProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options?.processorOptions || {};
    this._frameSamples = opts.frameSamples || 480; // 20 ms @ 24 kHz
    this._buf = new Float32Array(this._frameSamples);
    this._pos = 0;
    this._active = true;

    this.port.onmessage = (e) => {
      if (e.data.type === "pause") this._active = false;
      else if (e.data.type === "resume") this._active = true;
    };
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input || !this._active) return true;

    let i = 0;
    while (i < input.length) {
      const space = this._frameSamples - this._pos;
      const take = Math.min(space, input.length - i);
      this._buf.set(input.subarray(i, i + take), this._pos);
      this._pos += take;
      i += take;

      if (this._pos >= this._frameSamples) {
        // Compute RMS
        let sum = 0;
        for (let j = 0; j < this._frameSamples; j++) {
          sum += this._buf[j] * this._buf[j];
        }
        const rms = Math.sqrt(sum / this._frameSamples);

        // Convert to PCM16
        const pcm16 = new Int16Array(this._frameSamples);
        for (let j = 0; j < this._frameSamples; j++) {
          const s = Math.max(-1, Math.min(1, this._buf[j]));
          pcm16[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        this.port.postMessage(
          { type: "vad", rms, pcm: pcm16.buffer },
          [pcm16.buffer]
        );
        this._pos = 0;
      }
    }
    return true;
  }
}

registerProcessor("vad-processor", VadProcessor);
