// Posts fixed-size Float32 frames (~32 ms at 16 kHz) to the main thread.
class PCMWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(512);
    this.len = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    let i = 0;
    while (i < ch.length) {
      const n = Math.min(ch.length - i, this.buf.length - this.len);
      this.buf.set(ch.subarray(i, i + n), this.len);
      this.len += n;
      i += n;
      if (this.len === this.buf.length) {
        this.port.postMessage(this.buf.slice(0));
        this.len = 0;
      }
    }
    return true;
  }
}
registerProcessor("pcm-worklet", PCMWorklet);
