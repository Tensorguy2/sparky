# Server-side VAD and telephony audio pipeline — scope

## Problem

The voice chatbot's endpointing lives entirely in the browser (`vad-processor.js`
and `app.js`). A telephony client (SIP trunk, Twilio Media Stream, etc.) has no
browser, so the server must handle speech detection, endpointing, and audio
format conversion.

The browser VAD already has a known issue: `endpointMs=200ms` splits natural
speech at clause boundaries, producing answered fragments and unanswered long
segments. The server-side VAD should fix this, not replicate it.

## Audio format conversion

Telephony audio arrives as **8 kHz mu-law** (G.711). The pipeline needs:

| Consumer | Required format | Notes |
|----------|----------------|-------|
| STT (Parakeet) | 16 kHz PCM int16 | Model's native sample rate |
| TTS playback | 24 kHz PCM float32 | TTS server output; telephony needs 8 kHz mu-law back |

### Inbound (caller → server)

1. Receive 8 kHz mu-law frames from the telephony transport (WebSocket or RTP)
2. Decode mu-law → 16-bit linear PCM: `audioop.ulaw2lin(data, 2)` or a lookup table
3. Resample 8 kHz → 16 kHz for STT: `scipy.signal.resample_poly` or `torchaudio.transforms.Resample`
4. Feed 16 kHz PCM to the existing STT pipeline (Parakeet expects 16 kHz, but
   the browser sends 24 kHz which the server already resamples — verify the
   existing resample path)

### Outbound (server → caller)

1. TTS produces 24 kHz float32 PCM
2. Resample 24 kHz → 8 kHz: `scipy.signal.resample_poly(audio, 1, 3)` or
   `torchaudio.transforms.Resample(24000, 8000)`
3. Encode linear PCM → mu-law: `audioop.lin2ulaw(data, 2)` or lookup table
4. Send 8 kHz mu-law frames back over the telephony transport

### Implementation note

All resampling should be pre-initialized (resampler objects, filter coefficients)
at startup, not per-frame. The mu-law codec is a 256-entry lookup table — zero
latency. Resampling a 20 ms frame at 8 kHz (160 samples) to 16 kHz (320 samples)
costs <0.1 ms on CPU.

## Server-side VAD design

### Architecture

Add a new WebSocket endpoint `/ws/telephony` that accepts a continuous audio
stream and handles VAD internally:

```
Telephony client → /ws/telephony (8 kHz mu-law)
    → mu-law decode → resample to 16 kHz
    → server-side VAD
    → on speech-end: feed accumulated PCM to STT
    → on transcript: route through existing LLM pipeline
    → TTS output → resample to 8 kHz → mu-law encode → send back
```

### VAD algorithm

Use **Silero VAD** (MIT license, ~1 MB ONNX model, runs on CPU in <1 ms per
frame). It produces a speech probability per 30 ms frame, which is far more
robust than the browser's RMS-threshold approach.

Alternatively, use the simpler energy-based approach from the browser VAD but
with these fixes for the mid-utterance splitting problem:

#### Adaptive endpointing (recommended)

Instead of a fixed 200 ms silence threshold, use a tiered approach:

- **Short endpoint** (200 ms): triggers a "maybe done" state, but does NOT
  fire `stop_stt` yet
- **Medium endpoint** (600 ms): confirms end-of-speech for short utterances
  (<3 seconds)
- **Long endpoint** (1200 ms): confirms end-of-speech for longer utterances

The rationale: a caller who has been speaking for 10 seconds is in the middle
of explaining something and a 200 ms pause is just breathing. A caller who
said two words and paused for 600 ms is probably done.

This can be combined with the client-side continuation merging (already
implemented) as a fallback.

#### Pre-utterance context

Keep a 500 ms ring buffer of audio before speech onset. Feed this to STT along
with the speech audio so onset consonants are not clipped.

#### Echo cancellation

The telephony transport (SIP/RTP or Twilio) typically handles echo cancellation
at the network level. If not, the server needs to track when TTS audio is being
sent and suppress VAD during that window — equivalent to the browser's
`echoTailMs` mechanism.

## Transport options

### Twilio Media Streams

- WebSocket-based, sends 20 ms chunks of 8 kHz mu-law (base64 encoded in JSON)
- Receives 8 kHz mu-law back (base64 in JSON)
- Simplest integration: add a `/ws/twilio` endpoint that wraps the telephony VAD

### Raw SIP/RTP

- RTP packets with G.711 mu-law payload
- More complex: needs RTP parsing, jitter buffer
- Consider using `pjsua2` or `baresip` as a SIP user agent

### Recommended starting point

Twilio Media Streams — it handles the telephony stack (PSTN, SIP, codecs) and
presents a clean WebSocket interface. The server only needs to handle the
mu-law ↔ PCM conversion and VAD.

## Files to create/modify

| File | Action |
|------|--------|
| `services/telephony_vad.py` | New: server-side VAD with adaptive endpointing |
| `services/audio_codec.py` | New: mu-law ↔ PCM conversion + resampling |
| `routes/telephony.py` | New: `/ws/telephony` WebSocket endpoint |
| `server.py` (bytecode) | Mount the new router — may need a wrapper |
| `config.py` | Add telephony-related config (port, VAD thresholds) |

## Dependencies

- `silero-vad` (optional, for neural VAD): `pip install silero-vad`
- `scipy` (for resampling): already available
- `audioop` (for mu-law): stdlib in Python 3.12 (deprecated in 3.13+, use
  `audioop-lts` backport if needed)

## Estimated effort

- Audio codec + resampling: 1-2 hours
- Server-side VAD with adaptive endpointing: 4-6 hours
- Twilio Media Streams integration: 2-3 hours
- Testing with a live phone call: 2-3 hours
- Total: ~2 days
