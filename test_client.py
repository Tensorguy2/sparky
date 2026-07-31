"""
Quick WebSocket test client for the TTS server.
Usage:
    python test_client.py                          # use Mikey voice
    python test_client.py --voice_id mikey         # explicit voice
    python test_client.py --text "Hello world"
"""
import argparse
import asyncio
import json
import struct
import wave

import numpy as np
import websockets


async def stream_tts(text: str, voice_id: str, out_path: str, server: str):
    uri = f"ws://{server}/ws/tts"
    print(f"Connecting to {uri} …")

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"text": text, "language": "English", "voice_id": voice_id}))
        print("Request sent. Waiting for audio …")

        sample_rate = 24000
        all_pcm: list[bytes] = []
        expecting_binary = False

        async for msg in ws:
            if isinstance(msg, str):
                pkt = json.loads(msg)
                t = pkt.get("type")

                if t == "metadata":
                    sample_rate = pkt["sample_rate"]
                    print(f"[metadata] SR={sample_rate}  encoding={pkt['encoding']}  sentences={pkt['num_sentences']}")

                elif t == "sample_rate":
                    sample_rate = pkt["sample_rate"]
                    print(f"[sample_rate] Updated SR → {sample_rate}")

                elif t == "chunk_header":
                    idx = pkt["index"]
                    ns = pkt["num_samples"]
                    sb = pkt["size_bytes"]
                    print(f"[chunk {idx}] {ns} samples  {sb} bytes  ({ns/sample_rate*1000:.0f} ms)")
                    expecting_binary = True

                elif t == "done":
                    print("[done]")
                    break

                elif t == "error":
                    print(f"[ERROR] {pkt['message']}")
                    break

            elif isinstance(msg, bytes):
                if expecting_binary:
                    all_pcm.append(msg)
                    expecting_binary = False

        if all_pcm:
            audio = np.frombuffer(b"".join(all_pcm), dtype=np.float32)
            # Write WAV
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(out_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm16.tobytes())
            print(f"Saved {len(audio)/sample_rate:.2f}s of audio → {out_path}")
        else:
            print("No audio received.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="Hey there! This is a streaming TTS test. The audio should start playing almost immediately after the first sentence is ready.")
    p.add_argument("--voice_id", default="mikey")
    p.add_argument("--out", default="test_output.wav")
    p.add_argument("--server", default="localhost:25565")
    args = p.parse_args()

    asyncio.run(stream_tts(args.text, args.voice_id, args.out, args.server))
