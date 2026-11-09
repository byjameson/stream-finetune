"""Put the 1,579-byte SMS model in a MIDI file.

The model is base64url JSON in note-on events: each event carries two ASCII characters in
its note/velocity bytes. `midi.html` parses the track, rebuilds the string and classifies.
The text meta events hold the model card and SHA-256.

    python demo/build_midi.py
"""
import base64
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WAV = ROOT / "demo/model.wav"
OUT = ROOT / "demo/model.mid"


def read_model():
    raw = WAV.read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    off, data = 12, None
    while off + 8 <= len(raw):
        kind = raw[off:off + 4]
        size = struct.unpack("<I", raw[off + 4:off + 8])[0]
        if kind == b"data":
            data = raw[off + 8:off + 8 + size]
            break
        off += 8 + size + (size & 1)
    assert data is not None
    n = struct.unpack("<I", data[:4])[0]
    blob = data[4:4 + n]
    return blob, json.loads(blob)


def vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def text_event(text):
    data = text.encode("ascii")
    return b"\x00\xff\x01" + vlq(len(data)) + data


def build():
    blob, model = read_model()
    assert len(blob) == 1579
    payload = base64.urlsafe_b64encode(blob).decode().rstrip("=")
    assert len(payload) % 2 == 0, len(payload)
    digest = hashlib.sha256(blob).hexdigest()

    track = bytearray()
    track += text_event("AI in a MIDI")
    track += text_event(
        f"160-word SMS spam model; {len(blob)} bytes; {len(payload)} chars in note events")
    track += text_event(f"Accuracy {model['a'] * 100:.2f}% on 1,115 held-out SMS")
    track += text_event("Open midi.html to parse the notes and classify")
    track += text_event(f"SHA256:{digest}")
    track += b"\x00\xff\x51\x03\x07\xa1\x20"  # 120 bpm
    for i in range(0, len(payload), 2):
        note = ord(payload[i])
        vel = ord(payload[i + 1])
        assert 1 <= note <= 127 and 1 <= vel <= 127
        track += b"\x00\x90" + bytes([note, vel])
    track += b"\x00\xff\x2f\x00"

    midi = b"MThd" + struct.pack(">IHHH", 6, 0, 1, 96)
    midi += b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    OUT.write_bytes(midi)
    print(f"model={len(blob)} B  payload={len(payload)} chars  events={len(payload)//2}  "
          f"midi={len(midi)} B  sha={digest[:16]}...")


if __name__ == "__main__":
    build()
