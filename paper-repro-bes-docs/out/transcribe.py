#!/usr/bin/env python3
"""Transcribe audio with faster-whisper large-v3, GPU with graceful fallback.

Writes:
  out/transcript.json  - full segments + word timestamps
  out/transcript.txt   - readable [mm:ss -> mm:ss] segment transcript
"""
import json, sys
from faster_whisper import WhisperModel

AUDIO = "out/audio16k.wav"

def mmss(s):
    s = int(round(s)); return f"{s//60:02d}:{s%60:02d}"

# Try GPU float16 -> GPU int8_float16 -> CPU int8
attempts = [
    ("cuda", "float16"),
    ("cuda", "int8_float16"),
    ("cpu", "int8"),
]
model = None
for dev, ct in attempts:
    try:
        sys.stderr.write(f"loading large-v3 on {dev}/{ct} ...\n"); sys.stderr.flush()
        model = WhisperModel("large-v3", device=dev, compute_type=ct)
        sys.stderr.write(f"OK: {dev}/{ct}\n"); sys.stderr.flush()
        break
    except Exception as e:
        sys.stderr.write(f"FAILED {dev}/{ct}: {e}\n"); sys.stderr.flush()
if model is None:
    sys.exit("could not load model on any device")

segments, info = model.transcribe(
    AUDIO,
    language="en",
    beam_size=5,
    best_of=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
    word_timestamps=True,
    condition_on_previous_text=True,
)
sys.stderr.write(f"detected language: {info.language} (p={info.language_probability:.2f})\n")

seg_list, txt_lines = [], []
for seg in segments:
    words = [{"start": w.start, "end": w.end, "word": w.word} for w in (seg.words or [])]
    seg_list.append({"start": seg.start, "end": seg.end, "text": seg.text.strip(), "words": words})
    line = f"[{mmss(seg.start)} -> {mmss(seg.end)}] {seg.text.strip()}"
    txt_lines.append(line)
    sys.stderr.write(line + "\n"); sys.stderr.flush()

with open("out/transcript.json", "w") as f:
    json.dump({"language": info.language, "duration": info.duration, "segments": seg_list}, f, indent=2)
with open("out/transcript.txt", "w") as f:
    f.write("\n".join(txt_lines) + "\n")
sys.stderr.write(f"\nDONE: {len(seg_list)} segments\n")
