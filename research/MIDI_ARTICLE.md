# AI in a MIDI file: the model is the notes

The QR put the model in a URL. The favicon put it in pixels. The PDF put it in `/Keywords`.
This one puts it in **note-on events**.

The file is **4,493 bytes**. The model is the same 160-word SMS spam classifier, 1,579
bytes of int8 weights, **93.54%** on 1,115 held-out SMS. It is packed into **1,053 note-on
events**: each event carries two ASCII characters, one in the note byte and one in the
velocity byte. Together they are the base64url JSON of the model.

`midi.html` parses the MIDI track, rebuilds the string and classifies. The first 128 notes
are playable. The tune is the bytes.

- Demo: https://byjameson-mini-ai.static.hf.space/midi.html
- MIDI: https://byjameson-mini-ai.static.hf.space/model.mid

## The size accounting

| artifact | bytes |
|---|---:|
| model JSON | 1,579 |
| **MIDI** | **4,493** |
| overhead (events, text meta, header) | 2,914 |
| QR URL | 2,158 |
| favicon PNG | 1,519 |
| WAV | 1,627 |
| PDF | 3,832 |
| SVG | 5,004 |

The MIDI is larger than the QR/PNG/WAV because each base64 character becomes a note event
with a status byte and a delta byte. It is still small enough to email or commit.

## What is in the file

The track has five text meta events:

- the title and the model description;
- the measured accuracy;
- a pointer to the viewer;
- `SHA256:<64 hex>`.

Then 1,053 note-on events. The text meta events are readable in any MIDI tool; the note
events are the payload. The header is format 0, one track, 96 ticks per quarter note.

The page parses the events, collects `String.fromCharCode(note, velocity)` pairs, base64
decodes them and runs the int8 Complement NB. No server.

## Why a MIDI?

Because it is an event file that every DAW and player already understands. It is not a
rendered audio file; it is a sequence of events. The model travels in the event stream,
and the file can still be opened and played. A MIDI renderer turns the notes into sound and
throws the note bytes away — which is exactly the container trade.

It is also a good stress test for the parser: variable-length delta times, meta events,
running status, note-on/velocity. The file is small; the code is ordinary.

## What this is not

- **Not an audio model.** The MIDI is not a voice or a song; it is data that can be played.
- **Not a security proof.** The note events are inspectable. A MIDI file is a lossless
  event container, not obfuscation. The SHA-256 is integrity, not a signature.
- **Not robust to a DAW export.** A tool that quantizes, transposes, or renders to WAV can
  change or drop the events. A byte copy is safe; a musical edit is not.
- **Not a production spam filter.** One 80/20 split, 1,115 SMS, one machine.

## The ladder now

- SMS int8: **1,579 bytes**, 93.5% on 1,115 rows;
- QR URL: 2,158 bytes;
- favicon: 1,519 bytes;
- WAV: 1,627 bytes;
- PDF: 3,832 bytes;
- SVG: 5,004 bytes;
- MIDI: 4,493 bytes;
- 130 KB AG News coreset: a 178,761-byte PDF;
- 1.7 MB linear model: 88.5% embedded / 91.0% fresh;
- 5 MB adapter on a shared 4-bit base.

The behavior is the same. The container changes what travels with it and which tool can
destroy it.

- MIDI demo: https://byjameson-mini-ai.static.hf.space/midi.html
- Survival matrix: https://byjameson-mini-ai.static.hf.space/survival.html
- Repo: https://github.com/byjameson/stream-finetune
