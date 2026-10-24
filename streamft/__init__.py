"""streamft — fine-tune models that do not fit, one block at a time.

The engine memory-maps a pre-quantized NF4 checkpoint, materialises one transformer
block on the GPU at a time (with recompute in the backward) and keeps a LoRA adapter
trainable. Measured on a 6 GB RTX 3050 Laptop: a 32B model at 1024 tokens/step,
36.7 tok/s, 4.61 GB peak.

Modules:

  cli                command line entry point (``python -m streamft.cli --help``)
  disk_quant_store   mmap'd NF4 checkpoint, block loader, layer cache
  stream_quant_lora  streaming trainer (base frozen, LoRA trainable)
  stream_train       chunked loss/backward, per-layer forward hooks, streaming eval
  autotune           chooses the resident-block window from VRAM/RAM
  data               token shards and the train/eval samplers
  metrics            append-only JSONL evidence log
  lora               minimal LoRA layers and adapter state dicts

The training tag written to the evidence log stays ``exp54_disk_qlora`` so that old
and new rows are directly comparable.
"""

__all__ = [
    "autotune",
    "cli",
    "data",
    "disk_quant_store",
    "lora",
    "metrics",
    "stream_quant_lora",
    "stream_train",
]
