"""Will it fit? — a calculator for fine-tuning large models on small GPUs.

The arithmetic lives in fitcore.py; this file is only the UI.
"""
from __future__ import annotations

import gradio as gr

from fitcore import evaluate, render

EXAMPLES = [
    ["Qwen/Qwen2.5-3B", 6, 512],
    ["Qwen/Qwen2.5-7B-Instruct", 6, 512],
    ["Qwen/Qwen2.5-14B-Instruct", 6, 512],
    ["Qwen/Qwen2.5-32B-Instruct", 6, 1024],
    ["Qwen/Qwen2.5-14B-Instruct", 24, 2048],
]


def run(model: str, gpu: float, seq: int) -> str:
    model = (model or "").strip()
    if not model:
        return "Enter a Hugging Face model id, e.g. `Qwen/Qwen2.5-14B-Instruct`."
    try:
        return render(evaluate(model, float(gpu), int(seq)))
    except Exception as e:
        return (f"Could not read `config.json` for **{model}**.\n\n"
                f"`{type(e).__name__}: {e}`\n\n"
                "Use a model id that exists on the Hub and is not gated.")


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(label="Hugging Face model id", value="Qwen/Qwen2.5-14B-Instruct"),
        gr.Slider(4, 80, value=6, step=1, label="GPU VRAM (GB)"),
        gr.Slider(128, 4096, value=512, step=128, label="Sequence length (tokens)"),
    ],
    outputs=gr.Markdown(),
    title="Will it fit?",
    description=(
        "Fine-tune a model that does not fit: stream NF4 weights from disk, one transformer "
        "block on the GPU at a time. This computes whether the whole model fits, and shows "
        "**measured** speed only for configurations that were actually run (RTX 3050 Laptop "
        "6 GB). Unmeasured sizes are labelled as such instead of guessed."
    ),
    examples=EXAMPLES,
    article=(
        "Code, the gradient check and the matched PEFT comparison: "
        "[byjameson/stream-finetune](https://github.com/byjameson/stream-finetune)."
    ),
)

if __name__ == "__main__":
    demo.launch()
