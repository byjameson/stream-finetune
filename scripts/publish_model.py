"""Stage and publish the Atom Zero 49M weights.

    python scripts/publish_model.py --stage-only
    HF_TOKEN=... python scripts/publish_model.py --target hf
    MODELSCOPE_API_TOKEN=... python scripts/publish_model.py --target modelscope

The model is a custom PyTorch checkpoint, not a transformers AutoModel. The
staged folder contains the checkpoint, tokenizer, config, LICENSE, NOTICE and a
model card. Tokens are read from the environment and are never written to disk.

Canonical model publication is Hugging Face, mirrored to ModelScope. Both use
the same staged folder; ModelScope receives the card without HF front matter.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints/transformer_behavior_50m_scaled"
TOKENIZER = ROOT / "data/chat_en_bpe8k.json"
MODEL_CARD = ROOT / "docs/MODEL_CARD.md"
STAGE = ROOT / "dist/atom-zero-49m"

FRONT_MATTER = """---
license: apache-2.0
language:
- en
tags:
- atom-zero
- local-llm
- rag
- source-cited
- small-language-model
- pytorch
library_name: pytorch
---

"""


def stage() -> Path:
    if not (CHECKPOINT / "model.pt").exists():
        raise SystemExit(f"missing checkpoint: {CHECKPOINT / 'model.pt'}")
    if not TOKENIZER.exists():
        raise SystemExit(f"missing tokenizer: {TOKENIZER}")
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copy2(CHECKPOINT / "model.pt", STAGE / "model.pt")
    shutil.copy2(CHECKPOINT / "config.json", STAGE / "config.json")
    shutil.copy2(TOKENIZER, STAGE / "chat_en_bpe8k.json")
    shutil.copy2(ROOT / "LICENSE", STAGE / "LICENSE")
    shutil.copy2(ROOT / "NOTICE", STAGE / "NOTICE")
    card = FRONT_MATTER + MODEL_CARD.read_text(encoding="utf-8")
    (STAGE / "README.md").write_text(card, encoding="utf-8")
    return STAGE


def publish_hf(folder: Path, repo_id: str) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("HF_TOKEN missing; skipped Hugging Face upload")
        return
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(folder),
        commit_message="Atom Zero 0.8.0 weights")
    print(f"uploaded Hugging Face: https://huggingface.co/{repo_id}")


def _strip_hf_front_matter(text: str) -> str:
    """ModelScope renders HF YAML front matter poorly (it can return 500)."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5:].lstrip("\n")
    return text


def publish_modelscope(folder: Path, model_id: str) -> None:
    token = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get(
        "MODELSCOPE_TOKEN")
    if not token:
        print("MODELSCOPE_API_TOKEN missing; skipped ModelScope upload")
        return
    from modelscope.hub.api import HubApi
    api = HubApi()
    api.login(token)
    try:
        api.create_model(model_id)
    except Exception as exc:  # already exists is fine
        print(f"ModelScope create_model: {exc}")
    api.upload_folder(repo_id=model_id, folder_path=str(folder))
    card = folder / "README.md"
    if card.exists():
        original = card.read_text(encoding="utf-8")
        stripped = _strip_hf_front_matter(original)
        if stripped != original:
            tmp = folder / ".README.modelscope.md"
            tmp.write_text(stripped, encoding="utf-8")
            try:
                api.upload_file(
                    repo_id=model_id,
                    path_or_fileobj=str(tmp),
                    path_in_repo="README.md",
                    commit_message="docs: ModelScope model card without HF front matter",
                )
            finally:
                tmp.unlink(missing_ok=True)
    print(f"uploaded ModelScope: https://modelscope.cn/models/{model_id}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("stage", "hf", "modelscope", "both"),
                    default="stage")
    ap.add_argument("--hf-repo", default="byjameson/atom-zero-49m")
    ap.add_argument("--modelscope-repo", default="byjameson/atom-zero-49m")
    args = ap.parse_args()
    folder = stage()
    print(f"staged: {folder} ({sum(p.stat().st_size for p in folder.iterdir()) / 1e6:.1f} MB)")
    if args.target in ("hf", "both"):
        publish_hf(folder, args.hf_repo)
    if args.target in ("modelscope", "both"):
        publish_modelscope(folder, args.modelscope_repo)
    if args.target == "stage":
        print("stage only; set HF_TOKEN / MODELSCOPE_API_TOKEN to upload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
