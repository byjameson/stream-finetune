import os
from pathlib import Path

from huggingface_hub import HfApi

RID = "byjameson/mini-ai"
ROOT = Path(__file__).resolve().parent.parent

tok = os.environ["HF_TOKEN"]
api = HfApi(token=tok)
api.create_repo(RID, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_file(path_or_fileobj=str(ROOT / "demo/mini-ai.html"),
                path_in_repo="index.html", repo_id=RID, repo_type="space",
                commit_message="mini-ai: 130 KB browser classifier")
# Same page under its filename so the max-ai.html back-link works on the Space too.
api.upload_file(path_or_fileobj=str(ROOT / "demo/mini-ai.html"),
                path_in_repo="mini-ai.html", repo_id=RID, repo_type="space",
                commit_message="mini-ai: stable filename alias")
api.upload_file(path_or_fileobj=str(ROOT / "demo/max-ai.html"),
                path_in_repo="max-ai.html", repo_id=RID, repo_type="space",
                commit_message="mini-ai: 1.7 MB accuracy mode")
api.upload_file(path_or_fileobj=str(ROOT / "demo/maker.html"),
                path_in_repo="maker.html", repo_id=RID, repo_type="space",
                commit_message="mini-ai: Tiny AI Maker")
api.upload_file(path_or_fileobj=str(ROOT / "demo/qr.html"),
                path_in_repo="qr.html", repo_id=RID, repo_type="space",
                commit_message="mini-ai: AI in a QR code")
api.upload_file(path_or_fileobj=str(ROOT / "demo/qrcode.min.js"),
                path_in_repo="qrcode.min.js", repo_id=RID, repo_type="space",
                commit_message="mini-ai: QR encoder")
api.upload_file(path_or_fileobj=str(ROOT / "demo/qr-spam.png"),
                path_in_repo="qr-spam.png", repo_id=RID, repo_type="space",
                commit_message="mini-ai: QR model image")
api.upload_file(path_or_fileobj=str(ROOT / "demo/leaderboard.html"),
                path_in_repo="leaderboard.html", repo_id=RID, repo_type="space",
                commit_message="mini-ai: minimum-bit table")
api.upload_file(path_or_fileobj=str(ROOT / "demo/diff.html"),
                path_in_repo="diff.html", repo_id=RID, repo_type="space",
                commit_message="mini-ai: adapter autopsy")
api.upload_file(path_or_fileobj=str(ROOT / "demo/watermark.html"),
                path_in_repo="watermark.html", repo_id=RID, repo_type="space",
                commit_message="mini-ai: behavioural fingerprint")
api.upload_file(path_or_fileobj=str(ROOT / "demo/space_README.md"),
                path_in_repo="README.md", repo_id=RID, repo_type="space",
                commit_message="mini-ai: card")
print(f"https://huggingface.co/spaces/{RID}")
