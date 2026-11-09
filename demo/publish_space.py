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
api.upload_file(path_or_fileobj=str(ROOT / "demo/max-ai.html"),
                path_in_repo="max-ai.html", repo_id=RID, repo_type="space",
                commit_message="mini-ai: 1.7 MB accuracy mode")
api.upload_file(path_or_fileobj=str(ROOT / "demo/space_README.md"),
                path_in_repo="README.md", repo_id=RID, repo_type="space",
                commit_message="mini-ai: card")
print(f"https://huggingface.co/spaces/{RID}")
