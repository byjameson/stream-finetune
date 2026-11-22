# Changelog

## 1.0.1 - trainer-only metadata cleanup (2026-11-22)

- Removed the last assistant references from the package description, README,
  project URLs, docstrings and GitHub repository metadata.
- The source tree no longer contains the assistant package, docs, examples or
  tests; those live in the separate assistant repository.
- No trainer behavior changed.
- Release notes: [`docs/RELEASE_1.0.1.md`](docs/RELEASE_1.0.1.md).

## 1.0.0 - streamft trainer only (2026-11-22)

- Removed the bundled assistant modules and assistant console script from the
  streamft wheel; `streamft` is now the low-VRAM trainer only.
- Removed `pypdf` and `python-docx` from the default dependencies.
- The assistant continues as a separate distribution.
- `streamft 0.9.2` remains the final compatibility release with the bundled
  assistant.
- Release notes: [`docs/RELEASE_1.0.0.md`](docs/RELEASE_1.0.0.md).
