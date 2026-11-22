# Changelog

## 1.0.0 - streamft trainer only (2026-11-22)

- Removed the bundled assistant modules and assistant console script from the
  streamft wheel; `streamft` is now the low-VRAM trainer only.
- Removed `pypdf` and `python-docx` from the default dependencies.
- The assistant continues as a separate distribution.
- `streamft 0.9.2` remains the final compatibility release with the bundled
  assistant.
- Release notes: [`docs/RELEASE_1.0.0.md`](docs/RELEASE_1.0.0.md).
