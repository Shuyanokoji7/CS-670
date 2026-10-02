# Note on this freeze snapshot's hash files (added 2026-10-02, after the final B2 review)

The snapshot was taken at 2026-10-02T17:26:16+05:30, after the freeze entry and `protocol_frozen: true` and before the final
T = 50 accounting and sweep. Its contents and that timestamp are unchanged.

`SNAPSHOT_SHA256.txt` (the original, preserved unchanged) is **not** a reliable `sha256sum -c` manifest, for two reasons:
- It includes a hash of itself, taken while it was still being written.
- It includes hashes of mutable live files (`configs/b2.yaml`, `RESEARCH_LOG.md`) at their repository paths. Those files
  have changed since.

Its hashes of the snapshot's own copies remain valid evidence of the freeze-time contents.

`MANIFEST.sha256` (added later) is the reliable manifest:
- it lists every file in this directory except itself, with paths relative to the directory;
- verify it with `cd <this directory> && sha256sum -c MANIFEST.sha256`.
