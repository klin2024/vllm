# Vendored third-party sources

Provenance and licensing for files vendored into `vllm/third_party/` by this
branch. Pre-existing entries in this directory are not covered here; see their
own headers and `LICENSE` files.

## flash_attn_rotary.py

Pure-Triton rotary embedding kernel, vendored so vLLM can use it without the
`flash_attn` package installed.

| | |
|---|---|
| Upstream | https://github.com/Dao-AILab/flash-attention |
| Branch | `main` |
| Commit | `a9a3170fc98cbd22a4cc870937b390f3d483f1eb` (2025-04-25) |
| Upstream path | `flash_attn/ops/triton/rotary.py` |
| Blob | `ff4017fda3e4a6e18cf3a51b34f0fb073d8f678a` (7088 bytes) |
| License | BSD-3-Clause |
| Copyright | Copyright (c) 2025, Tri Dao |
| Modifications | None. A provenance header was prepended; the body is byte-for-byte identical to the upstream blob. |

The revision was pinned by blob hash rather than by date or tag: the vendored
body hashes to `ff4017fd...`, which matches that upstream path at commit
`a9a3170f` and at no other commit that has touched the file. Upstream `main`
still carries the same blob, so the copy is current as of vendoring.

To check that the body has not drifted:

```bash
tail -n +10 vllm/third_party/flash_attn_rotary.py | git hash-object --stdin
# -> ff4017fda3e4a6e18cf3a51b34f0fb073d8f678a
```

The upstream copyright notice is retained in the file, immediately below the
provenance header, as BSD-3-Clause requires.
