# Browser app (Phases 5 and 6)

Static Hugging Face Space. Everything runs in the visitor's browser; no video is uploaded.

Planned pieces (see `docs/PLAN.md`):

- `detector`: ONNX person detector via onnxruntime-web, WebGPU with WASM fallback.
- `tracker`: ByteTrack ported to TypeScript, parity tested against the Python tracker.
- `analytics`: line counts, zones, dwell, heatmap, event log (same schema as Python).
- `qa`: query functions over the event log plus a small in browser LLM (WebLLM or
  transformers.js) that picks a function and words the answer.

Tooling choice (plain TypeScript + Vite, or similar) is made at the start of Phase 5.
