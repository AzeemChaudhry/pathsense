"""Gradio fallback demo (optional, not built yet).

Gradio is not a dependency until this demo is built: it pins Pillow below 13, which has
known vulnerabilities (see CHANGELOG.md). Re-add an `app` extra with a Gradio release
that allows a patched Pillow.

The main product is the in browser app in `web/` (docs/PLAN.md, Phases 5 and 6). This
Gradio app is only a fallback for visitors whose browser is too slow for in browser
inference: upload a short clip, get the annotated video, counts and event log back.
It would run on a ZeroGPU Space with GPU work wrapped in `@spaces.GPU`.
"""
