"""Gradio app for the Hugging Face Space (Phase 5).

Plan: upload a clip (max about 30 seconds), draw a count line, get back the annotated
video, heatmap, counts chart and events CSV. GPU work wrapped in `@spaces.GPU` for ZeroGPU,
weights pulled from the Hub model repo at startup.
"""
