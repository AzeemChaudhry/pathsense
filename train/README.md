# Fine tuning (Phase 4)

- Pick one failure case (overhead CCTV views recommended).
- Film 10 to 20 clips with consent; split train and test **by clip, never by frame**.
- Annotate 300 to 800 frames in CVAT; push to a Hugging Face dataset repo.
- Fine tune YOLO26n locally (4 GB VRAM) and RF-DETR Nano (cloud GPU if it runs out of memory).
- Log every run and add the result to NOTES.md and the README benchmark table.
