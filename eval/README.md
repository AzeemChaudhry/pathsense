# Evaluation (Phase 3)

- Export tracks in MOTChallenge format: `frame, id, x, y, w, h, conf, -1, -1, -1`.
- Score with [TrackEval](https://github.com/JonathonLuiten/TrackEval) (HOTA, IDF1, MOTA).
- Benchmark data: MOT17 train, split each sequence in half (first half train, second half val).
- MOTChallenge data is CC BY NC SA 3.0: research and evaluation only, never in this repo.
