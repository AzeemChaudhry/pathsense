# Market and novelty research

Researched 2026-10-27. Question: is a people detection and tracking app worth building, or
has it been done too often?

## Verdict

Plain counting (YOLO + ByteTrack + line counter + heatmap) has been done many times and
reads as a tutorial project. The market is real, so the skills sell on Fiverr, but the
portfolio edge has to come from elsewhere. Chosen edge: private in browser processing plus
plain language questions over an event log (see `docs/PLAN.md`).

## How crowded it is

| Signal | Finding |
| --- | --- |
| [Ultralytics Solutions](https://docs.ultralytics.com/solutions/) | 17 ready made solutions incl. object counting, counting in regions, heatmaps, queue management, track objects in zone, speed estimation. Counting is about 15 lines of Python or `yolo solutions count`. |
| Hugging Face Spaces | At least 8 "people counting" Spaces, e.g. [matthewrt/people-counting](https://hf.co/spaces/matthewrt/people-counting), several created 2025 to 2026. |
| [GitHub topic people-counting](https://github.com/topics/people-counting) | 32 repos under the topic (many more untagged); one already pairs YOLO counting with a vision language model. |
| [Upwork](https://www.upwork.com/services/product/development-it-a-people-counting-solution-from-video-using-yolo-and-bytetracker-1777468879399690240) | "People counting solution using YOLO and ByteTracker" sold as a packaged service. |

## The market

- People counting systems: about **US$1.52B in 2026**, about **US$3.53B by 2033**,
  12.8% CAGR. Video based systems hold over 60%; software is the fastest growing component;
  retail is the largest application; Asia Pacific the largest and fastest region.
  Key players: RetailNext, Axis, Xovis, Hikvision, Sensormatic, FLIR, DILAX, FootfallCam,
  V-Count, Dahua. ([Persistence Market Research](https://www.persistencemarketresearch.com/market-research/people-counting-system-market.asp))
- Commercial product example ([V-Count 2026 guide](https://v-count.com/retail-people-counting-foot-traffic-analytics-2026-guide/)):
  3D stereo entrance counters from US$299 plus US$9 to US$49 per sensor per month; features
  move beyond counts to dwell, queues, heatmaps, staff exclusion, and linking footfall to
  sales; "up to 99% accuracy" claims, vendor reported.

## What is new in 2026

1. **Vision language models over tracking.** NVIDIA's video search and summarization
   blueprint: a deterministic detector and tracker triggers a VLM to verify alerts, summarise
   scenes and answer questions in plain language. ([NVIDIA Metropolis](https://perspectives.nvidia.com/nvidiametropolis/task/faq/vision-language-models-video-understanding-1/))
2. **Privacy and the EU AI Act.** Anonymous counting is allowed; identification, biometric
   categorisation and emotion recognition are restricted. Article 50 transparency rules apply
   from 2 Aug 2026; stand alone high risk obligations from 2 Dec 2027. Systems must show what
   they do not do. ([BumbeeLabs](https://bumbeelabs.com/eu-ai-act-people-counting/))
3. **Open vocabulary tracking.** [SAM 3](https://docs.ultralytics.com/models/sam-3) tracks
   any concept from a text prompt, but costs about 2.9 s per image on a 96 GB GPU versus
   8.4 ms for YOLO26n seg: a labelling teacher, not a live model. Weights are gated on HF.
4. **LLMs in the browser.** WebLLM and transformers.js run Qwen3 0.6B (335 MB download) to
   Qwen3.5 2B (1 GB) on WebGPU; 1B class models reach roughly 27 to 67 tokens/s on an M3
   Pro. WebGPU on Linux is limited (Chrome: Intel Gen12+; Firefox: none).
   ([Pinggy, Sep 2026](https://pinggy.io/blog/run_llm_in_browser_webgpu/))

## Implications for PathSense

- Never use or advertise identification, faces, demographics or emotion. Say so in the UI.
- Process on device by default; that is both the privacy story and the free hosting story.
- Keep counting as the foundation, but make the event log and Q&A the headline.
