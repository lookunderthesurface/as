# GPU model evaluation plan

The implementation boundary remains fixed: REAL Screenpipe, REAL Ollama with
the existing `qwen3-vl:2b-instruct-q4_K_M`, REAL SQLite/Policy, and Mock Cloud.
The evaluation is a measurement pass, not a model expansion.

Record for text and vision separately:

- provider/model/runtime version and `nvidia-smi` driver/GPU memory snapshot;
- wall latency, Ollama total duration, model-load duration, prompt/output token
  counts, prompt-eval and generation durations;
- structured-output success rate and recovery after a forced local timeout or
  malformed response;
- action agreement on the ten privacy-safe replay scenarios;
- shadow notification candidates versus actual delivered notifications (zero
  delivered in shadow mode);
- CPU/GPU utilization, peak VRAM, and behavior after a 15-minute bounded run.

Acceptance gates are conservative: no privacy regression, no raw text leakage,
no notification hard-rule bypass, no growing inference backlog, and no control
of external Screenpipe. Only after these pass should a separate experiment
consider changing an inference setting; no new model is part of this plan.
