# NVIDIA / Windows opt-in checklist

This is a manual checklist for the existing RTX 4070 SUPER machine. It is
outside the CPU stabilization pass and does not authorize installing drivers,
CUDA, Ollama models, or changing versions.

1. Confirm the NVIDIA driver is installed and `nvidia-smi` reports the expected
   adapter. Record the driver version and available VRAM.
2. Confirm the already-installed Ollama runtime answers its local version and
   tags endpoints, and that only `qwen3-vl:2b-instruct-q4_K_M` is selected.
3. Run `secretary preflight` and then the explicit local inference status probe.
4. Run text and vision smoke tests, recording latency and memory behavior.
5. Run the short real Screenpipe shadow session with `--mock-notifications`.
6. Inspect `session-report` and `recent-decisions`; confirm no screenshots or
   raw OCR appear in logs/database output.
7. Stop the run and verify that only Secretary-owned Screenpipe is cleaned up;
   an external Screenpipe must remain untouched.

Do not run a GPU test while the machine is unattended. Stop if the configured
model is missing, a driver/runtime change is proposed, or privacy flags are not
visible in the managed command.

