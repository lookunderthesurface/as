# Ambient Secretary

Privacy-first Windows work companion MVP. It observes a real Screenpipe runtime by
default, reduces events to safe normalized signals, sends them through the
configured local provider (`MockInferenceProvider` by default or the explicit
Ollama qwen3-vl provider), maintains bounded working state, remembers important
events in real SQLite, watches repeated failures, and interrupts only after
deterministic evidence, model candidate thresholds, and hard notification gates
agree.

The default configuration deliberately does not automate the computer, record
audio, call a real LLM, send screenshots to the cloud, or install system services.
An explicit Ollama provider/smoke command is available for local-only testing.

## Quick start

From `C:\Users\huwencan\workspace\0828`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The package has no mandatory third-party runtime dependencies. Optional live
Windows Toast and tray support is available with:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[windows]"
```

That command installs only into this project's `.venv`; it does not modify global
Python or Windows. It is not required for unit tests or replay.

## Commands

```powershell
# Deterministic tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Reproduce the state machine (isolated in-memory state)
.\.venv\Scripts\python.exe -m secretary replay scenarios\repeated_failure.jsonl

# Check local development readiness; no token is printed
.\.venv\Scripts\python.exe -m secretary preflight
.\.venv\Scripts\python.exe -m secretary doctor

# Show local inference configuration without probing a model runtime
.\.venv\Scripts\python.exe -m secretary inference-status

# Explicitly check Ollama version and configured model; never pulls/upgrades
.\.venv\Scripts\python.exe -m secretary inference-status --probe

# One-shot Ollama checks; these do not start Screenpipe
$env:INFERENCE_PROVIDER = "ollama"
$env:OLLAMA_TEXT_MODEL = "qwen3-vl:2b-instruct-q4_K_M"
.\.venv\Scripts\python.exe -m secretary inference-smoke --text
.\.venv\Scripts\python.exe -m secretary inference-smoke --vision tests\fixtures\landscape.ppm

# Start/reuse real Screenpipe once (requires SCREENPIPE_API_KEY)
$env:SCREENPIPE_API_KEY = Read-Host "Screenpipe API key"
.\.venv\Scripts\python.exe -m secretary run --once

# Long-running managed real observer; reuses an existing instance or starts a
# pinned 0.4.41 child, then owns/stops only that child
.\.venv\Scripts\python.exe -m secretary run

# Explicit test-only fake capture; no real desktop observation
.\.venv\Scripts\python.exe -m secretary run --mock-capture --mock-notifications --once

# Explicit reuse-only mode; never starts Screenpipe
.\.venv\Scripts\python.exe -m secretary run --external

# Real capture/inference/policy with notification delivery suppressed
$env:INFERENCE_PROVIDER = "ollama"
.\.venv\Scripts\python.exe -m secretary run --shadow

# Bounded decision diagnostics from the latest SQLite session
.\.venv\Scripts\python.exe -m secretary session-report
.\.venv\Scripts\python.exe -m secretary recent-decisions --limit 20
.\.venv\Scripts\python.exe -m secretary recent-decisions --action WOULD_NOTIFY --suppressed

# Run the ten privacy-safe deterministic CPU scenarios
.\.venv\Scripts\python.exe -m secretary benchmark
```

Managed mode is the default. The configured command is pinned to
`npx.cmd -y screenpipe@0.4.41 record` and, after checking the 0.4.41
`record --help` output, adds `--disable-audio`,
`--disable-clipboard-capture`, and repeated flags
`--ignored-windows 1Password` and `--ignored-windows KeePass`.
Custom `SECRETARY_EXCLUDED_APPS` values are passed to the same recorder-level
exclusion. The lifecycle manager waits
for health and an authenticated search before real capture begins. If launch or
readiness fails, status is `DEGRADED` and the app does not activate fake capture.
The real Windows Toast provider is also the default for `run`; `--mock-notifications`
is only for tests. `--shadow` uses real capture, the configured inference provider,
real memory/policy/watch state, and a recording-only notification sink. A policy
`NOTIFY` is stored as `WOULD_NOTIFY` in the session report and never reaches a
Windows Toast API.

The tray UI runs its event loop on the main thread. A capture worker polls
Screenpipe and replaces a single latest-state slot; a separate inference worker
owns Ollama, WorkingState, Policy, and SQLite mutation. The capture worker also
performs low-frequency process/readiness supervision (10 seconds by default).
Inference requests carry a generation, creation time, activity snapshot, and
bounded context statistics. Results are classified as FRESH, SLIGHTLY_STALE, or
STALE; only fresh results can escalate policy. A
Secretary-owned child may enter `RESTARTING` and be restarted; an external
Screenpipe instance is never started or killed and is reported as `DEGRADED` if
it stops responding. `--tray` enables the optional real system tray shell.

Inference is now provider-neutral: a bounded `InferenceRequest` is built from
the current event, a short coalesced trajectory, working state, watch
hypotheses, recent failures, and prior decisions. `EventCoalescer` lets one
capture poll become one semantic context, while `InferenceScheduler` keeps one
pending latest request and discards stale work before a slow provider can build
a backlog. The default provider remains deterministic `MockInferenceProvider`.

`OllamaInferenceProvider` is implemented behind an injectable standard-library
HTTP transport for offline contract tests. It targets `/api/chat`, but its
constructor and ordinary `inference-status` command do not contact Ollama; the
explicit `inference-status --probe` command checks `/api/version` and
`/api/tags` without pulling or upgrading anything. `inference-smoke --text` and
`inference-smoke --vision <fixture>` are the only one-shot model calls. The
current `mock` configuration still does not require Ollama.

When Ollama is enabled, capture and inference are separate workers. Capture
continues polling while the inference worker owns ContextBuilder, the
latest-state scheduler, policy, and SQLite mutation. A new state replaces the
single pending request; stale work is discarded. Ollama requests use
`temperature: 0` and the current qwen3-vl smoke confirmed `think: false`.
The provider records only wall/Ollama duration and token-count metadata. Search
results are explicitly sorted by normalized event timestamp; recent trajectory
is chronological and the current event is the latest timestamp, independent of
Screenpipe response order.

Policy thresholds are configurable through the `policy.model` example in
`config\config.example.yaml` or the corresponding `POLICY_*` environment
variables. Defaults are conservative: model WATCH/INVESTIGATE/NOTIFY require
increasing confidence and importance, and model NOTIFY also requires a high
interrupt score plus active WATCH evidence.

The earlier offline preparation intentionally did not access Ollama. The current
integration pass uses only explicit localhost probe/smoke commands; it does not
install, pull, upgrade, or modify Ollama. No real model or GPU is required by
the unit, replay, or default mock runtime paths.

Stop a foreground run with `Ctrl+C`; the process finally block closes only a
Secretary-owned Screenpipe child. To remove the project, stop any run and delete
this `0828` directory; no system service, scheduled task, registry key, or global
package is created by the project. The optional package cache entry mentioned in
[EXTERNAL_CHANGES.md](EXTERNAL_CHANGES.md) can be removed separately if desired.

## Design boundaries

- `ScreenpipeCaptureProvider` is the only module that knows Screenpipe JSON.
- Default runtime capture is real Screenpipe; `MockCaptureProvider` is restricted
  to replay, tests, and explicit `--mock-capture`.
- `MockInferenceProvider` is the current local CPU inference boundary and can be
  replaced by `OllamaInferenceProvider`/llama.cpp later without changing capture.
- `InferenceResult.secretary.candidate_action` is advisory only. Policy Fusion
  combines it with deterministic evidence, working state, and active WATCH
  evidence; `HardRules` still decides the final notification gate. `ASK_CLOUD`
  is recorded as a candidate only while Cloud remains Mock.
- Vision is deterministic and gated by `visual_required`, sparse text, or
  obviously visual application context. Images are resized and encoded in
  memory; missing/corrupt images fail closed without crashing the worker.
- Excluded apps are filtered before extraction, storage, logging, or notification.
- For Secretary-managed Screenpipe 0.4.41, audio and clipboard capture are
  disabled at recorder launch, and configured excluded app names are forwarded
  through `--ignored-windows`. Keyboard and click capture are not disabled by
  this v1 configuration.
- The recorder flags are the strongest boundary verified here. Secretary's
  `PrivacyFilter` is defense in depth: it can prevent excluded events from
  entering Secretary inference, SQLite, logs, or notifications, but it cannot
  claim to prevent Screenpipe itself from capturing a window unless that
  recorder-level exclusion is active. External Screenpipe is not modified.
- Stored events contain a bounded summary and classification, not raw OCR text.
- Policy actions are fixed: `IGNORE`, `REMEMBER`, `WATCH`, `INVESTIGATE`,
  `ASK_CLOUD`, `NOTIFY`.
- Notification rate is capped at two per hour by default, a cooldown is applied,
  and repeated signatures are suppressed. Model WATCH hypotheses are expiring,
  deduplicated, and capped at three active hypotheses by default.
- `session-report` and `recent-decisions` expose only bounded metadata (app,
  event type, candidate/final action, scores, evidence, suppression, and latency).
  They do not print full OCR, prompts, screenshots, passwords, or tokens.
- SQLite is stored under `%LOCALAPPDATA%\AmbientSecretary` for normal installed
  runs, or under the source checkout's `data\` and `logs\` in an explicit project
  configuration. `SECRETARY_DATA_DIR`, `SECRETARY_DB_PATH`, and
  `SECRETARY_LOG_DIR` make the locations explicit. The schema is versioned,
  previous active sessions are marked `ABORTED` after a crash, and bounded
  decision/session retention never deletes semantic memories.

## Tests

Tests are separated into `tests\unit`, `tests\integration`, and `tests\live`.
The portable unit suite requires no Windows desktop, Screenpipe, network, or
GPU. `benchmark` runs the ten CPU scenarios in `scenarios\benchmark.json`.
Integration requires `SCREENPIPE_API_KEY`; managed lifecycle live tests require
`SECRETARY_LIVE_TESTS=1`, the key, an interactive Windows session, and a stopped
Screenpipe runtime. None is enabled by default. Five JSONL scenarios cover ordinary
coding, repeated failures, cross-app context, privacy suppression, and WATCH
expiration.

## Packaging

`installer\README.md` and `.github\workflows\windows.yml` document the optional
PyInstaller/Inno Setup path. System-level installers are intentionally not
installed by this project.

## Final manual checks

On a Windows interactive session, opt in and verify the tray icon, Toast, real
Screenpipe events, Pause/Resume, and both external-versus-managed process
ownership paths. The deferred GPU-only steps are in
[`docs\NVIDIA_WINDOWS_CHECKLIST.md`](docs/NVIDIA_WINDOWS_CHECKLIST.md) and
[`docs\GPU_MODEL_EVALUATION.md`](docs/GPU_MODEL_EVALUATION.md). See
[EXTERNAL_CHANGES.md](EXTERNAL_CHANGES.md) for the external change record.
