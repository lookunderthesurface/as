# External changes

No project-initiated filesystem changes outside
`C:\Users\huwencan\workspace\0828` have been made, except for the disposable
cache/runtime entries described below.

The Screenpipe API is used only through its existing localhost HTTP endpoint. The
API key is supplied ephemerally through the process environment during opt-in
checks and is not stored in the project.

## 2026-08-28

### External change 001

Reason:
Creating the project virtual environment and installing this local package in
editable mode used pip's normal user-level temporary build/cache directories.

Command:
`python -m venv .venv` and `.venv\\Scripts\\python.exe -m pip install -e .[dev]`

Possible external paths:
`C:\\Users\\huwencan\\AppData\\Local\\Temp\\pip-*` and the existing pip cache.

Impact:
Temporary package build metadata/cache only; no global package, system setting,
service, registry key, or Screenpipe data was intentionally changed.

Reversal:
The temporary cache is disposable and can be removed by the user after pip is no
longer using it. The project environment is inside `0828\\.venv`.

### External change 002

Reason:
The user requested an opt-in real managed Screenpipe lifecycle verification. The
test will start the already-installed pinned Screenpipe launcher with audio
disabled, then stop only the child PID owned by the test.

Command:
`.venv\\Scripts\\python.exe -m unittest tests.live.test_windows_live_opt_in -v`
with `SECRETARY_LIVE_TESTS=1` and `SCREENPIPE_API_KEY` supplied ephemerally.

Possible external paths:
The existing npm cache and Screenpipe's existing local data directory while the
runtime records the desktop. No project fixture receives the captured content.

Impact:
Short-lived local Screenpipe process and potentially a small amount of local
screen-capture data; audio remains disabled. No external service upload is made.

Reversal:
The lifecycle test stops its owned child in `finally`; any newly-created
Screenpipe data/cache is external and can be reviewed or removed manually by the
user. The project will not delete it automatically.

Result:
The live managed lifecycle passed. The exact owned process tree was cleaned and
the 3030 listener was absent after the test. The npm cache entry was left intact.
An initial attempt exposed a PATH issue in the npx wrapper; the project now passes
the existing Node directory only to the owned child environment and does not
modify the machine PATH. The follow-up run used the recorder-level audio,
clipboard, and repeated ignored-window flags and confirmed `audio_status` was
disabled through the authenticated runtime health check.

### External change 003

Reason:
The user requested verification of the already-cached Screenpipe 0.4.41 CLI
options before adding recorder-level privacy flags.

Command:
`C:\\Program Files\\nodejs\\npx.cmd --offline --yes screenpipe@0.4.41 record --help`

Possible external paths:
The existing npm cache and npm debug log directory.

Impact:
Help output only; no audio or capture session is started by the command. The
output verified `--disable-audio`, `--disable-clipboard-capture`, and
`--ignored-windows <IGNORED_WINDOWS>` (case-insensitive app/window matching,
with `App::Title` available for scoped matching). It did not require adding
keyboard or click suppression to this v1 scope. A parse-only follow-up with
the flag repeated once per excluded app also exited successfully; the managed
command uses that repeatable form rather than relying on comma splitting.

Reversal:
No project or system reversal is needed. Existing npm cache entries are left
untouched.

### External change 004

Reason:
Prepared the provider-neutral inference pipeline, offline Ollama transport, and
image fixtures requested for the next integration phase.

Impact:
No Ollama process, executable, download, model registry, or port 11434 was
accessed. No package installation, PATH change, or system-level change was made
for this work. All new files and generated fixtures are inside `0828`.

### External change 005

Reason:
The user requested the first explicit local Ollama integration for
`qwen3-vl:2b-instruct-q4_K_M`.

Commands/requests:
`ollama --version` was attempted only to inspect the CLI; it was not available
on the current PowerShell PATH. The explicit Secretary probe queried the local
Ollama `/api/version` and `/api/tags` endpoints. One text smoke, one vision
smoke using `tests/fixtures/landscape.ppm`, and a short managed Screenpipe plus
Ollama run were executed with `--mock-notifications`.

Impact:
The local Ollama runtime reported version `0.33.1` and the configured model was
present. No model was pulled, no runtime was upgraded or configured, and PATH
was not changed. The short Screenpipe run used the existing cached pinned
Screenpipe `0.4.41` launcher, started only a Secretary-owned child, and cleaned
that child on exit. No Cloud API or real notification was used.

Reversal:
No project or system reversal is needed. Runtime processes were stopped by
their respective ownership/cleanup paths; existing npm/Ollama caches were left
untouched.

### External change 006

Reason:
The user requested real-local-AI shadow validation after the Policy Fusion and
decision observability changes.

Commands/requests:
The local Ollama `/api/version` and `/api/tags` probe was repeated, followed by
one text smoke and one vision smoke using the existing
`qwen3-vl:2b-instruct-q4_K_M` model. A short managed Screenpipe plus Ollama
`run --shadow` session was started and stopped explicitly; the Screenpipe API
key was supplied only through the process environment.

Impact:
Ollama reported version `0.33.1` and the configured model was available. Text
and vision structured-output requests completed; no model was downloaded,
pulled, upgraded, or reconfigured. The short shadow session started the
existing cached Screenpipe `0.4.41` child with audio and clipboard capture
disabled and the configured ignored-window flags. Shadow notification delivery
was recording-only; no Windows Toast was invoked.

Reversal:
The Secretary-owned Screenpipe process was stopped during cleanup and the
3030 listener was absent afterward. Existing npm/Ollama caches and local
Screenpipe runtime data were left intact for manual review/removal.
