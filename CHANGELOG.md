# Changelog

## 0.1.0 - 2026-08-28

- Added the privacy-first observation pipeline, deterministic policy, WATCH state,
  SQLite memory, replay scenarios, preflight checks, and authenticated Screenpipe adapter.
- Added safe optional Windows notification, tray, and process ownership modules.
- Corrected runtime defaults to real Screenpipe with managed lifecycle; MockCapture
  is now explicit-only, while MockInference remains the CPU development boundary.
- Added explicit `DEGRADED` capture state, ready/auth probing, scoped Node PATH
  injection, descendant cleanup, and real managed lifecycle coverage.
