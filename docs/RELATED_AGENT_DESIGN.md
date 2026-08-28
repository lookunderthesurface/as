# Related Agent Design Research

Research notes distilled from the two reference projects named for the
Persistent Agent Learning Phase. Nothing here is vendored; both projects
remain documentation-only inputs. Reading date: 2026-08.

Sources read:
- thunlp/ProactiveAgent (README, eval/README, paper ICLR 2025 arXiv:2410.12361)
- Letta docs: stateful agents, memory blocks (v1 SDK memory docs)

---

## Part 1 — ProactiveAgent (THUNLP, ICLR 2025)

### What it is
A data-driven pipeline for *proactive* LLM agents: sense user activity,
predict tasks the user may need help with, propose assistance without being
asked, and learn from human judgement. ProactiveBench: 6,790 events across
coding / writing / daily life, plus a reward model (F1 0.918 vs human
annotators) used as an automatic evaluator.

### Answers to our questions

1. **What counts as an intervention opportunity?**
   An event window where the predicted "next task" is worth proposing. The
   agent watches an activity stream (ActivityWatcher: window / web / vscode
   / afk watchers) and the LLM decides both *whether* and *what* to propose.
   Silence is a legitimate output — their best models propose on roughly half
   the opportunities.

2. **How are proposals generated?**
   The LLM receives a bounded activity trace and returns a proposed task +
   notification, or nothing. Proposal generation is a *judgement*, not a
   threshold: thresholds alone cannot distinguish "user stuck" from "user
   busy".

3. **How are Accept / Reject / Ignore represented?**
   Three explicit user actions on the toast:
   - **Accept** — user engages with the proposal.
   - **Reject** — user explicitly declines.
   - **Ignore** — user does nothing; toast auto-dismisses after an interval.
   Their doc is explicit about Ignore semantics: *"doing nothing will make
   the agent know that you are busy and ignored the proposal; the agent will
   try to make less proposals in the following turns."*

4. **How does feedback affect later behavior?**
   Two ways: (a) at runtime, accepted/rejected/ignored feedback modulates the
   proposal frequency of subsequent turns (a dynamic feedback loop inside
   their GYM environment); (b) offline, labelled data trains both a
   fine-tuned proactive agent and a reward model that grades any agent.

5. **How are False Alarm / Missed Need evaluated?**
   TP / FP / TN / FN against a ground-truth judge:
   - TP: agent proposed, judge accepts.
   - FP: agent proposed, judge rejects (False Alarm).
   - FN: agent stayed silent, judge would have accepted (Missed-Need, MN).
   - TN: agent stayed silent, judge agrees silence was right (Non-Response).
   Reported: Precision / Recall / Accuracy / False-Alarm / F1. Notably even
   GPT-4o achieves ~48% precision / ~52% false-alarm; their fine-tuned
   7B model reaches ~50% precision / 66.5% F1. **Proactiveness is hard, and
   the false-alarm rate of naive implementations is enormous** — direct
   empirical support for our "宁可漏提醒，不要多提醒" priority.

6. **What is research/demo-specific and must NOT be copied?**
   - ActivityWatcher as a sensor (we have Screenpipe; installing a second
     observer is forbidden).
   - Their GYM simulation environment and data-generation pipeline.
   - The reward model (an LLM judge) — we do not download it and it must not
     become a runtime dependency; our Critic is deterministic first.
   - The toast/dialogue UX flow and fine-tuning prompts.

### What we take
- The TP/FP/TN/FN evaluation matrix and precision-first mindset (already
  implemented in `secretary/evaluation`).
- Explicit Accept / Reject / **Ignore** semantics where Ignore is *weak*
  evidence (busy ≠ bad suggestion). Implemented as `UserReaction` with
  `is_explicit` / `is_weak` and `label_weight`.
- The feedback-modulates-future-proposals loop: our version is durable
  (preferences + knowledge in SQLite) rather than in-session model memory.

---

## Part 2 — Letta (MemGPT lineage)

### What it is
A stateful-agent platform: all state (memory blocks, messages, tool calls)
persists in a database; the agent self-edits always-in-context memory blocks
via tools; archival memory is out-of-context and search-retrieved;
sleeptime agents reorganize memory in the background.

### Answers to our questions

1. **What information stays always in context?**
   *Memory blocks*: small structured strings (label / description / value /
   char `limit`) prepended to the system prompt, e.g. `persona` and `human`
   blocks. Blocks are *always visible — no retrieval needed*. Each block
   enforces a character budget (`chars_current` vs `chars_limit` is shown to
   the model so it can manage its own budget). Blocks can be read-only or
   shared between agents.

2. **What stays outside context?**
   Everything else: full message history (recall), archival passages, tool
   outputs. Old messages survive eviction because *everything* is persisted;
   the context window only ever holds a bounded projection.

3. **How is durable memory different from raw history?**
   Raw history is append-only and never interpreted. Core memory is a
   *curated, rewritten abstraction* the agent maintains about the user and
   itself; archival memory is searchable raw-ish text. The context hierarchy
   (core in-context / recall+archival out-of-context) keeps prompts bounded
   as usage grows.

4. **How does dreaming consolidate recent experience?**
   The *sleeptime agent* runs between interactions: it receives the recent
   conversation, reasons about what is durable, and rewrites memory blocks
   with memory tools (reorganize, update, split, archive). Key properties:
   it is *deferred* (never blocks the interactive turn), *auditable* (block
   edits are visible tool calls), and *bounded*.

5. **How does memory doctor prevent drift/bloat?**
   Blocks carry hard character limits; the platform surfaces usage; memory
   reorganization (manual or sleeptime) merges/dedupes/retires stale blocks.
   The inspectability principle: every memory mutation is an explicit,
   reviewable operation — never a silent embedding-side drift.

6. **How does the system preserve inspectability / version history?**
   All state is DB rows; AgentFile (.af) export serializes an agent; block
   values are plain strings a human can read and diff. Nothing important is
   hidden in opaque vectors (their vector archival is an *additional*
   recall index, not the canonical store).

7. **What does NOT fit a high-frequency GUI continuing agent?**
   - The agent rewriting its own core memory via LLM tool calls *per
     interaction* — at our frame rates that is a GPU call per frame; our
     profile updates are event-driven (feedback / consolidation), never
     per-frame.
   - Letta Server, agent identity/conversations, MCP tool ecosystem —
     different product shape.
   - Vector archival memory as a second store — we keep SQLite canonical
     with deterministic retrieval; no vector DB this phase.
   - Unbounded scratchpad blocks — our core memory is hard-capped (800
     chars) and checked by `memory-doctor`.

### What we take
- The **tiered memory hierarchy with a tiny always-in-context core**
  (implemented: `MemoryTier.CORE` with an 800-char budget + `get_memory_context`).
- **Deferred, bounded, auditable consolidation** ("dreaming") triggered by
  session end / explicit command — implemented as `MemoryConsolidator`
  (deterministic v1; LLM variant lands behind a validator this phase).
- **Memory hygiene tooling** — `memory-doctor` read-only dry run.
- **Source attribution on every durable fact** (confidence + source +
  source_episode_ids) so "why do you remember this?" is always answerable.
- This phase adds what Letta calls *external knowledge discipline*:
  `status` lifecycle (ACTIVE / SUPERSEDED / DISPROVED / STALE) so knowledge
  can be falsified instead of accumulating contradictions.

---

## Part 3 — Mapping into Ambient Secretary

| Reference idea | Ambient Secretary implementation |
|---|---|
| Activity trace sensing | Screenpipe (already), GUI-first perception |
| Proposal generation | WATCH accumulator + InterventionCritic + ProactivePolicy |
| Accept/Reject/Ignore | `UserReaction` explicit/weak weighting |
| Feedback modulates future proposals | durable preferences + knowledge gates in policy |
| TP/FP/TN/FN + False-Alarm | `secretary/evaluation` matrix + label CLI |
| Reward-model judge | NOT copied; `InterventionCritic` is deterministic; future local critic only with real feedback data |
| Core memory blocks | `MemoryTier.CORE`, always loaded, 800-char budget |
| Recall / archival | episodic + semantic tiers in SQLite |
| Sleeptime dreaming | `MemoryConsolidator` (session end / idle / explicit), LLM variant behind validator |
| Memory doctor / reorganization | `memory-doctor` CLI (read-only) |
| AgentFile export | not needed yet; SQLite rows are already inspectable |

### Division of labor (never confused)
- **ProactiveAgent** taught us *how to evaluate and learn when to speak*.
- **Letta** taught us *how a long-lived agent should organize memory*.
- **Ambient Secretary** itself owns *continuously perceiving and
  understanding the real desktop world*.
