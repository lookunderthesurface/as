from __future__ import annotations

from ..events.schema import NormalizedEvent
from .schema import Action, InferenceEvent, InferenceRequest, InferenceResult, SecretaryAssessment
from .status import InferenceRuntimeState, LocalInferenceStatus


class MockInferenceProvider:
    """Deterministic CPU substitute for a future local VLM backend."""

    name = "mock"
    model = None
    ACTION_SPACE = tuple(action.value for action in Action)
    _failure_terms = (
        ("traceback", "traceback"),
        ("exception", "exception"),
        ("error", "error"),
        ("failed", "test-failure"),
        ("failure", "test-failure"),
        ("npm err", "npm-failure"),
        ("command not found", "command-failure"),
        ("cannot find module", "module-failure"),
    )

    def analyze(self, request: InferenceRequest | NormalizedEvent) -> InferenceResult:
        if isinstance(request, NormalizedEvent):
            request = InferenceRequest(current_event=request)
        event = request.current_event
        app = event.foreground_app
        app_lower = app.casefold()
        text_lower = event.text.casefold()
        haystack = f"{app_lower} {event.window_title.casefold()} {text_lower}"
        signature = self._failure_signature(haystack)
        if signature:
            activity = "terminal" if any(token in haystack for token in ("terminal", "powershell", "cmd.exe", "bash", "shell")) else "coding"
            return self._result(
                event_type="failure",
                activity=activity,
                summary="A repeatable failure signal was observed",
                importance=0.88,
                novelty=0.75,
                confidence=0.94,
                failure_signature=signature,
                topic=signature,
                candidate_action=Action.WATCH,
                reason="Failure evidence should be evaluated by deterministic policy",
            )
        if self._is_documentation(haystack):
            return self._result(
                event_type="documentation",
                activity="research",
                summary="Documentation or search activity was observed",
                importance=0.28,
                novelty=0.35,
                confidence=0.90,
                topic=self._topic(haystack),
            )
        if any(token in haystack for token in ("visual studio", "vscode", "code.exe", "jetbrains", "editor")):
            return self._result(
                event_type="coding",
                activity="editor",
                summary="Code editing activity was observed",
                importance=0.20,
                novelty=0.20,
                confidence=0.93,
            )
        if any(token in haystack for token in ("terminal", "powershell", "cmd.exe", "bash")):
            return self._result(
                event_type="terminal",
                activity="terminal",
                summary="Terminal activity was observed",
                importance=0.20,
                novelty=0.15,
                confidence=0.93,
            )
        if event.event_source.casefold() in {"app_switch", "window_focus"}:
            return self._result(
                event_type="app_switch",
                activity="navigation",
                summary="The active application changed",
                importance=0.12,
                novelty=0.20,
                confidence=0.96,
            )
        return self._result(
            event_type="activity",
            activity="desktop",
            summary="Work activity was observed",
            importance=0.10,
            novelty=0.10,
            confidence=0.80,
        )

    def status(self) -> LocalInferenceStatus:
        return LocalInferenceStatus(
            provider=self.name,
            status=InferenceRuntimeState.MOCK,
            real_model_required=False,
        )

    def classify(self, prompt: str) -> dict[str, object]:
        """Compatibility helper for callers that still pass a text prompt."""
        return {"action": Action.IGNORE.value, "confidence": 0.5, "source": "deterministic-mock"}

    def _result(self, *, event_type: str, activity: str, summary: str, importance: float, novelty: float, confidence: float, failure_signature: str | None = None, topic: str | None = None, candidate_action: Action = Action.IGNORE, reason: str = "No high-value intervention is indicated") -> InferenceResult:
        return InferenceResult(
            event=InferenceEvent(
                event_type=event_type,
                activity=activity,
                summary=summary,
                importance=importance,
                novelty=novelty,
                confidence=confidence,
                failure_signature=failure_signature,
                topic=topic,
            ),
            secretary=SecretaryAssessment(candidate_action=candidate_action, reason=reason),
            provider=self.name,
        )

    def _failure_signature(self, haystack: str) -> str | None:
        kind = next((value for term, value in self._failure_terms if term in haystack), None)
        if kind is None:
            return None
        ecosystem = "python" if any(term in haystack for term in ("pytest", "python", "traceback")) else "node" if any(term in haystack for term in ("npm", "node", "javascript")) else "general"
        return f"{kind}:{ecosystem}"

    @staticmethod
    def _is_documentation(haystack: str) -> bool:
        return any(token in haystack for token in ("chrome", "edge", "firefox", "documentation", "docs", "stackoverflow", "github"))

    @staticmethod
    def _topic(haystack: str) -> str:
        for token in ("python", "pytest", "npm", "typescript", "javascript", "rust", "git"):
            if token in haystack:
                return token
        return "general"
