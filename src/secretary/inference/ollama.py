from __future__ import annotations

import json
import re
import socket
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .image import ImagePreprocessor
from .metrics import InferenceMetrics
from .schema import InferenceRequest, InferenceResult, RESPONSE_SCHEMA, validate_inference_result
from .status import InferenceRuntimeState, LocalInferenceStatus


def _stdlib_post(url: str, payload: Mapping[str, object], timeout: float) -> object:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _stdlib_get(url: str, timeout: float) -> object:
    request = Request(url, method="GET")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True)
class OllamaProbeResult:
    status: InferenceRuntimeState
    version: str | None
    configured_model: str | None
    model_available: bool
    error_type: str | None = None
    detail: str | None = None


class OllamaInferenceProvider:
    """Offline-testable Ollama ``/api/chat`` adapter.

    Construction never probes Ollama. The default HTTP transport is invoked only
    when ``analyze`` is explicitly called by a future Ollama configuration.
    """

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        text_model: str | None = None,
        vision_model: str | None = None,
        timeout_seconds: float = 120.0,
        keep_alive: str = "30m",
        temperature: float = 0.0,
        think: bool | None = False,
        http_post: Callable[..., object] | None = None,
        http_get: Callable[..., object] | None = None,
        image_preprocessor: ImagePreprocessor | None = None,
        system_prompt: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint or f"{self.base_url}/api/chat"
        self.model = text_model
        self.text_model = text_model
        self.vision_model = vision_model or text_model
        self.timeout_seconds = max(1.0, timeout_seconds)
        self.keep_alive = keep_alive
        try:
            self.temperature = min(2.0, max(0.0, float(temperature)))
        except (TypeError, ValueError):
            self.temperature = 0.0
        self.think = think
        self.http_post = http_post or _stdlib_post
        self.http_get = http_get or _stdlib_get
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()
        self.system_prompt = system_prompt or self._load_system_prompt()
        self._status = LocalInferenceStatus(
            provider=self.name,
            status=InferenceRuntimeState.NOT_CHECKED,
            model=self.model,
            real_model_required=True,
        )

    def analyze(self, request: InferenceRequest) -> InferenceResult:
        started = time.monotonic()
        self._status = LocalInferenceStatus(
            provider=self.name,
            status=InferenceRuntimeState.BUSY,
            model=self.vision_model if request.use_vision else self.text_model,
            last_mode="vision" if request.use_vision else "text",
            real_model_required=True,
        )
        model = self.vision_model if request.use_vision else self.text_model
        if not model:
            return self._failure("model_not_configured", started, request, model, ValueError("model is not configured"))
        payload = self._payload(request, model)
        try:
            response = self._call_transport(payload)
            parsed = self._extract_response(response)
            metrics = self._metrics(response, started, request.use_vision)
            result = replace(
                validate_inference_result(parsed, provider=self.name, model=model),
                metrics=metrics,
            )
            if result.error_type:
                self._set_failure(result.error_type, started, request, model, metrics)
            else:
                self._set_success(started, request, model, metrics)
            return result
        except HTTPError as exc:
            exc.close()
            return self._failure("http_error", started, request, model, exc)
        except (TimeoutError, socket.timeout) as exc:
            return self._failure("timeout", started, request, model, exc)
        except URLError as exc:
            error_type = "timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "connection_error"
            return self._failure(error_type, started, request, model, exc)
        except (ConnectionError, ConnectionRefusedError, OSError) as exc:
            return self._failure("connection_error", started, request, model, exc)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return self._failure("malformed_response", started, request, model, exc)

    def status(self) -> LocalInferenceStatus:
        return self._status

    def probe(self) -> OllamaProbeResult:
        """Explicitly inspect Ollama; never called by construction or normal status."""
        try:
            version_response = self._call_get(f"{self.base_url}/api/version")
            version = self._read_version(version_response)
        except Exception as exc:
            return self._probe_failure("connection_error", f"Ollama runtime unavailable: {exc.__class__.__name__}")
        if not version:
            return self._probe_failure("malformed_response", "Ollama runtime version was not returned")

        try:
            tags_response = self._call_get(f"{self.base_url}/api/tags")
            available_models = self._read_model_names(tags_response)
        except Exception as exc:
            return OllamaProbeResult(
                status=InferenceRuntimeState.DEGRADED,
                version=version,
                configured_model=self.text_model,
                model_available=False,
                error_type="model_list_error",
                detail=f"Ollama model list unavailable: {exc.__class__.__name__}",
            )

        configured_models = tuple(model for model in (self.text_model, self.vision_model) if model)
        model_available = bool(configured_models) and all(model in available_models for model in configured_models)
        if not model_available:
            result = OllamaProbeResult(
                status=InferenceRuntimeState.DEGRADED,
                version=version,
                configured_model=self.text_model,
                model_available=False,
                error_type="model_not_found",
                detail="configured Ollama model is not present in /api/tags",
            )
            self._set_probe_status(result)
            return result

        required = self._required_version(configured_models)
        parsed_version = self._version_tuple(version)
        if required is not None and (parsed_version is None or parsed_version < required):
            result = OllamaProbeResult(
                status=InferenceRuntimeState.DEGRADED,
                version=version,
                configured_model=self.text_model,
                model_available=True,
                error_type="incompatible_runtime",
                detail=f"Ollama runtime incompatible; qwen3-vl requires >= {'.'.join(map(str, required))}",
            )
            self._set_probe_status(result)
            return result

        result = OllamaProbeResult(
            status=InferenceRuntimeState.READY,
            version=version,
            configured_model=self.text_model,
            model_available=True,
        )
        self._set_probe_status(result)
        return result

    def _payload(self, request: InferenceRequest, model: str) -> dict[str, object]:
        message: dict[str, object] = {
            "role": "user",
            "content": request.context_text or self._fallback_context(request),
        }
        if request.use_vision and request.image_path:
            prepared = self.image_preprocessor.prepare_image(request.image_path)
            if prepared is not None:
                message["images"] = [prepared.data]
        return {
            "model": model,
            "messages": [{"role": "system", "content": self.system_prompt}, message],
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
            **({"think": self.think} if self.think is not None else {}),
        }

    def _call_transport(self, payload: Mapping[str, object]) -> object:
        try:
            return self.http_post(self.endpoint, payload, self.timeout_seconds)
        except TypeError as first_error:
            # Small two-argument fakes are convenient for contract tests.
            try:
                return self.http_post(self.endpoint, payload)
            except TypeError:
                raise first_error

    def _call_get(self, url: str) -> object:
        try:
            return self.http_get(url, self.timeout_seconds)
        except TypeError as first_error:
            try:
                return self.http_get(url)
            except TypeError:
                raise first_error

    @staticmethod
    def _extract_response(response: object) -> object:
        if not isinstance(response, Mapping):
            raise ValueError("response is not an object")
        message = response.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str):
                return json.loads(content)
            if isinstance(content, Mapping):
                return content
            raise ValueError("missing message content")
        if "event" in response or "secretary" in response:
            return response
        raise ValueError("missing message content")

    def _set_success(self, started: float, request: InferenceRequest, model: str, metrics: InferenceMetrics) -> None:
        self._status = LocalInferenceStatus(
            provider=self.name,
            status=InferenceRuntimeState.READY,
            model=model,
            last_mode="vision" if request.use_vision else "text",
            last_latency_ms=(time.monotonic() - started) * 1000,
            last_success_at=datetime.now(timezone.utc),
            real_model_required=True,
            last_metrics=metrics,
        )

    def _set_failure(self, error_type: str, started: float, request: InferenceRequest, model: str | None, metrics: InferenceMetrics | None = None) -> None:
        self._status = LocalInferenceStatus(
            provider=self.name,
            status=InferenceRuntimeState.DEGRADED,
            model=model,
            last_mode="vision" if request.use_vision else "text",
            last_latency_ms=(time.monotonic() - started) * 1000,
            last_error_type=error_type,
            real_model_required=True,
            last_metrics=metrics or self._wall_metrics(started, request.use_vision),
        )

    def _failure(self, error_type: str, started: float, request: InferenceRequest, model: str | None, error: Exception) -> InferenceResult:
        metrics = self._wall_metrics(started, request.use_vision)
        self._set_failure(error_type, started, request, model, metrics)
        return replace(InferenceResult.safe(error_type, provider=self.name, model=model), metrics=metrics)

    @staticmethod
    def _wall_metrics(started: float, use_vision: bool) -> InferenceMetrics:
        return InferenceMetrics(
            wall_latency_ms=(time.monotonic() - started) * 1000,
            mode="vision" if use_vision else "text",
        )

    @classmethod
    def _metrics(cls, response: object, started: float, use_vision: bool) -> InferenceMetrics:
        values = response if isinstance(response, Mapping) else {}
        return InferenceMetrics(
            wall_latency_ms=(time.monotonic() - started) * 1000,
            ollama_total_duration_ms=cls._duration_ms(values.get("total_duration")),
            model_load_ms=cls._duration_ms(values.get("load_duration")),
            prompt_tokens=cls._integer(values.get("prompt_eval_count")),
            output_tokens=cls._integer(values.get("eval_count")),
            prompt_eval_ms=cls._duration_ms(values.get("prompt_eval_duration")),
            generation_ms=cls._duration_ms(values.get("eval_duration")),
            mode="vision" if use_vision else "text",
        )

    @staticmethod
    def _duration_ms(value: object) -> float | None:
        try:
            return float(value) / 1_000_000 if value is not None else None
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _integer(value: object) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError, OverflowError):
            return None

    def _set_probe_status(self, result: OllamaProbeResult) -> None:
        self._status = LocalInferenceStatus(
            provider=self.name,
            status=result.status,
            model=self.text_model,
            last_error_type=result.error_type,
            real_model_required=True,
        )

    def _probe_failure(self, error_type: str, detail: str) -> OllamaProbeResult:
        result = OllamaProbeResult(
            status=InferenceRuntimeState.DEGRADED,
            version=None,
            configured_model=self.text_model,
            model_available=False,
            error_type=error_type,
            detail=detail,
        )
        self._set_probe_status(result)
        return result

    @staticmethod
    def _read_version(response: object) -> str | None:
        if not isinstance(response, Mapping):
            return None
        value = response.get("version")
        return str(value).strip() if value else None

    @staticmethod
    def _read_model_names(response: object) -> set[str]:
        if not isinstance(response, Mapping):
            raise ValueError("model list is not an object")
        models = response.get("models")
        if not isinstance(models, list):
            raise ValueError("model list is missing")
        names: set[str] = set()
        for item in models:
            if isinstance(item, Mapping):
                name = item.get("name") or item.get("model")
                if name:
                    names.add(str(name))
        return names

    @staticmethod
    def _version_tuple(value: str) -> tuple[int, int, int] | None:
        match = re.search(r"(?:^|[^0-9])v?(\d+)\.(\d+)(?:\.(\d+))?", value)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)

    @classmethod
    def _required_version(cls, models: tuple[str, ...]) -> tuple[int, int, int] | None:
        if any(model.casefold().startswith("qwen3-vl") for model in models):
            return (0, 12, 7)
        return None

    @staticmethod
    def _fallback_context(request: InferenceRequest) -> str:
        event = request.current_event
        return f"CURRENT EVENT\napp={event.foreground_app}; window={event.window_title}; text={event.text[:2000]}"

    @staticmethod
    def _load_system_prompt() -> str:
        path = Path(__file__).resolve().parents[3] / "prompts" / "local_secretary_system.txt"
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return "You are an ambient desktop secretary. Prefer silence over unnecessary interruption."
