"""Layer 4: Language models with graceful degradation (NFR2).

The reasoning chain tries, in order:

1. a cloud large language model (any OpenAI-compatible API), only when the
   dashboard's connectivity switch is ONLINE and CLOUD_LLM_URL/KEY are set;
2. a local small language model served by Ollama (which runs on llama.cpp),
   for example `ollama pull llama3.2:3b`;
3. the built-in deterministic reasoning, which always works.

So the system never stops reasoning when the connection drops; it only falls
back to a smaller engine.
"""
from __future__ import annotations

import threading
import time

import requests

from ..config import SETTINGS


class _OllamaProbe:
    """Checks every 30 s, in a background thread, whether the local model is installed.

    The check is kept off the detection path: a failed connection can take about
    two seconds on Windows, and that wait must not be counted as detection latency.
    One probe is shared by every ReasoningEngine in the process.
    """
    _lock = threading.Lock()
    _started = False
    ok = False

    @classmethod
    def start(cls) -> None:
        with cls._lock:
            if cls._started:
                return
            cls._started = True
        threading.Thread(target=cls._loop, daemon=True, name="ollama-probe").start()

    @classmethod
    def _loop(cls) -> None:
        while True:
            try:
                r = requests.get(f"{SETTINGS.ollama_url}/api/tags", timeout=2)
                names = [m.get("name", "") for m in r.json().get("models", [])]
                cls.ok = any(n.split(":")[0] == SETTINGS.ollama_model.split(":")[0] for n in names)
            except Exception:
                cls.ok = False
            time.sleep(30)


class ReasoningEngine:
    def __init__(self):
        self.online = True               # toggled from the dashboard to simulate losing the uplink
        self._ollama_ok = None
        self._ollama_checked = 0.0
        self.calls = {"cloud": 0, "local_slm": 0, "built_in": 0}
        self.last_engine = "built_in"

    # ------------------------------------------------------------------ availability
    def cloud_available(self) -> bool:
        return self.online and bool(SETTINGS.cloud_url and SETTINGS.cloud_key and SETTINGS.cloud_model)

    def ollama_available(self) -> bool:
        _OllamaProbe.start()
        return _OllamaProbe.ok

    def active_engine(self) -> str:
        if self.cloud_available():
            return f"Cloud LLM ({SETTINGS.cloud_model})"
        if self.ollama_available():
            return f"Local SLM ({SETTINGS.ollama_model} via Ollama)"
        return "Built-in reasoning (no language model installed)"

    # ------------------------------------------------------------------ completion
    def complete(self, prompt: str, system: str = "", max_tokens: int = 200) -> tuple[str | None, str]:
        """Returns (text, engine). text is None when only built-in reasoning is available."""
        if self.cloud_available():
            try:
                r = requests.post(
                    f"{SETTINGS.cloud_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {SETTINGS.cloud_key}"},
                    json={"model": SETTINGS.cloud_model, "max_tokens": max_tokens, "temperature": 0,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]},
                    timeout=20)
                r.raise_for_status()
                self.calls["cloud"] += 1
                self.last_engine = "cloud"
                return r.json()["choices"][0]["message"]["content"], "cloud"
            except Exception as exc:
                print(f"[llm] cloud call failed ({exc}); trying local model")
        if self.ollama_available():
            try:
                r = requests.post(f"{SETTINGS.ollama_url}/api/generate",
                                  json={"model": SETTINGS.ollama_model, "system": system, "prompt": prompt,
                                        "stream": False, "options": {"temperature": 0, "num_predict": max_tokens}},
                                  timeout=60)
                r.raise_for_status()
                self.calls["local_slm"] += 1
                self.last_engine = "local_slm"
                return r.json().get("response", ""), "local_slm"
            except Exception as exc:
                print(f"[llm] local model call failed ({exc}); using built-in reasoning")
                _OllamaProbe.ok = False
        self.calls["built_in"] += 1
        self.last_engine = "built_in"
        return None, "built_in"
