"""Central settings for the AFDDT.

Every value can be changed with an environment variable, so nothing in the
code has to be edited to run the system on a different machine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TI_DIR = DATA_DIR / "ti"
DB_PATH = DATA_DIR / "afddt.db"
RESULTS_DIR = ROOT / "results"
STATIC_DIR = Path(__file__).resolve().parent / "layer8_application" / "static"


@dataclass
class Settings:
    # Simulation of the physical deployment (Layer 1)
    tick_seconds: float = float(os.getenv("AFDDT_TICK", "1.0"))
    device_classes: tuple = ("smart_meter", "env_sensor", "ip_camera")
    devices_per_class: int = int(os.getenv("AFDDT_DEVICES_PER_CLASS", "4"))
    edge_nodes: int = 2
    warmup_samples: int = 400          # benign samples used to train each model
    seed: int = int(os.getenv("AFDDT_SEED", "7"))

    # Agentic layer
    fusion_window_ticks: int = 25      # how far back the fusion agent looks
    gateway_hold_ticks: int = 4        # gateway findings wait this long for corroboration
    finding_cooldown_ticks: int = 20   # an agent will not re-report the same twin inside this window

    # Reasoning engines (Layer 4 / 6.4).  All optional: without them the
    # system uses its built-in deterministic reasoning.
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    cloud_url: str = os.getenv("CLOUD_LLM_URL", "")      # OpenAI-compatible base URL, e.g. https://api.openai.com/v1
    cloud_key: str = os.getenv("CLOUD_LLM_KEY", "")
    cloud_model: str = os.getenv("CLOUD_LLM_MODEL", "")

    # Embedding model for enrichment: "tfidf" (default, no download) or "st" (sentence-transformers)
    embedder: str = os.getenv("AFDDT_EMBEDDER", "tfidf")
    st_model: str = os.getenv("AFDDT_ST_MODEL", "sentence-transformers/multi-qa-mpnet-base-dot-v1")

    # Connection services
    aiotndt_url: str = os.getenv("AIOTNDT_URL", "")      # cloud network twin endpoint (optional)
    mqtt_host: str = os.getenv("MQTT_HOST", "")          # set to use a real MQTT broker such as Mosquitto
    mqtt_port: int = int(os.getenv("MQTT_PORT", "1883"))

    # Dashboard
    host: str = os.getenv("AFDDT_HOST", "127.0.0.1")
    port: int = int(os.getenv("AFDDT_PORT", "8000"))


SETTINGS = Settings()
