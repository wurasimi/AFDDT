"""Layer 8: Application layer.

A FastAPI web server that runs the AFDDT in the background and serves the
operator dashboard at http://127.0.0.1:8000.  The dashboard calls the JSON
endpoints below; you can also open http://127.0.0.1:8000/docs to try each
endpoint by hand.
"""
from __future__ import annotations

import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import STATIC_DIR
from ..engine import Engine


class Ask(BaseModel):
    question: str


class Feedback(BaseModel):
    kind: str                       # "false_positive" or "confirm"
    technique: str | None = None


class Connectivity(BaseModel):
    online: bool


def create_app(engine: Engine | None = None) -> FastAPI:
    eng = engine or Engine()
    app = FastAPI(title="AFDDT operator API", version="1.0")
    app.state.engine = eng
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.on_event("startup")
    def _start():
        threading.Thread(target=eng.run_forever, daemon=True, name="afddt-loop").start()

    @app.on_event("shutdown")
    def _stop():
        eng.running = False

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/state")
    def state():
        return eng.snapshot()

    @app.get("/api/finding/{finding_id}")
    def finding(finding_id: str):
        f = next((x for x in eng.findings if x.finding_id == finding_id), None)
        if f is None:
            raise HTTPException(404, "finding not found")
        with eng.lock:
            d = eng.finding_dict(f, full=True)
            d["members"] = [eng.finding_dict(m) for m in f.members]
            return d

    @app.post("/api/scenario/{name}")
    def scenario(name: str, target: str | None = None):
        try:
            return eng.start_scenario(name, target)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/actions/{action_id}/approve")
    def approve(action_id: str):
        return eng.approve(action_id)

    @app.post("/api/actions/{action_id}/reject")
    def reject(action_id: str):
        return eng.reject(action_id)

    @app.post("/api/findings/{finding_id}/feedback")
    def feedback(finding_id: str, body: Feedback):
        return eng.feedback(finding_id, body.kind, body.technique)

    @app.post("/api/connectivity")
    def connectivity(body: Connectivity):
        return eng.set_connectivity(body.online)

    @app.post("/api/ask")
    def ask(body: Ask):
        return eng.ask(body.question)

    @app.post("/api/reset")
    def reset():
        return eng.reset()

    @app.post("/api/ti/reload")
    def reload_ti():
        with eng.lock:
            n = eng.svc.connection.reload_threat_intelligence()
        return {"objects": n}

    return app
