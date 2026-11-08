"""The live server: the web app in `site/` plus a JSON API over the agent, for a demo with the model running.

    PPROF_AGENT_ENDPOINT=http://localhost:8000/v1 PPROF_AGENT_MODEL=qwen3-8b-awq uvicorn server:app --port 8765

Without an endpoint the agent still runs (numbered replies, the harness's own plan). A session is one `Agent`
in memory; a message is answered in a background thread (a planned analysis takes minutes) while the client
polls the session for the transcript, the planner's log lines and the bundle of what exists so far.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from pprof_agent import actions, export  # noqa: E402
from pprof_agent.chat import Agent  # noqa: E402
from pprof_agent.engine import executor  # noqa: E402

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
EXAMPLES = Path(os.environ.get("PPROF_AGENT_EXAMPLES", ROOT / "examples"))
OUT = Path(os.environ.get("PPROF_AGENT_OUT", ROOT / "runs" / "live"))
ENDPOINT = os.environ.get("PPROF_AGENT_ENDPOINT")
MODEL = os.environ.get("PPROF_AGENT_MODEL")

app = FastAPI(title="pprof Agent")
SESSIONS: Dict[str, "Live"] = {}


def _client():
    if os.environ.get("PPROF_AGENT_FAKE"):          # the deterministic fake model of the tests: the whole flow without a GPU
        from pprof_agent.fake import FakeClient
        return FakeClient(), "fake"
    if not ENDPOINT or not MODEL:
        return None, None
    from openai import OpenAI
    return OpenAI(base_url=ENDPOINT, api_key=os.environ.get("OPENAI_API_KEY", "none")), MODEL


class Live:
    def __init__(self, sid: str, data_path: str, dataset_dir: Optional[Path]):
        client, model = _client()
        self.id = sid
        self.dataset_dir = dataset_dir
        self.log: List[str] = []
        self.busy = False
        self.error: Optional[str] = None
        self.lock = threading.Lock()
        self.agent = Agent(data_path, str(OUT), client=client, model=model, name=sid, log=self._log)
        self.started = time.time()

    def _log(self, line: str) -> None:
        self.log.append(f"{time.strftime('%H:%M:%S')} {line}")

    def start(self) -> str:
        return self.agent.start()

    def submit_async(self, text: str) -> None:
        if self.busy:
            raise HTTPException(409, "the session is still answering the previous message")
        self.busy, self.error = True, None

        def run():
            try:
                self.agent.submit(text)
            except Exception as exc:  # noqa: BLE001
                self.error = f"{type(exc).__name__}: {exc}"
                self._log(f"error: {self.error}")
            finally:
                try:
                    self.agent.session.save("trace.json")
                except Exception:  # noqa: BLE001
                    pass
                self.busy = False
        threading.Thread(target=run, daemon=True).start()

    def status(self) -> Dict[str, Any]:
        s = self.agent.session
        return {"id": self.id, "busy": self.busy, "error": self.error, "phase": s.phase, "steps": s.steps, "results": len(s.results),
                "model_calls": len(s.model_calls), "transcript": s.transcript, "log": self.log[-200:], "run_dir": s.run_dir,
                "with_model": self.agent.has_model, "elapsed": round(time.time() - self.started)}

    def bundle(self) -> Dict[str, Any]:
        with self.lock:
            return export.bundle_of(self.agent.session, str(self.dataset_dir) if self.dataset_dir else None)


class Message(BaseModel):
    text: str


class NewSession(BaseModel):
    dataset: Optional[str] = None


@app.get("/api/datasets")
def datasets() -> Dict[str, Any]:
    out = []
    for d in sorted(EXAMPLES.iterdir()) if EXAMPLES.exists() else []:
        if (d / "data.csv").exists():
            scen = {p.stem: p.read_text(encoding="utf-8") for p in sorted((d / "scenarios").glob("*.txt"))} if (d / "scenarios").exists() else {}
            spec = (d / "specification.json").read_text(encoding="utf-8") if (d / "specification.json").exists() else None
            out.append({"name": d.name, "data_file": "data.csv", "scenarios": scen, "published_specification": spec,
                        "dictionary": (d / "dictionary.md").read_text(encoding="utf-8") if (d / "dictionary.md").exists() else None})
    return {"datasets": out, "model": "fake" if os.environ.get("PPROF_AGENT_FAKE") else (MODEL if ENDPOINT else None), "mode": "live"}


@app.post("/api/sessions")
def new_session(req: NewSession) -> Dict[str, Any]:
    if not req.dataset:
        raise HTTPException(400, "name a data set, or upload a file")
    d = EXAMPLES / req.dataset
    if not (d / "data.csv").exists():
        raise HTTPException(404, f"no data set {req.dataset}")
    sid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    live = Live(sid, str(d / "data.csv"), d)
    SESSIONS[sid] = live
    opening = live.start()
    return {"id": sid, "opening": opening, "status": live.status()}


@app.post("/api/sessions/upload")
async def upload_session(file: UploadFile = File(...)) -> Dict[str, Any]:
    sid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    d = OUT / "uploads" / sid
    d.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "data.csv").suffix or ".csv"
    p = d / f"data{suffix}"
    p.write_bytes(await file.read())
    live = Live(sid, str(p), None)
    SESSIONS[sid] = live
    opening = live.start()
    return {"id": sid, "opening": opening, "status": live.status()}


def _live(sid: str) -> Live:
    if sid not in SESSIONS:
        raise HTTPException(404, "no such session")
    return SESSIONS[sid]


@app.get("/api/sessions/{sid}")
def session_status(sid: str) -> Dict[str, Any]:
    return _live(sid).status()


@app.post("/api/sessions/{sid}/messages")
def post_message(sid: str, msg: Message) -> Dict[str, Any]:
    live = _live(sid)
    live.submit_async(msg.text)
    return live.status()


@app.get("/api/sessions/{sid}/bundle")
def session_bundle(sid: str) -> JSONResponse:
    return JSONResponse(_live(sid).bundle())


@app.post("/api/sessions/{sid}/subset/{pid}")
def session_subset(sid: str, pid: str) -> Dict[str, Any]:
    live = _live(sid)
    s = live.agent.session
    if s.phase != "DONE":
        raise HTTPException(409, "no finished analysis to break down")
    if not s.has(f"subset:{pid}"):
        probs = actions.validate("subset", {"provider": pid}, s, budget=False)
        if probs:
            raise HTTPException(400, "; ".join(probs))
        with live.lock:
            executor.execute(s, "subset", {"provider": pid}, source="harness", reason="breakdown requested in the explorer")
            s.save("trace.json")
    res = s.get(f"subset:{pid}")
    return {"id": res.id, "summary": res.summary, "tables": {k: export._records(Path(v)) for k, v in res.tables.items()}}


app.mount("/", StaticFiles(directory=str(SITE), html=True), name="site")
