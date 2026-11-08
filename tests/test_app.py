"""The Gradio UI's functions without launching a server: needs gradio in the environment and
PPROF_AGENT_DATA (the UI runs the no-model path: numbered replies, the harness plans)."""
import importlib
import os
import sys
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")
if not (Path(__file__).resolve().parents[1] / "app.py").exists():
    pytest.skip("app.py (the Gradio UI) is archived; the web app in site/ replaced it", allow_module_level=True)
DATA = os.environ.get("PPROF_AGENT_DATA")
pytestmark = pytest.mark.skipif(not DATA, reason="PPROF_AGENT_DATA not set")


class _Upload:
    def __init__(self, path: str):
        self.name = path


def test_app_functions(tmp_path, monkeypatch):
    monkeypatch.setenv("PPROF_AGENT_OUT", str(tmp_path))
    monkeypatch.delenv("PPROF_AGENT_ENDPOINT", raising=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    app = importlib.import_module("app")
    app = importlib.reload(app)
    assert isinstance(app.demo, gr.Blocks)
    assert app.OUT == str(tmp_path)
    agent, history, msg, figs, rep = app.start(None, [])
    assert agent is None and "Upload" in history[-1]["content"]
    agent, history, msg, figs, rep = app.start(_Upload(str(Path(DATA) / "data.csv")), [])
    assert agent is not None and "rows" in history[-1]["content"] and msg == ""
    agent, history, msg, figs, rep = app.submit(agent, "   ", history)
    assert len(history) == 1                                         # an empty message changes nothing
    agent, history, msg, figs, rep = app.submit(agent, "1) AVF 2) higher_is_better 3) patient_month 6) B", history)
    assert agent.session.phase == "DONE", history[-1]["content"]
    assert history[-1]["role"] == "assistant" and "Report:" in history[-1]["content"]
    assert rep and Path(rep).name == "report.html" and Path(rep).exists()
    assert figs and all(f.endswith(".png") and Path(f).exists() for f in figs)
