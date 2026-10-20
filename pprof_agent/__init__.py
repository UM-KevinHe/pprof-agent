"""pprof_agent -- a plan-and-execute agent for provider profiling.

The harness (this package) is deterministic. A language model, when one is configured, acts only
where the system meets ordinary language: typing a message, reading a measure specification out of
prose, planning the next engine operation from a closed set, reading an observation, choosing the
report's figures from a catalogue, and writing prose with named references. Every number comes from
the engine: `pprofR` and `grplasso` (R, through `engine.rbridge`) and `pprof_py` (Python, the
presentation layer and cross-checks).

Design: Agent/DESIGN.md (confirmed by Yubo 2026-10-18). Project memory: chapterIII/CLAUDE.md.
"""
__version__ = "0.1.0"
