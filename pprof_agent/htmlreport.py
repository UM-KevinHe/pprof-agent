"""The HTML report: the Markdown report rendered to one self-contained page, figures embedded inline.

A small converter for the subset of Markdown `report.render_markdown` writes: headings, paragraphs,
bullet lists, pipe tables, bold, code spans and image links. SVG figures are inlined (base64 data URIs),
so the page travels as one file. Harness-rendered, no model involved.
"""
from __future__ import annotations

import base64
import html
import re
from pathlib import Path
from typing import List

_STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap');
:root { --bg: #F4F6F8; --panel: #FFFFFF; --panel-2: #EEF2F6; --ink: #182028; --muted: #5E6B78; --line: #D9DFE6; --accent: #27588F; --accent-soft: #E3ECF7;
  --worse: #B9432E; --better: #2E7A4E; }
@media (prefers-color-scheme: dark) { :root { --bg: #10161D; --panel: #171F28; --panel-2: #1F2933; --ink: #E7ECF1; --muted: #9AA7B4; --line: #2B3742; --accent: #86B4EA; --accent-soft: #1E3147; --worse: #E0836F; --better: #6FC48E; color-scheme: dark; } }
body { background: var(--bg); color: var(--ink); font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif; font-size: 15px; line-height: 1.5; margin: 0; padding: 24px max(16px, 4vw) 48px; }
.page { max-width: 960px; margin: 0 auto; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 28px 36px; box-shadow: 0 1px 2px rgba(20,30,40,.06), 0 8px 24px -16px rgba(20,30,40,.25); }
h1, h2, h3 { font-family: 'Fraunces', 'Iowan Old Style', Georgia, serif; font-weight: 600; line-height: 1.2; text-wrap: balance; }
h1 { font-size: 1.7rem; margin: 0 0 .2em; }
h1 + p { color: var(--muted); }
h2 { font-size: 1.25rem; margin: 2em 0 .5em; padding-top: 1em; border-top: 1px solid var(--line); }
h3 { font-size: 1.05rem; margin: 1.4em 0 .4em; }
p { max-width: 72ch; }
table { border-collapse: collapse; font-size: .84rem; margin: .6rem 0 1rem; max-width: 100%; display: block; overflow-x: auto; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--line); padding: .3rem .6rem; text-align: left; white-space: nowrap; }
th { background: var(--panel-2); font-weight: 600; color: var(--muted); font-size: .76rem; text-transform: uppercase; letter-spacing: .05em; }
code { font-family: 'IBM Plex Mono', Consolas, monospace; background: var(--panel-2); padding: 0 .3rem; border-radius: 4px; font-size: .86em; }
strong { font-weight: 600; }
ul { padding-left: 1.3em; }
li { margin: .2em 0; }
img { max-width: 100%; height: auto; display: block; margin: .6rem 0; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 6px; }
.caption { font-size: .85rem; color: var(--muted); }
.brand { font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; color: var(--accent); font-weight: 600; margin-bottom: .4rem; }
"""


def _inline(text: str) -> str:
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"_([^_]+)_", r"<em>\1</em>", t)
    return t


def _image(alt: str, src: str, run_dir: Path) -> str:
    p = (run_dir / src) if not Path(src).is_absolute() else Path(src)
    if p.exists():
        data = p.read_bytes()
        if p.suffix.lower() == ".svg":
            uri = "data:image/svg+xml;base64," + base64.b64encode(data).decode()
        else:
            uri = f"data:image/{p.suffix.lower().lstrip('.')};base64," + base64.b64encode(data).decode()
        return f'<img alt="{html.escape(alt)}" src="{uri}">'
    return f'<p class="caption">[figure not found: {html.escape(src)}]</p>'


def render(markdown_text: str, run_dir: Path, title: str = "Provider profiling report") -> str:
    out: List[str] = []
    lines = markdown_text.splitlines()
    i = 0
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            close_list()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            if len(cells) >= 2 and all(re.fullmatch(r"-+", c) for c in cells[1]):
                head, body = cells[0], cells[2:]
            else:
                head, body = None, cells
            out.append("<table>")
            if head:
                out.append("<tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr>")
            for r in body:
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            out.append("</table>")
            continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", line)
        if m:
            close_list()
            out.append(_image(m.group(1), m.group(2), run_dir))
        elif line.startswith("# "):
            close_list()
            out.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("## "):
            close_list()
            out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(line[2:])}</li>")
        elif line.strip() == "":
            close_list()
        else:
            close_list()
            out.append(f"<p>{_inline(line)}</p>")
        i += 1
    close_list()
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{_STYLE}</style></head><body><div class=\"page\"><div class=\"brand\">pprof Agent</div>" + "\n".join(out) + "</div></body></html>")


def write(markdown_text: str, run_dir: str, title: str = "Provider profiling report") -> Path:
    p = Path(run_dir) / "report.html"
    p.write_text(render(markdown_text, Path(run_dir), title), encoding="utf-8")
    return p
