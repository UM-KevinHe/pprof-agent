"""The session driver: one analysis as a conversation.

`Agent.start()` profiles the data and asks what the harness cannot settle. `Agent.submit(message)`
reads a numbered reply itself; prose goes to the model for typing (intent) and, when it declares,
for extraction (spec_extraction), after which the harness completes, verifies, refuses or runs the
loop with the model as planner, lets the model choose the figures from the catalogue, and has it
write the prose of the report. Without a model the agent still runs: numbered replies and the
harness's own plan.

    python -m pprof_agent.chat --data data.csv --out runs/ --endpoint http://localhost:8000/v1 --model qwen3-8b-awq \
        --script conversation.txt        # one message per line; '#' lines are comments
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import actions, boundary, compare, knowledge, loop, pipeline, report, rules, specification as S
from .engine import executor
from .engine.rbridge import RError
from .state import Session

_NUMBERED = re.compile(r"(?:^|\s)\d{1,2}\)\s")
SCOPE_WORDS = {
    "crossed_providers": "it adds a second provider level (such as the discharging hospital) crossed with the facility; this version fits one "
                         "provider level and keeps the second as a caveat, since the staged model for crossed providers is later work",
    "time_to_event_outcome": "the outcome is a time to event; this version handles binary and binomial outcomes only",
    "count_outcome": "the outcome is a count or a rate over time; this version handles binary and binomial outcomes only",
    "mixed_effects": "it asks for random or mixed provider effects; this version fits fixed provider effects, which do not shrink a provider towards the norm",
    "reliability": "it asks for a reliability or inter-unit reliability measure, which this version does not compute",
    "individual_prediction": "it asks for a prediction about an individual patient; the model profiles providers, not patients",
    "causal_claim": "it asks for a causal statement; the analysis establishes departures from expected outcomes, not their causes",
    "data_processing": "it asks the agent to change the data; the agent reads the file as given and never edits, filters or derives columns",
    "other": "it is not one of the operations of this version",
}
_FIGURE_PATH = re.compile(r"[\w./-]*?/figures/([\w.-]+\.svg)")


def _figure_names(text: str) -> List[str]:
    """The figure files a reply names (their basenames), for the transcript."""
    return list(dict.fromkeys(_FIGURE_PATH.findall(text or "")))


class Agent:
    def __init__(self, data_path: str, out_dir: str, client: Any = None, model: Optional[str] = None,
                 budget: int = 30, name: Optional[str] = None, log: Callable[[str], None] = print):
        run = pipeline.new_run_dir(out_dir, name)
        self.session = Session(data_path=str(data_path), run_dir=str(run), budget=budget)
        self.client, self.model, self.log = client, model, log
        self.has_model = client is not None and model is not None
        self.report_paths: Dict[str, str] = {}

    # ------------------------------------------------------------------ opening
    def start(self) -> str:
        s = self.session
        executor.profile(s)
        spec, items = S.complete(S.MeasureSpecification(), s.profile)
        s.spec, s.open_items = spec, items
        prof = s.profile
        cands = prof.get("candidates") or {}
        lines = [f"Read {prof.get('rows')} rows and {prof.get('n_columns')} columns from {Path(s.data_path).name}.",
                 f"Columns that could be the outcome: {', '.join(S._aslist(cands.get('outcome'))) or 'none'}; "
                 f"provider: {', '.join(S._aslist(cands.get('provider'))) or 'none'}; patient: {', '.join(S._aslist(cands.get('patient'))) or 'none'}."]
        for n in spec.notes:
            lines.append(f"Settled from the data: {n}.")
        if items:
            lead = self._word("ask", self._ask_material(items)) if self.has_model else None
            lines.append(lead or "A few fields are still open; the data cannot settle them.")
            lines.append(S.render_question(items))
            s.question = S.render_question(items)
        else:
            lines.append("Everything is settled; say `run` to start, or change a field.")
        text = "\n".join(lines)
        s.transcript.append({"role": "agent", "text": text, "figures": []})
        return text

    # ------------------------------------------------------------------ one message
    def submit(self, message: str) -> str:
        """One analyst message in, one reply out; both are kept in the transcript (with the figures the reply
        rendered) so a run can be replayed as it was shown."""
        s = self.session
        s.transcript.append({"role": "analyst", "text": message, "figures": []})
        reply = self._submit(message)
        s.transcript.append({"role": "agent", "text": reply, "figures": _figure_names(reply)})
        return reply

    def _submit(self, message: str) -> str:
        s = self.session
        s.turns.append(message)
        if _NUMBERED.search(" " + message):
            spec, problems = S.parse_reply(message, s.spec, s.profile)
            s.spec = spec
            out = [f"Not read: {p}" for p in problems]
            return "\n".join(out + [self._after_declaration()])
        if message.strip().lower() in ("run", "go", "go ahead", "run it", "start"):
            return self._after_declaration()
        if not self.has_model:
            if s.phase == "DONE":
                return ("No language model is configured, so a question in prose cannot be read; the report and the flag "
                        "explorer hold the results, and a numbered reply changes the measure.")
            return ("No language model is configured, so prose cannot be read. Answer by number:\n"
                    + (s.question or S.render_question(s.open_items)))
        intents = boundary.classify(self.client, self.model, message, self._state_for_model(), record_to=s.model_calls)
        kinds = [i["kind"] for i in intents]
        s.plan_log.append({"turn": len(s.turns), "intents": intents})
        if "declare_specification" in kinds or "edit_specification" in kinds:
            before = s.spec.config()
            spec, notes = boundary.extract_specification(self.client, self.model, message, s.profile, s.spec,
                                                         record_to=s.model_calls)
            spec, _ = S.fill_from_phrases(spec, message)      # the fixed phrasings the model left out, read by rule
            s.spec = spec
            prefix = [f"Not taken: {n}" for n in notes]
            if s.phase == "DONE" and spec.config() == before:
                return "\n".join(prefix + ["Nothing in the measure changed; the result stands."])
            if s.phase == "DONE":
                self._reset_results()
            return "\n".join(prefix + [self._after_declaration()])
        if "request_analysis" in kinds:
            return self._after_declaration()
        if "ask_about_result" in kinds:
            if s.phase != "DONE":
                return "No analysis has run yet; the open items are:\n" + (s.question or "")
            return self._explain(message)
        if "request_figure" in kinds:
            if s.phase != "DONE":
                return "No analysis has run yet, so there is nothing to draw."
            return self._figures(message)
        if "ask_about_method" in kinds:
            ans = knowledge.method_answer(message)
            if s.phase == "DONE":
                return self._explain(message, note=ans)
            return ans or "That is not covered by the method notes."
        if "out_of_scope" in kinds:
            reason = next((i.get("out_of_scope_reason") for i in intents if i["kind"] == "out_of_scope"), "other")
            why = SCOPE_WORDS.get(reason or "other", SCOPE_WORDS["other"])
            text = self._word("refusal", f"Reasons: the request is outside what this version does: {why}. Columns: " + ", ".join(S.profile_columns(s.profile)))
            return text or f"This version does not do that: {why}."
        return "I could not type that message. Answer by number, or ask about the result once one exists."

    def _reset_results(self) -> None:
        """A changed measure is a new analysis: the ledger, the plan and the verification start over in
        a fresh run directory; the profile and the turns are kept."""
        s = self.session
        run = pipeline.new_run_dir(str(Path(s.run_dir).parent), Path(s.run_dir).name + "_edit")
        s.run_dir = str(run)
        s.results, s.failed, s.plan_log, s.refusals = [], [], [], []
        s.steps, s.cost_used, s.finished, s.recommended = 0, 0, False, {}
        s.verification, s.admissible, s.config_hash = None, None, None
        executor.profile(s)

    # ------------------------------------------------------------------ declaration -> run
    def _after_declaration(self) -> str:
        s = self.session
        spec, items = S.complete(s.spec, s.profile)
        s.spec, s.open_items = spec, items
        if items:
            s.question = S.render_question(items)
            lead = self._word("ask", self._ask_material(items)) if self.has_model else None
            return (lead + "\n" if lead else "") + s.question
        return self._verify_and_run()

    def _verify_and_run(self) -> str:
        s = self.session
        try:
            card = executor.verify_card(s)
        except RError as exc:
            s.verification = S.Verification(False, [f"verification failed in the engine: {exc.info.get('message')}"], [], {})
            return "Refused: " + s.verification.refusals[0]
        s.verification = S.verify(s.spec, card)
        if not s.verification.ok:
            s.save("trace.json")
            text = self._word("refusal", "Reasons: " + "; ".join(s.verification.refusals) + ". Columns: " + ", ".join(S.profile_columns(s.profile)))
            return (text + "\n" if text else "") + "Refused:\n" + "\n".join(f"- {r}" for r in s.verification.refusals)
        s.admissible = rules.admissible(card, s.spec)
        s.config_hash = pipeline.config_hash(s.spec, s.data_path)
        planner = boundary.model_planner(self.client, self.model, s, knowledge.playbook) if self.has_model else None
        loop.run_loop(s, planner=planner, log=self.log)
        self._render_figures()
        prose = None
        if self.has_model:
            try:
                prose, notes = boundary.write_prose(self.client, self.model, report.build_references(s), self._facts(),
                                                    record_to=s.model_calls, columns=S.profile_columns(s.profile))
                s.prose = dict(prose)
                s.refusals += [f"prose: {n}" for n in notes]
            except ValueError as exc:
                s.refusals.append(f"prose call failed: {exc}")
        self.report_paths = pipeline.write_artifacts(s, prose=prose)
        rec = s.recommended
        from . import compare
        df = compare.table_of(s, rec["test"], rec["reference"])
        c = compare.counts(df)
        card = s.card or {}
        lines = [f"Done. {card.get('eligible_providers')} of {card.get('providers')} providers ({s.spec.provider}) profiled on "
                 f"{card.get('rows_complete')} rows with a fixed-effects logistic model adjusted for {len(s.spec.adjusters or [])} risk factors.",
                 f"Reported test: {rec['test']} ({rec['statistic']} statistic, {rec['variance']} variance) under the {rec['reference']} null.",
                 f"Flagged: {c['worse']} worse than expected, {c['better']} better, {c['as_expected']} as expected."]
        summary = self._prose_summary()
        if summary:
            lines += ["", summary]
        figs = self._figure_lines()
        if figs:
            lines += [""] + figs
        lines += ["", f"Report: {self.report_paths.get('html') or self.report_paths['report']}"]
        return "\n".join(lines)

    def _prose_summary(self) -> str:
        """The model's paragraphs on the flags and on the alternatives, every reference filled in by the harness
        (the same text the report carries), so the reply reads as an answer and not as a status line."""
        s = self.session
        if not s.prose:
            return ""
        refs = report.build_references(s)
        out = []
        for key in ("flags", "alternatives"):
            text = s.prose.get(key)
            if not text:
                continue
            try:
                out.append(report.resolve(text, refs))
            except KeyError:
                continue
        return "\n\n".join(out)

    def _figure_lines(self) -> List[str]:
        """The figures rendered so far, one line each with the file path (the web app shows them inline)."""
        return [f"{r.args.get('display')} of {r.args.get('of')}: {r.summary['path']}"
                for r in self.session.results if r.action == "figure" and (r.summary or {}).get("path")]

    # ------------------------------------------------------------------ figures, explanation, wording
    def _render_figures(self) -> None:
        s = self.session
        from .engine import pycore
        if self.has_model:
            try:
                chosen = boundary.select_figures(self.client, self.model, s, knowledge.display_catalogue(), record_to=s.model_calls)
            except ValueError as exc:
                s.refusals.append(f"figure selection failed: {exc}")
                chosen = []
            for f in chosen:
                try:
                    executor.execute(s, "figure", {"display": f["display"], "of": f["of"]}, source="planner", reason=f["reason"])
                except Exception as exc:  # noqa: BLE001
                    s.refusals.append(f"figure {f['display']} failed: {exc}")
            if chosen:
                return
        pycore.default_figures(s, log=self.log)

    def _figures(self, message: str) -> str:
        s = self.session
        try:
            chosen = boundary.select_figures(self.client, self.model, s, knowledge.display_catalogue() + "\n\nThe analyst asked:\n" + message,
                                             record_to=s.model_calls)
        except ValueError as exc:
            return f"The figure request could not be read: {exc}"
        out = []
        for f in chosen:
            rid = actions.result_id("figure", {"display": f["display"], "of": f["of"]})
            res = s.get(rid)
            if res is None:
                res = executor.execute(s, "figure", {"display": f["display"], "of": f["of"]}, source="planner", reason=f["reason"])
            out.append(f"{f['display']} of {f['of']}: {res.summary.get('path') or res.summary.get('error')}")
        pipeline.write_artifacts(s)
        return "\n".join(out) if out else "No display in the catalogue answers that request for the results at hand."

    def _providers_named(self, text: str) -> List[str]:
        """Fitted provider ids the question writes, from the gamma table."""
        s = self.session
        fit = s.get("fit")
        if fit is None or not fit.tables.get("gamma"):
            return []
        import csv
        with open(fit.tables["gamma"], encoding="utf-8") as fh:
            ids = [row["provider"] for row in csv.DictReader(fh)]
        from . import textmatch
        return textmatch.names_mentioned(text, ids)

    @staticmethod
    def _level_words(group: str, level: Any) -> str:
        """A breakdown level in words, never a digit: `with COPD`, `without COPD`, `with AGE in its low third`."""
        lv = str(level)
        if lv in ("1", "1.0", "True"):
            return f"with {group}"
        if lv in ("0", "0.0", "False"):
            return f"without {group}"
        if lv in ("low", "middle", "high"):
            return f"with {group} in its {lv} third"
        return f"in the {group} group {lv}"

    _TEST_PHRASES = {"robust_score": ("robust score",), "robust_wald": ("robust wald",),
                     "stabilized_score": ("stabilized score", "stabilised score"), "stabilized_wald": ("stabilized wald", "stabilised wald"),
                     "exact": ("exact",), "score": ("naive score", "the score test", "score test", "score statistic"),
                     "wald": ("naive wald", "the wald test", "wald test", "wald statistic")}

    def _tests_named(self, question: str) -> List[str]:
        """The tests of the taxonomy the question names."""
        low = question.lower().replace("_", " ")
        found = [t for t, phrases in self._TEST_PHRASES.items() if any(p in low for p in phrases)]
        if "score" in found and any(t in found for t in ("robust_score", "stabilized_score")):
            found.remove("score")
        if "wald" in found and any(t in found for t in ("robust_wald", "stabilized_wald")):
            found.remove("wald")
        return found

    def _breakdown_facts(self, pid: str) -> List[str]:
        """One provider's breakdown in words (no numbers): its status, and where its ratio is highest and lowest."""
        s = self.session
        res = s.get(f"subset:{pid}")
        if res is None:
            return []
        import pandas as pd
        from . import compare
        rec = s.recommended or {}
        df = compare.table_of(s, rec.get("test"), rec.get("reference", "theoretical"))
        status = None
        if df is not None:
            row = df[df["provider"].astype(str) == pid]
            status = row["status"].iloc[0] if len(row) else None
        if status is None:
            facts = [f"{pid} is not among the tested providers"]
        elif status == "as_expected":
            facts = [f"{pid} is not flagged under the reported configuration: its status is as expected, neither worse nor better"]
        else:
            facts = [f"{pid} is flagged {status} under the reported configuration"]
        tab = pd.read_csv(list(res.tables.values())[0]) if res.tables else None
        if tab is not None and len(tab) > 1:
            overall = float(tab.loc[tab["group"] == "all rows", "ratio"].iloc[0])
            facts.append(f"{pid}'s observed-to-expected ratio is {'below' if overall < 1 else 'above'} one overall"
                         + (", a difference the reported test does not distinguish from chance" if status == "as_expected" else ""))
            sub = tab[(tab["group"] != "all rows") & (tab["n"] >= 10)].copy()
            if len(sub):
                hi = sub.loc[sub["ratio"].idxmax()]
                lo = sub.loc[sub["ratio"].idxmin()]
                facts.append(f"within {pid}, the ratio is highest among patients {self._level_words(hi['group'], hi['level'])} "
                             f"and lowest among those {self._level_words(lo['group'], lo['level'])}")
                worse_side = "below" if s.spec.event_direction == "higher_is_better" else "above"
                uniform = (sub["ratio"] < 1).all() if worse_side == "below" else (sub["ratio"] > 1).all()
                facts.append(f"the departure is {'present in every patient group' if uniform else 'not uniform across patient groups'}")
        cm = res.summary.get("case_mix") or {}
        diff = [k for k, v in cm.items() if isinstance(v, dict) and v.get("population_mean") not in (None, 0)
                and abs(float(v["provider_mean"]) - float(v["population_mean"])) > 0.25 * abs(float(v["population_mean"]))]
        if diff:
            facts.append(f"{pid}'s case mix differs from the population's on {', '.join(diff)}; its expected events already reflect "
                         f"that case mix, because the model adjusts for the declared adjusters patient by patient")
        return facts

    AXIS_WORDS = {
        "reference": ("theoretical", "empirical", "null", "reference distribution"),
        "standardization": ("direct", "indirect", "standardiz", "transport"),
        "variance": ("variance", "robust", "stabilized", "stabilised", "naive", "sandwich", "cluster"),
        "statistic": ("score", "wald", "exact", "statistic"),
    }

    # the level of each axis a question may name, in the order tried (`indirect` before `direct`)
    LEVEL_WORDS = {
        "reference": (("empirical", "empirical"), ("theoretical", "theoretical")),
        "variance": (("stabilized", "stabilized"), ("stabilised", "stabilized"), ("robust", "robust"), ("naive", "naive")),
        "statistic": (("exact", "exact"), ("score", "score"), ("wald", "wald")),
        "standardization": (("indirect", "indirect"), ("direct", "direct")),
    }

    def _levels_named(self, question: str) -> Dict[str, str]:
        """The level the question names on each axis (`theoretical`, `robust`, `score`, `direct`), where it names one."""
        low = question.lower()
        out: Dict[str, str] = {}
        for axis, pairs in self.LEVEL_WORDS.items():
            for word, level in pairs:
                if word in low:
                    out[axis] = level
                    break
        return out

    def _reported_level(self, axis: str) -> Optional[str]:
        """The reported configuration's level on an axis; the reported flags come from the test of the provider
        effect, which is the indirect scale, so `direct` is always the alternative."""
        rec = self.session.recommended or {}
        if axis == "standardization":
            return "indirect"
        return rec.get(axis)

    def _axis_facts(self, question: str) -> Tuple[List[str], List[str]]:
        """What the computed alternatives on the axes the question names would change: the flags under each
        (as references) and the providers that change status (as ids the wording may repeat). When every level
        the question names is the reported one, the first fact says so: nothing would change."""
        s = self.session
        rec = s.recommended or {}
        low = question.lower()
        facts: List[str] = []
        ids: List[str] = []
        named = self._levels_named(question)
        if named and all(self._reported_level(a) == lv for a, lv in named.items()):
            what = " and ".join(f"the {lv} {'null' if a == 'reference' else a}" for a, lv in named.items())
            facts.append(f"the question names {what}, which is already the reported configuration: the reported flags are the "
                         f"flags under it and nothing would change; what follows is what the computed alternatives would change instead")
        for axis, words in self.AXIS_WORDS.items():
            if not any(w in low for w in words):
                continue
            r = s.get(f"compare:{axis}")
            if r is None:
                continue
            comps = r.summary.get("comparisons") or []
            if not comps:
                facts.append(f"no alternative on the {axis} axis was computed for this run")
                continue
            if axis == "standardization" and len(comps) > 1:
                # the direct rate was tested with several standard errors: the one matching the reported variance
                # treatment is the alternative worth telling (a naive one only says what the naive variance says)
                want = {"robust": "robust_wald", "stabilized": "stabilized_wald", "naive": "model"}.get(rec.get("variance"), "robust_wald")
                match = [c for c in comps if c["alternative"].startswith(f"direct_rate:{want}/")]
                if not match:
                    match = [c for c in comps if not c["alternative"].startswith("direct_rate:model/")]
                comps = match or comps[:1]
            for c in comps:
                tag = compare.alternative_tag(axis, c["alternative"])
                alt = compare.alternative_words(tag)
                same_w, same_b = c.get("worse_alt") == c.get("worse_base"), c.get("better_alt") == c.get("better_base")
                facts.append(f"under {alt} instead of the reported {rec.get('test')} under the {rec.get('reference')} null: "
                             + (f"the same number of providers flagged worse ([worse_under_{tag}]), " if same_w else f"[worse_under_{tag}] providers flagged worse instead of [flagged_worse], ")
                             + (f"the same number flagged better ([better_under_{tag}]), " if same_b else f"[better_under_{tag}] flagged better instead of [flagged_better], ")
                             + f"[changed_under_{tag}] providers change status")
                if not c.get("changed"):
                    facts.append(f"no provider changes status under {alt}")
                for key, what in (("worse_only_alt_providers", "newly flagged worse"), ("worse_only_base_providers", "no longer flagged worse"),
                                  ("better_only_alt_providers", "newly flagged better"), ("better_only_base_providers", "no longer flagged better")):
                    full = [str(x) for x in (c.get(key) or [])]
                    lst = full[:5]
                    if lst:
                        facts.append(f"{what} under {alt}: {', '.join(lst)}" + (" and others" if len(full) > 5 else ""))
                        ids += lst
            if axis == "reference" and s.has(f"en:{rec.get('test')}"):
                facts.append("the empirical null's widest size group has scale [en_scale_max] against one under the theoretical null "
                             "(ratio of the widest to the narrowest group scale [en_scale_ratio]); the group nulls are estimated from the "
                             "statistics themselves by size group, so they absorb the overdispersion the theoretical null ignores")
        return facts, ids

    _FLAG_WORDS = ("which", "list", "name", "who", "sure", "confident", "certain", "trust", "stable", "fragile", "reliable", "how many")

    def _flag_facts(self, question: str) -> Tuple[List[str], List[str]]:
        """For a question about which providers are flagged, or how sure the flags are: the flagged providers of
        the reported configuration by tail (ids the wording may repeat), how many keep their status under every
        computed alternative and which do not, and the calibration evidence when it exists."""
        s = self.session
        low = question.lower()
        if "flag" not in low or not any(w in low for w in self._FLAG_WORDS):
            return [], []
        stab = compare.stability(s)
        if not stab:
            return [], []
        facts: List[str] = []
        ids: List[str] = []
        tails = [t for t in ("worse", "better") if t in low] or ["worse", "better"]
        for tail in tails:
            full = stab[tail]["flagged"]
            lst = full[:20]
            ids += lst
            facts.append(f"flagged {tail} under the reported configuration ([flagged_{tail}] providers): "
                         + (", ".join(lst) + (" and others" if len(full) > 20 else "") if lst else "none"))
            if full:
                fragile = stab[tail]["fragile"]
                facts.append(f"of the providers flagged {tail}, [{tail}_under_every_alternative] are flagged {tail} under every one of the "
                             f"[n_alternatives] computed alternatives (other statistic, variance, reference distribution, standardization)"
                             + (f"; flagged {tail} under the reported configuration but not under every alternative: "
                                + ", ".join(fragile[:10]) + (" and others" if len(fragile) > 10 else "") if fragile else ""))
                ids += fragile[:10]
        refs = report.build_references(s)
        if refs.get("calibration_power_pct") is not None:
            facts.append(f"the calibration simulation ([calibration_n_sim] data sets, departures of odds ratio [calibration_effect_or] planted in a "
                         f"small share of the providers) put the reported test's type I error in the {s.spec.consequential_tail} tail at "
                         f"[calibration_rate_pct] against the nominal [calibration_nominal_pct] and its power at [calibration_power_pct] "
                         f"under the reported null")
        elif refs.get("calibration_rate_pct") is not None:
            facts.append(f"the calibration simulation ([calibration_n_sim] data sets with no provider truly different) put the reported test's "
                         f"type I error in the {s.spec.consequential_tail} tail at [calibration_rate_pct] against the nominal "
                         f"[calibration_nominal_pct] under the reported null")
        else:
            facts.append("no calibration simulation was run in this analysis; the reported test was chosen by the rules")
        return facts, ids

    def _explain(self, question: str, note: str = "") -> str:
        s = self.session
        extra: List[str] = [f"method note: {note}"] if note else []
        allow = self._providers_named(question)
        ran = False
        for pid in allow[:2]:
            if not s.has(f"subset:{pid}"):
                probs = actions.validate("subset", {"provider": pid}, s, budget=False)
                if not probs:
                    try:
                        executor.execute(s, "subset", {"provider": pid}, source="harness", reason="a question about this provider")
                        ran = True
                    except RError as exc:
                        s.refusals.append(f"subset {pid}: {exc.info.get('message', '')}")
                else:
                    s.refusals.append(f"subset {pid} not run: " + "; ".join(probs))
            extra += self._breakdown_facts(pid)
        axis_facts, axis_ids = self._axis_facts(question)
        extra += axis_facts
        flag_facts, flag_ids = self._flag_facts(question)
        extra += flag_facts
        axis_ids = axis_ids + [i for i in flag_ids if i not in axis_ids]
        adm, rec = s.admissible, s.recommended or {}
        for t in reversed(self._tests_named(question)):
            if adm and t in rules.TESTS:
                why = "it is the reported test" if t == rec.get("test") else self._words(rules.why_not(adm, t))
                extra.insert(0, f"the question names the {t} test: {why}")
        allow = allow + [i for i in axis_ids if i not in allow]
        if ran:
            pipeline.write_artifacts(s)
        refs = report.build_references(s)
        # the explanation is offered the numbers only: the names of the test, reference and standardization are
        # in the facts as words, so a word-valued reference would only be written beside its value
        offered = [k for k, v in refs.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        material = ("References (each stands where a number would be read): " + "; ".join(f"[{k}]: {boundary._meaning(k)}" for k in offered)
                    + "\nFacts:\n" + "\n".join("- " + f for f in self._facts() + extra) + "\nQuestion: " + question)
        text = self._word("explain", material, allow=allow)
        if not text:
            return "The explanation could not be written within the rules (no numbers, only named references)."
        try:
            out = report.resolve(text, refs)
        except KeyError as exc:
            return f"The explanation named an unknown quantity ({exc}); here it is with the name left in: {text}"
        return out + self._flag_lists(question, out)

    def _flag_lists(self, question: str, answer: str = "") -> str:
        """The flagged providers of the reported configuration, listed by the harness (never the model) under an
        answer to a question that asks which providers are flagged; the ones that are not flagged under every
        computed alternative are starred. Skipped for a tail the answer already names in full."""
        low = question.lower()
        if "flag" not in low or not any(w in low for w in ("which", "list", "name", "who")):
            return ""
        stab = compare.stability(self.session)
        if not stab:
            return ""
        tails = [t for t in ("worse", "better") if t in low] or ["worse", "better"]
        lines = []
        for tail in tails:
            ids = stab[tail]["flagged"]
            if not ids:
                continue
            stable = set(stab[tail]["stable"])
            if all(i in answer for i in ids) and len(stable) == len(ids):
                continue
            lines.append(f"Flagged {tail} under the reported configuration ({len(ids)}): "
                         + ", ".join(i + ("" if i in stable else "*") for i in ids)
                         + ("" if len(stable) == len(ids) else "  (* not under every computed alternative)"))
        return ("\n\n" + "\n".join(lines)) if lines else ""

    def _word(self, act: str, material: str, allow: Optional[List[str]] = None) -> Optional[str]:
        if not self.has_model:
            return None
        try:
            return boundary.word(self.client, self.model, act, material, S.profile_columns(self.session.profile),
                                 record_to=self.session.model_calls, allow=allow)
        except ValueError as exc:
            self.session.refusals.append(f"{act} wording failed: {exc}")
            return None

    def _ask_material(self, items: List[S.OpenItem]) -> str:
        return ("Open items (the harness will list them with their options):\n"
                + "\n".join(f"- {it.field}: {it.question} ({it.evidence})" for it in items)
                + "\nColumns: " + ", ".join(S.profile_columns(self.session.profile)))

    def _state_for_model(self) -> str:
        s = self.session
        return s.state_line() + (" | result: present" if s.phase == "DONE" else " | result: none")

    _NUMBER_REFS = ("patients_per_provider_median", "rows_per_patient_max", "n_invariant", "n_eligible", "n_providers", "eligibility",
                    "few_patients_threshold", "en_min_providers", "stabilized_max_rows", "exact_max_trials", "en_scale_max", "en_scale_ratio",
                    "n_providers_excluded", "n_adjusters", "calibration_rate_pct", "calibration_nominal_pct", "calibration_n_sim",
                    "worse_under_every_alternative", "better_under_every_alternative", "n_alternatives",
                    "en_location_max", "en_scale_threshold", "en_location_threshold", "calibration_default_rate_pct",
                    "calibration_power_pct", "calibration_default_power_pct", "calibration_effect_or")
    _SMALL = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")

    def _words(self, text: str, refs: Optional[Dict[str, Any]] = None) -> str:
        return words_with_refs(text, refs if refs is not None else report.build_references(self.session), self._NUMBER_REFS)

    def _facts(self) -> List[str]:
        """The analysis in words, for the writing acts: no numbers, so none can be copied."""
        s = self.session
        spec, adm, rec = s.spec, s.admissible, s.recommended or {}
        facts = [f"the outcome column is {spec.outcome}; one row is a {spec.unit}; a higher rate is {'worse' if spec.event_direction == 'higher_is_worse' else 'better'}",
                 f"the provider column is {spec.provider}; the patient column is {spec.patient}",
                 f"the adjusters were {spec.sources.get('adjusters')} by the analyst; the eligibility threshold was {spec.sources.get('eligibility')}",
                 f"standardization: {spec.standardization}; norm: {spec.norm}; the {spec.consequential_tail} tail carries the consequences"]
        if adm:
            facts.append("statistics computed: " + ", ".join(adm.tests_computed))
            facts.append("admissible as the reported test: " + ", ".join(adm.tests_reported))
            for k, v in adm.reasons.items():
                facts.append(f"rule ({k}): {self._words(v)}")
        if rec:
            facts.append(f"the reported test is {rec.get('test')} ({rec.get('statistic')} statistic, {rec.get('variance')} variance) under the {rec.get('reference')} null")
            facts.append("why the reference: " + self._words(str(rec.get("reasons", {}).get("reference", ""))))
            facts.append("why the test: " + self._words(str(rec.get("reasons", {}).get("test", ""))))
            if adm:
                refs = report.build_references(s)
                for t in adm.tests_computed:
                    if t == rec.get("test"):
                        continue
                    if t == adm.default_test and refs.get("calibration_default_power_pct") is not None:
                        facts.append(f"why not {t}: it is the rule default, but the calibration simulation scored the reported test higher: "
                                     f"in the {spec.consequential_tail} tail the default had type I error [calibration_default_rate_pct] and "
                                     f"power [calibration_default_power_pct], the reported test [calibration_rate_pct] and [calibration_power_pct], "
                                     f"against the nominal [calibration_nominal_pct] under the {rec.get('reference')} null")
                    elif t == adm.default_test and refs.get("calibration_default_rate_pct") is not None:
                        facts.append(f"why not {t}: it is the rule default, but the calibration simulation put its type I error in the "
                                     f"{spec.consequential_tail} tail at [calibration_default_rate_pct] against the nominal [calibration_nominal_pct] "
                                     f"under the {rec.get('reference')} null, and the reported test's [calibration_rate_pct] came nearer")
                    else:
                        facts.append(f"why not {t}: {self._words(rules.why_not(adm, t))}")
        for axis in ("statistic", "variance", "reference", "standardization"):
            r = s.get(f"compare:{axis}")
            if r and r.summary.get("comparisons"):
                mx = r.summary.get("max_changed") or 0
                facts.append(f"under another {axis}: {'no provider' if mx == 0 else 'some providers' if mx < 10 else 'many providers'} change status")
        if s.has("sm"):
            n = (s.get("sm").summary.get("flips") or {}).get("n") or 0
            facts.append("verdicts flip between indirect and direct standardization for " + ("no provider" if n == 0 else "some providers"))
        if s.has("selection"):
            facts.append("the adjusters were selected by cross-validated penalized regression with the provider effects unpenalized")
        return facts


_SMALL_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def words_with_refs(text: str, refs: Dict[str, Any], names: Tuple[str, ...]) -> str:
    """Rule text for the model: every number becomes the reference that carries it, a small integer its word,
    anything else 'some' -- so the wording can only name references, never copy a digit. A number followed
    by % can only be a percent reference; 'nominal' before it picks the nominal level; otherwise the first
    reference in `names` with that value wins."""
    by_value: Dict[str, List[str]] = {}
    for k in names:
        v = refs.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
            continue
        forms = {str(v), f"{v:.2f}", f"{v:.1f}", f"{v:.3f}"}
        if float(v).is_integer():
            forms.add(str(int(v)))
        for f in forms:
            by_value.setdefault(f, []).append(k)

    def sub(m: "re.Match[str]") -> str:
        tok = m.group(0)
        pct = text[m.end():m.end() + 1] == "%"
        before = text[max(0, m.start() - 16):m.start()].lower()
        cands = by_value.get(tok, [])
        if pct:
            cands = [c for c in cands if c.endswith("_pct")]
            if "nominal" in before:
                cands = [c for c in cands if "nominal" in c] or cands
            return f"[{cands[0]}]" if cands else "some"
        if tok.isdigit() and int(tok) <= 2:
            return _SMALL_WORDS[int(tok)]      # the constants of rule text (one axis, two tails), never a reference
        cands = [c for c in cands if not c.endswith("_pct")] or cands
        if cands:
            return f"[{cands[0]}]"
        if tok.isdigit() and int(tok) <= 10:
            return _SMALL_WORDS[int(tok)]
        return "some"
    out = re.sub(r"\d+(?:\.\d+)?", sub, text)
    return re.sub(r"(\[[a-z_]+_pct\])\s*%", r"\1", out)


# ---------------------------------------------------------------------------------------- scripted CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pprof_agent.chat")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default=os.environ.get("PPROF_AGENT_ENDPOINT"))
    ap.add_argument("--model", default=os.environ.get("PPROF_AGENT_MODEL"))
    ap.add_argument("--script", help="file with one message per line")
    ap.add_argument("--fake", action="store_true", help="use the fake client (tests)")
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--name")
    a = ap.parse_args(argv)
    client = None
    if a.fake:
        from .fake import FakeClient
        client, a.model = FakeClient(), "fake"
    elif a.endpoint:
        from openai import OpenAI
        client = OpenAI(base_url=a.endpoint, api_key=os.environ.get("OPENAI_API_KEY", "none"))
    agent = Agent(a.data, a.out, client=client, model=a.model, budget=a.budget, name=a.name)
    print(">>> start\n" + agent.start(), flush=True)
    messages = []
    if a.script:
        messages = [l.strip() for l in Path(a.script).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    else:
        messages = [l.strip() for l in sys.stdin if l.strip()]
    for m in messages:
        print(f"\n>>> analyst: {m}", flush=True)
        print(agent.submit(m), flush=True)
    s = agent.session
    s.save("trace.json")
    print(f"\n>>> session phase {s.phase}; model calls {len(s.model_calls)}; refusals/notes {len(s.refusals)}")
    for mc in s.model_calls:
        print(f"    {mc.get('name')}: prompt {mc.get('prompt_tokens')} completion {mc.get('completion_tokens')} "
              f"{mc.get('seconds')}s finish {mc.get('finish_reason')}" + (f" thinking {mc.get('thinking_chars')} chars" if mc.get('thinking_chars') else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
