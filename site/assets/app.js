/* pprof Agent — the web app. Reads one run bundle (data/<dataset>/<run>.json) and shows it: the measure and
   the data, the conversation and the planner's steps, the flag explorer across every computed configuration,
   the report, and the audit trail. Replay mode needs no server; live mode talks to server.py when present. */
(() => {
'use strict';
const $ = (sel, el) => (el || document).querySelector(sel);
const $$ = (sel, el) => Array.from((el || document).querySelectorAll(sel));
const state = { index: null, bundle: null, dataset: null, run: null, tab: 'overview', cfg: null, pid: null,
                showRefNames: false, rendered: {}, mode: 'replay', live: null };

// ------------------------------------------------------------------ small utilities
function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function fmt(x, d) {
  if (x == null || x === '' || (typeof x === 'number' && isNaN(x))) return '—';
  if (typeof x !== 'number') return String(x);
  if (Number.isInteger(x) && d == null) return x.toLocaleString();
  const dd = d == null ? (Math.abs(x) >= 100 ? 0 : Math.abs(x) >= 10 ? 1 : Math.abs(x) >= 1 ? 2 : 3) : d;
  return x.toLocaleString(undefined, { minimumFractionDigits: dd, maximumFractionDigits: dd });
}
function pct(x, d) { return x == null ? '—' : fmt(x, d == null ? 1 : d) + '%'; }
function el(tag, attrs, children) {
  const e = document.createElement(tag);
  for (const k in (attrs || {})) {
    if (k === 'class') e.className = attrs[k];
    else if (k === 'html') e.innerHTML = attrs[k];
    else if (k === 'text') e.textContent = attrs[k];
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), attrs[k]);
    else if (attrs[k] != null) e.setAttribute(k, attrs[k]);
  }
  for (const c of [].concat(children || [])) if (c != null) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function words(s) { return String(s || '').replace(/_/g, ' '); }
function titleCase(s) { s = words(s); return s.charAt(0).toUpperCase() + s.slice(1); }
const STATUS_WORD = { worse: 'worse than expected', better: 'better than expected', as_expected: 'as expected' };

// ------------------------------------------------------------------ the taxonomy and the configurations
const STAT_OF = { exact: 'exact', score: 'score', wald: 'wald', robust_wald: 'wald', robust_score: 'score', stabilized_wald: 'wald', stabilized_score: 'score' };
const VAR_OF = { exact: 'naive', score: 'naive', wald: 'naive', robust_wald: 'robust', robust_score: 'robust', stabilized_wald: 'stabilized', stabilized_score: 'stabilized' };
const SE_OF = { naive: 'model', robust: 'robust_wald', stabilized: 'stabilized_wald' };
function testFor(stat, variance) { for (const t in STAT_OF) if (STAT_OF[t] === stat && VAR_OF[t] === variance) return t; return null; }
function tailOf(raw, ed) {
  if (raw == null || raw === '' || isNaN(Number(raw))) return null;
  const r = Number(raw);
  if (r === 0) return 'as_expected';
  const higher = r === 1;
  return ed === 'higher_is_worse' ? (higher ? 'worse' : 'better') : (higher ? 'better' : 'worse');
}
function reportedCfg(b) {
  const r = b.recommended || {};
  const t = r.test || (b.admissible || {}).default_test || 'score';
  return { scale: 'effect', stat: STAT_OF[t], variance: VAR_OF[t], reference: r.reference || 'theoretical' };
}
function cfgKey(c) {
  if (c.scale === 'direct') return `measure_direct_rate_${SE_OF[c.variance]}_${c.reference}`;
  const t = testFor(c.stat, c.variance);
  return t ? (c.reference === 'empirical' ? `en_${t}` : `test_${t}`) : null;
}
function cfgLabel(c) {
  if (c.scale === 'direct') return `direct rate · ${words(SE_OF[c.variance])} se · ${c.reference} null`;
  return `${words(testFor(c.stat, c.variance))} · ${c.reference} null`;
}
function tableFor(b, c) {
  const key = cfgKey(c);
  const rows = key && b.tables[key];
  if (!rows) return null;
  const flagCol = c.scale === 'effect' && c.reference === 'empirical' ? 'flag_en_raw' : 'flag_raw';
  const ed = b.specification.event_direction;
  return { key, rows, flagCol, cfg: c, label: cfgLabel(c),
           status: Object.fromEntries(rows.map(r => [String(r.provider), tailOf(r[flagCol], ed)])) };
}
function sameCfg(a, c) { return a && c && a.scale === c.scale && a.stat === c.stat && a.variance === c.variance && a.reference === c.reference; }
function allConfigs(b) {
  const out = [];
  for (const key of Object.keys(b.tables)) {
    let c = null;
    if (key.startsWith('test_')) { const t = key.slice(5); if (STAT_OF[t]) c = { scale: 'effect', stat: STAT_OF[t], variance: VAR_OF[t], reference: 'theoretical' }; }
    else if (key.startsWith('en_') && !key.startsWith('en_groups_')) { const t = key.slice(3); if (STAT_OF[t]) c = { scale: 'effect', stat: STAT_OF[t], variance: VAR_OF[t], reference: 'empirical' }; }
    else if (key.startsWith('measure_direct_rate_')) {
      const m = key.match(/^measure_direct_rate_(model|robust_wald|stabilized_wald)_(theoretical|empirical)$/);
      if (m) c = { scale: 'direct', stat: 'wald', variance: { model: 'naive', robust_wald: 'robust', stabilized_wald: 'stabilized' }[m[1]], reference: m[2] };
    }
    if (c) { const t = tableFor(b, c); if (t) out.push(t); }
  }
  const order = { effect: 0, direct: 1 }, vo = { naive: 0, robust: 1, stabilized: 2 }, so = { score: 0, wald: 1, exact: 2 }, ro = { theoretical: 0, empirical: 1 };
  out.sort((x, y) => (order[x.cfg.scale] - order[y.cfg.scale]) || (vo[x.cfg.variance] - vo[y.cfg.variance]) || (so[x.cfg.stat] - so[y.cfg.stat]) || (ro[x.cfg.reference] - ro[y.cfg.reference]));
  return out;
}
function counts(t) { const c = { worse: 0, better: 0, as_expected: 0 }; for (const k in t.status) if (t.status[k]) c[t.status[k]]++; return c; }
function consequentialTailRaw(spec) {
  const worseHigher = spec.event_direction === 'higher_is_worse';
  if (spec.consequential_tail === 'better') return worseHigher ? 'lower' : 'higher';
  return worseHigher ? 'higher' : 'lower';
}
function truthOf(b, pid) {
  const t = (b.dataset || {}).truth;
  if (!t) return null;
  if ((t.worse || []).includes(pid)) return { kind: 'worse', text: `planted worse (${(t.worse_size_class || {})[pid] || ''} facility)` };
  if ((t.better || []).includes(pid)) return { kind: 'better', text: `planted better (${(t.better_size_class || {})[pid] || ''} facility)` };
  if ((t.invariant || []).includes(pid)) return { kind: 'warn', text: 'invariant outcomes (all zero)' };
  if (t.decoy === pid) return { kind: 'accent', text: 'the decoy: extreme case mix, true effect zero' };
  if ((t.tiny || []).includes(pid)) return { kind: 'expected', text: 'tiny facility (three patients or fewer)' };
  return { kind: 'expected', text: 'ordinary facility (no planted departure)' };
}

// ------------------------------------------------------------------ Plotly theme
function plotLayout(extra) {
  const ink = cssVar('--ink'), muted = cssVar('--muted'), line = cssVar('--line');
  return Object.assign({ paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: cssVar('--body'), color: ink, size: 12 },
    margin: { l: 56, r: 16, t: 28, b: 48 }, xaxis: { gridcolor: line, zerolinecolor: line, color: muted, title: { font: { color: muted } } },
    yaxis: { gridcolor: line, zerolinecolor: line, color: muted, title: { font: { color: muted } } }, hoverlabel: { font: { family: cssVar('--mono'), size: 12 } },
    legend: { orientation: 'h', y: 1.12, x: 0, font: { color: muted } } }, extra || {});
}
const PLOT_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'] };
const COLOR = () => ({ worse: cssVar('--worse'), better: cssVar('--better'), as_expected: cssVar('--expected'), accent: cssVar('--accent') });

// ------------------------------------------------------------------ data loading
async function loadIndex() {
  const r = await fetch('data/index.json');
  if (!r.ok) throw new Error('data/index.json not found');
  state.index = await r.json();
  const ds = $('#dataset');
  ds.innerHTML = '';
  for (const d of state.index.datasets) ds.append(el('option', { value: d.name, text: `${d.title} (${d.name})` }));
  const prefer = state.index.datasets.find(d => d.name === 'demo') || state.index.datasets.find(d => d.name === 'sfr_2025') || state.index.datasets[0];
  ds.value = prefer.name;
  if (state.index.datasets.length < 2) ds.closest('label').hidden = true;
  fillRuns();
}
function fillRuns() {
  const d = state.index.datasets.find(x => x.name === $('#dataset').value);
  const rs = $('#run');
  rs.innerHTML = '';
  const runs = d.runs.slice().sort((a, b) => (b.with_model - a.with_model) || (b.messages - a.messages) || a.name.localeCompare(b.name));
  for (const r of runs) rs.append(el('option', { value: r.file, text: r.title ? (r.title.length > 70 ? r.title.slice(0, 67) + '…' : r.title) : `${r.name}${r.with_model ? ' · with the model' : ' · harness only'}` }));
  const pick = runs.find(r => r.name === 'demo') || runs[0];
  if (pick) rs.value = pick.file;
}
async function loadRun(file) {
  $('#main').innerHTML = '<div class="loading">Loading the run…</div>';
  const r = await fetch(file);
  if (!r.ok) throw new Error(`${file} not found`);
  state.bundle = await r.json();
  state.cfg = reportedCfg(state.bundle);
  state.pid = null;
  state.rendered = {};
  const b = state.bundle;
  document.title = `pprof Agent · ${b.dataset.name} / ${b.run}`;
  showTab(state.tab, true);
}

// ------------------------------------------------------------------ tabs
const TABS = ['overview', 'conversation', 'explorer', 'report', 'audit'];
function showTab(tab, force) {
  if (!TABS.includes(tab)) tab = 'overview';
  state.tab = tab;
  for (const t of $$('.tab')) t.setAttribute('aria-selected', String(t.dataset.tab === tab));
  if (location.hash.slice(1) !== tab) history.replaceState(null, '', '#' + tab);
  if (!state.bundle) return;
  if (force || !state.rendered[tab]) {
    const main = $('#main');
    main.innerHTML = '';
    const ready = state.bundle.phase === 'DONE' && state.bundle.specification;
    if (!ready && tab !== 'conversation' && tab !== 'audit') {
      main.append(el('div', { class: 'card' }, [el('h2', { text: 'No finished analysis yet' }), el('p', { class: 'note', text: state.bundle.phase === 'REFUSED' ? 'The gate refused this specification; the conversation tab has the reasons.' : 'Declare the measure in the conversation; the analysis runs and this view fills in.' }),
        el('button', { class: 'btn primary', text: 'Go to the conversation', onclick: () => showTab('conversation') })]));
      state.rendered = { [tab]: true };
      return;
    }
    main.append({ overview: renderOverview, conversation: renderConversation, explorer: renderExplorer, report: renderReport, audit: renderAudit }[tab]());
    state.rendered = { [tab]: true };
    requestAnimationFrame(() => { for (const p of $$('.js-plot')) p._draw && p._draw(); });
  }
}

// ------------------------------------------------------------------ overview
function sourceChip(b, field) { const s = (b.specification.sources || {})[field]; return s ? el('span', { class: 'chip src', text: s }) : null; }
function renderOverview() {
  const b = state.bundle, spec = b.specification, card = b.card || {}, adm = b.admissible || {}, rec = b.recommended || {};
  const rep = tableFor(b, reportedCfg(b));
  const c = rep ? counts(rep) : { worse: 0, better: 0, as_expected: 0 };
  const stab = b.stability || {}, nAlt = stab.n_alternatives || 0;
  const view = el('div', { class: 'view' });
  const lede = (b.dataset.dictionary || '').split('\n').find(l => l.startsWith('One row is')) || '';
  view.append(el('div', { class: 'hero' }, [
    el('div', { class: 'card' }, [
      el('div', { class: 'eyebrow', text: 'the analysis' }),
      el('h1', { text: b.dataset.title || b.dataset.name }),
      el('p', { class: 'lede', text: lede.replace(/`/g, '') }),
      el('div', { class: 'chips' }, [
        el('span', { class: 'chip accent', text: `run: ${b.run}` }),
        el('span', { class: 'chip', text: b.model_calls.some(m => m.name === 'next_step') ? 'planned by the language model' : 'planned by the harness alone' }),
        el('span', { class: 'chip', text: `${b.results.length} steps` }),
        b.config_hash ? el('span', { class: 'chip mono', text: `config ${String(b.config_hash).slice(0, 10)}` }) : null,
      ]),
    ]),
    el('div', { class: 'card' }, [
      el('div', { class: 'eyebrow', text: 'how it works' }),
      el('div', { class: 'steps', style: 'margin-top:8px' }, [
        step('1', 'The analyst declares the measure in plain language: outcome, unit, direction, adjusters.'),
        step('2', 'The harness profiles the data and derives which tests the data admit, by rule.'),
        step('3', 'The model plans the analysis one step at a time; every proposal is verified before it runs.'),
        step('4', 'Flags are reported per tail, with the flags under every other admissible choice.'),
        step('5', 'The model writes the report; every number is a named reference the engine fills in.'),
      ]),
      el('p', { class: 'note', style: 'margin-top:10px', text: 'The model plans · evidence decides · numbers only from the engine.' }),
    ]),
  ]));
  function step(n, t) { return el('div', { class: 'step' }, [el('b', { text: n }), el('span', { text: t })]); }

  // the flags
  const flags = el('div', { class: 'card' }, [
    el('h2', { text: 'Flags under the reported configuration' }),
    el('p', { class: 'sub', text: rep ? `${rep.label}` : 'no reported table' }),
    el('div', { class: 'tiles' }, [
      tile(c.worse, 'flagged worse than expected', 'worse'), tile(c.better, 'flagged better than expected', 'better'), tile(c.as_expected, 'as expected', 'expected'),
      tile(nAlt ? `${(stab.worse || {}).stable ? stab.worse.stable.length : 0} / ${c.worse}` : '—', `of the worse flags hold under every one of ${nAlt} alternatives`),
    ]),
    el('p', { class: 'note', style: 'margin-top:10px' }, [
      'Reported test: ', el('b', { text: words(rec.test || '') }), ` (${rec.statistic || ''} statistic, ${rec.variance || ''} variance) under the ${rec.reference || ''} null. `,
      el('span', { text: (rec.reasons || {}).test || '' }),
    ]),
    el('div', { style: 'margin-top:10px' }, [el('button', { class: 'btn primary', text: 'Open the flag explorer', onclick: () => showTab('explorer') })]),
  ]);
  function tile(v, k, cls) { return el('div', { class: 'tile ' + (cls || '') }, [el('div', { class: 'v num', text: v }), el('div', { class: 'k', text: k })]); }
  view.append(flags);

  // measure and data
  const kv = el('dl', { class: 'kv' });
  const rows = [['outcome', spec.outcome + (spec.trials ? ` out of ${spec.trials}` : '')], ['one row is', words(spec.unit)], ['direction', words(spec.event_direction)],
    ['provider', spec.provider], ['patient', spec.patient], ['eligibility', `${spec.eligibility} patients`], ['adjusters', null], ['standardization', spec.standardization],
    ['norm', words(spec.norm)], ['consequential tail', spec.consequential_tail], ['level', spec.level]];
  for (const [k, v] of rows) {
    kv.append(el('dt', { text: k }));
    const field = { 'one row is': 'unit', direction: 'event_direction', 'consequential tail': 'consequential_tail' }[k] || k;
    if (k === 'adjusters') kv.append(el('dd', {}, [el('div', { class: 'chips' }, (spec.adjusters || []).map(a => el('span', { class: 'chip mono', text: a })).concat([sourceChip(b, 'adjusters')]))]));
    else kv.append(el('dd', {}, [el('span', { text: v == null ? '—' : String(v) }), ' ', sourceChip(b, field)]));
  }
  const sizes = el('div', { class: 'plot js-plot', style: 'min-height:200px' });
  sizes._draw = () => plotSizes(sizes);
  view.append(el('div', { class: 'grid two' }, [
    el('div', { class: 'card' }, [el('h2', { text: 'The measure as declared' }), el('p', { class: 'sub', text: 'each field with its source: quoted from the analyst, inferred from the data, asked, or a default' }), kv]),
    el('div', { class: 'card' }, [el('h2', { text: 'The data' }), el('div', { class: 'tiles' }, [
      tile(fmt(card.rows_complete), 'rows'), tile(fmt(card.patients), 'patients'), tile(`${fmt(card.eligible_providers)} / ${fmt(card.providers)}`, 'eligible providers'),
      tile(pct(100 * (card.event_rate || 0)), 'event rate'), tile(fmt(card.max_rows_per_patient), 'rows per patient, at most'),
      tile(fmt((card.patients_per_provider || {}).median), 'median patients per provider'), tile(fmt((card.invariant_providers || {}).total), 'providers with invariant outcomes'),
      tile(fmt((card.small_providers || {}).le_3), 'providers with three patients or fewer')]),
      el('h4', { text: 'Patients per provider', style: 'margin-top:12px' }), sizes]),
  ]));

  // rules
  const reasons = adm.reasons || {};
  const ul = el('ul', { class: 'rules' }, Object.keys(reasons).map(k => el('li', {}, [el('span', { class: 'name', text: titleCase(k) + ': ' }), el('span', { text: reasons[k] })])));
  view.append(el('div', { class: 'card' }, [
    el('h2', { text: 'What the data admit' }),
    el('p', { class: 'sub', text: 'the rules the harness applied to this file, before any model call' }),
    el('div', { class: 'chips', style: 'margin-bottom:10px' }, [
      el('span', { class: 'chip', text: 'computed: ' + (adm.tests_computed || []).map(words).join(', ') }),
      el('span', { class: 'chip accent', text: 'reportable: ' + (adm.tests_reported || []).map(words).join(', ') }),
      el('span', { class: 'chip', text: 'references: ' + (adm.references_computed || []).join(', ') }),
      el('span', { class: 'chip', text: 'direct rate tested on its own scale: ' + (adm.direct_transport ? 'yes' : 'no') }),
    ]), ul,
    (adm.caveats || []).length ? el('p', { class: 'note', text: 'Caveats: ' + adm.caveats.join(' ') }) : null,
  ]));
  return view;
}
function plotSizes(div) {
  const g = state.bundle.tables.gamma || [];
  if (!g.length) { div.textContent = 'no fit table'; return; }
  Plotly.newPlot(div, [{ type: 'histogram', x: g.map(r => r.n_patients), marker: { color: cssVar('--accent') }, hovertemplate: '%{x} patients: %{y} providers<extra></extra>' }],
    plotLayout({ height: 200, margin: { l: 40, r: 10, t: 10, b: 36 }, xaxis: { title: { text: 'patients' }, gridcolor: cssVar('--line'), color: cssVar('--muted') }, yaxis: { title: { text: 'providers' }, gridcolor: cssVar('--line'), color: cssVar('--muted') } }), PLOT_CONFIG);
}

// ------------------------------------------------------------------ conversation and the planner's timeline
function bubbleHtml(text) {
  let t = esc(text || '');
  t = t.replace(/\S*\/figures\/([\w.-]+\.svg)/g, '$1').replace(/Report: \S+/g, 'Report: see the Report tab.');
  t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
  return t;
}
function figureNode(name) {
  const f = (state.bundle.figures || {})[name];
  if (!f) return null;
  const d = el('div', { class: 'fig' });
  d.innerHTML = f.svg;
  if (f.caption) d.append(el('div', { class: 'cap', text: f.caption }));
  return d;
}
function messageNode(m) {
  const node = el('div', { class: 'msg ' + (m.role === 'analyst' ? 'analyst' : 'agent') }, [el('div', { class: 'who', text: m.role === 'analyst' ? 'analyst' : 'pprof agent' })]);
  const bub = el('div', { class: 'bubble', html: bubbleHtml(m.text) });
  for (const f of m.figures || []) { const n = figureNode(f); if (n) bub.append(n); }
  node.append(bub);
  return node;
}
function renderConversation() {
  const b = state.bundle;
  const view = el('div', { class: 'view' });
  const chat = el('div', { class: 'chat' });
  const transcript = b.transcript && b.transcript.length ? b.transcript : null;
  const head = el('div', { class: 'card' }, [
    el('h2', { text: 'The conversation' }),
    el('p', { class: 'sub', text: transcript ? `recorded with ${b.model_calls.some(m => m.name === 'next_step') ? 'the local model (Qwen3-8B)' : 'the harness alone'}; replayed here exactly as it was shown` : 'this run was recorded before transcripts were kept: the scripted messages are shown, the answers are in the report' }),
    el('div', { style: 'display:flex;gap:8px;flex-wrap:wrap' }, [
      transcript ? el('button', { class: 'btn primary', text: 'Replay', onclick: () => replay(chat, transcript) }) : null,
      transcript ? el('button', { class: 'btn', text: 'Show all', onclick: () => fill(chat, transcript) }) : null,
    ]),
  ]);
  const msgs = transcript || scriptMessages(b);
  fill(chat, msgs);
  const left = el('div', { style: 'display:grid;gap:12px;min-width:0' }, [chat]);
  if (live.sid && b.run === live.sid) left.append(composer());
  view.append(head, el('div', { class: 'convo' }, [left, timelineCard(b)]));
  return view;
}
function scriptMessages(b) {
  const lines = String((b.scenario || {}).script || '').split('\n').map(l => l.trim()).filter(l => l && !l.startsWith('#'));
  return lines.map(l => ({ role: 'analyst', text: l, figures: [] }));
}
function fill(chat, msgs) { chat.innerHTML = ''; for (const m of msgs) chat.append(messageNode(m)); }
let replayTimer = null;
function replay(chat, msgs) {
  if (replayTimer) clearTimeout(replayTimer);
  chat.innerHTML = '';
  let i = 0;
  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const next = () => {
    if (i >= msgs.length) { replayTimer = null; return; }
    const m = msgs[i++];
    if (m.role === 'analyst') { chat.append(messageNode(m)); chat.lastChild.scrollIntoView({ block: 'nearest', behavior: reduce ? 'auto' : 'smooth' }); replayTimer = setTimeout(next, reduce ? 0 : 700); return; }
    const typing = el('div', { class: 'msg agent' }, [el('div', { class: 'who', text: 'pprof agent' }), el('div', { class: 'bubble' }, [el('span', { class: 'typing' }, [el('i'), el('i'), el('i')])])]);
    chat.append(typing);
    replayTimer = setTimeout(() => { typing.replaceWith(messageNode(m)); chat.lastChild.scrollIntoView({ block: 'nearest', behavior: reduce ? 'auto' : 'smooth' }); replayTimer = setTimeout(next, reduce ? 0 : 900); }, reduce ? 0 : 1100);
  };
  next();
}
function timelineCard(b) {
  const results = b.results.slice().sort((x, y) => x.step - y.step);
  const log = b.plan_log || [];
  const nProp = log.filter(e => e.proposal).length, nRef = log.filter(e => e.problems && e.problems.length).length, nFall = log.filter(e => e.fallback).length;
  const planner = b.model_calls.filter(m => m.name === 'next_step');
  const secs = planner.reduce((a, m) => a + (m.seconds || 0), 0);
  const card = el('div', { class: 'card' }, [
    el('h2', { text: 'How the analysis was planned' }),
    el('p', { class: 'sub', text: planner.length ? `${planner.length} planner calls, ${nRef} proposals refused by the harness, ${nFall} harness defaults, ${fmt(secs / 60, 1)} minutes of planning` : 'the harness walked its protected core without a model' }),
    el('div', { class: 'legend' }, [el('span', {}, [el('i', { style: 'background:' + cssVar('--accent') }), 'planner']), el('span', {}, [el('i', { style: 'background:' + cssVar('--expected') }), 'harness / core']), el('span', {}, [el('i', { style: 'background:' + cssVar('--warn') }), 'harness default after refusals'])]),
  ]);
  const tl = el('div', { class: 'timeline', style: 'margin-top:10px' });
  for (const r of results) {
    const node = el('div', { class: 'tstep ' + r.source }, [el('div', { class: 'dot' })]);
    const body = el('div');
    body.append(el('div', {}, [el('span', { class: 'id', text: r.id }), ' ', el('span', { class: 'meta', text: `step ${r.step} · ${r.source.replace('_', ' ')}${r.seconds != null ? ` · ${fmt(r.seconds, 1)} s` : ''}` })]));
    const acc = log.find(e => e.step === r.step && e.proposal && !(e.problems && e.problems.length));
    if (acc && acc.proposal.reading) body.append(el('div', { class: 'reading', text: acc.proposal.reading }));
    if (r.reason) body.append(el('div', { class: 'why', text: r.reason }));
    for (const e of log.filter(e => e.step === r.step && e.problems && e.problems.length)) {
      const p = e.proposal || {};
      body.append(el('div', { class: 'refused', text: `refused: ${p.action} ${JSON.stringify(p.args || {})} — ${(e.problems || []).join('; ')}` }));
    }
    node.append(body);
    tl.append(node);
  }
  card.append(tl);
  return card;
}

// ------------------------------------------------------------------ the flag explorer
function reasonFor(b, axis, value) {
  const r = (b.admissible || {}).reasons || {};
  if (axis === 'variance' && value === 'naive') return r.naive || 'not computed for this file';
  if (axis === 'variance' && value === 'stabilized') return r.stabilized || 'not computed for this file';
  if (axis === 'stat' && value === 'wald') return r.wald || 'not computed for this file';
  if (axis === 'stat' && value === 'exact') return r.exact || 'the exact test is only computed with the naive variance';
  if (axis === 'reference' && value === 'empirical') return r.empirical_null || 'no empirical null for this test';
  if (axis === 'scale' && value === 'direct') return 'no transported test of the direct rate was computed with this variance and reference';
  return 'not computed for this file';
}
function cfgWith(c, axis, value) {
  const n = Object.assign({}, c, { [axis]: value });
  if (n.scale === 'direct') n.stat = 'wald';
  return n;
}
function segControl(b, axis, options, labelText) {
  const seg = el('div', { class: 'seg', role: 'group', 'aria-label': labelText });
  const rep = reportedCfg(b);
  for (const [value, label] of options) {
    const cand = cfgWith(state.cfg, axis, value);
    const directStat = axis === 'stat' && state.cfg.scale === 'direct' && value !== 'wald';
    const ok = !directStat && !!tableFor(b, cand);
    const btn = el('button', { type: 'button', 'aria-pressed': String(state.cfg[axis] === value), disabled: ok ? null : 'disabled',
      title: ok ? '' : (directStat ? 'the directly standardized rate is tested with a z statistic on its own scale' : reasonFor(b, axis, value)), onclick: () => { state.cfg = cand; drawExplorer(); } }, [label]);
    if (rep[axis] === value && axis !== 'scale') btn.append(el('span', { class: 'star', text: ' ★' }));
    seg.append(btn);
  }
  return el('label', {}, [labelText, seg]);
}
function renderExplorer() {
  const b = state.bundle;
  const view = el('div', { class: 'view' });
  view.append(el('div', { class: 'card', id: 'controls' }));
  const left = el('div', { style: 'display:grid;gap:18px;min-width:0' });
  left.append(el('div', { class: 'card', id: 'summary' }));
  left.append(el('div', { class: 'card' }, [el('h3', { text: 'Funnel' }), el('p', { class: 'sub', text: 'each provider against its precision; the limits are those of the test and reference in use, so the flags and the funnel agree' }), el('div', { class: 'plot tall js-plot', id: 'funnel' })]));
  left.append(el('div', { class: 'card' }, [el('h3', { text: 'Caterpillar' }), el('p', { class: 'sub', text: 'providers in order of their estimate, with intervals where the test provides them' }), el('div', { class: 'plot js-plot', id: 'cater' })]));
  left.append(el('div', { class: 'card' }, [el('h3', { text: 'Flag stability across every computed configuration' }), el('p', { class: 'sub', text: 'providers flagged under at least one configuration; click a row to inspect the provider; ★ marks the reported configuration' }), el('div', { class: 'plot js-plot', id: 'stability' })]));
  const cal = b.tables.calibration_tests;
  if (cal && cal.length) left.append(el('div', { class: 'card' }, [el('h3', { text: 'Calibration by simulation' }), el('p', { class: 'sub', text: 'the design copied with no provider truly different: each test\'s type I error per tail against the nominal level, under both reference distributions' }), el('div', { class: 'plot js-plot', id: 'caltests' })]));
  left.append(el('div', { class: 'grid two' }, [
    el('div', { class: 'card' }, [el('h3', { text: 'The empirical null by size group' }), el('div', { id: 'engroups' })]),
    el('div', { class: 'card' }, [el('h3', { text: 'The risk-adjustment model' }), el('div', { id: 'modelcard' }), el('div', { class: 'plot js-plot', id: 'decile', style: 'min-height:240px' })]),
  ]));
  const drawer = el('div', { class: 'drawer', id: 'drawer' });
  view.append(el('div', { class: 'explorer' }, [left, drawer]));
  requestAnimationFrame(drawExplorer);
  return view;
}
function drawExplorer() {
  const b = state.bundle;
  const controls = $('#controls');
  if (!controls) return;
  controls.innerHTML = '';
  controls.append(el('h2', { text: 'Flag explorer' }), el('p', { class: 'sub', text: 'the same providers under every admissible choice; greyed options are the ones the data do not admit (hover for the rule)' }));
  controls.append(el('div', { class: 'controls' }, [
    segControl(b, 'scale', [['effect', 'provider effect'], ['direct', 'direct rate']], 'scale'),
    segControl(b, 'stat', [['score', 'score'], ['wald', 'Wald'], ['exact', 'exact']], 'statistic'),
    segControl(b, 'variance', [['naive', 'naive'], ['robust', 'robust'], ['stabilized', 'stabilized']], 'variance'),
    segControl(b, 'reference', [['theoretical', 'theoretical'], ['empirical', 'empirical']], 'reference distribution'),
    el('label', {}, ['', el('button', { class: 'btn small', type: 'button', text: '★ back to the reported configuration', onclick: () => { state.cfg = reportedCfg(b); drawExplorer(); } })]),
  ]));
  const t = tableFor(b, state.cfg);
  const rep = tableFor(b, reportedCfg(b));
  const sum = $('#summary');
  sum.innerHTML = '';
  if (!t) { sum.append(el('p', { class: 'empty', text: 'This configuration was not computed.' })); return; }
  const c = counts(t);
  const tiles = [tile(c.worse, 'worse than expected', 'worse'), tile(c.better, 'better than expected', 'better'), tile(c.as_expected, 'as expected', 'expected')];
  if (rep && rep.key !== t.key) {
    const newly = Object.keys(t.status).filter(p => t.status[p] === 'worse' && rep.status[p] !== 'worse');
    const lost = Object.keys(rep.status).filter(p => rep.status[p] === 'worse' && t.status[p] !== 'worse');
    const changed = Object.keys(t.status).filter(p => t.status[p] && rep.status[p] && t.status[p] !== rep.status[p]);
    tiles.push(tile(changed.length, 'change status against the reported configuration'));
    sum.append(el('h3', { text: t.label }), el('div', { class: 'tiles' }, tiles));
    sum.append(el('div', { class: 'chips', style: 'margin-top:10px' }, [
      el('span', { class: 'chip worse', text: `newly worse: ${newly.length ? newly.slice(0, 12).join(', ') + (newly.length > 12 ? ' …' : '') : 'none'}` }),
      el('span', { class: 'chip expected', text: `no longer worse: ${lost.length ? lost.slice(0, 12).join(', ') + (lost.length > 12 ? ' …' : '') : 'none'}` }),
    ]));
  } else {
    sum.append(el('h3', {}, [t.label, ' ', el('span', { class: 'chip accent', text: 'reported' })]), el('div', { class: 'tiles' }, tiles));
  }
  sum.append(el('div', { class: 'legend', style: 'margin-top:10px' }, [el('span', { class: 'w' }, [el('i'), 'worse']), el('span', { class: 'b' }, [el('i'), 'better']), el('span', { class: 'e' }, [el('i'), 'as expected'])]));
  plotFunnel($('#funnel'), t);
  plotCaterpillar($('#cater'), t);
  plotStability($('#stability'));
  if ($('#caltests')) plotCalTests($('#caltests'));
  engroups($('#engroups'));
  modelCard($('#modelcard'), $('#decile'));
  drawer();
  function tile(v, k, cls) { return el('div', { class: 'tile ' + (cls || '') }, [el('div', { class: 'v num', text: v }), el('div', { class: 'k', text: k })]); }
}
function hoverText(r, t) {
  const bits = [`<b>${esc(r.provider)}</b>`];
  if (r.observed != null) bits.push(`observed ${fmt(r.observed)} · expected ${fmt(r.expected, 1)}`);
  if (r.n_patients != null) bits.push(`${fmt(r.n_patients)} patients · ${fmt(r.n_rows)} rows`);
  if (t.cfg.scale === 'direct') bits.push(`rate ${fmt(r.measure, 3)} (reference ${fmt(r.ref, 3)})`);
  else if (r.estimate != null) bits.push(`ratio ${fmt(r.estimate, 3)}`);
  if (r.z_en != null) bits.push(`z ${fmt(r.stat, 2)} → ${fmt(r.z_en, 2)} under the empirical null (group ${r.group})`);
  else if (r.stat != null) bits.push(`z ${fmt(r.stat, 2)}`);
  else if (r.z != null) bits.push(`z ${fmt(r.z, 2)}`);
  if (r.p != null) bits.push(`p ${fmt(r.p, 3)}`);
  bits.push(STATUS_WORD[t.status[String(r.provider)]] || 'not tested');
  return bits.join('<br>');
}
function byStatus(t, rows, xy) {
  const col = COLOR();
  return ['as_expected', 'better', 'worse'].map(s => {
    const rs = rows.filter(r => t.status[String(r.provider)] === s);
    const sel = rs.map(r => String(r.provider) === state.pid);
    return Object.assign({ type: 'scatter', mode: 'markers', name: STATUS_WORD[s], x: rs.map(xy.x), y: rs.map(xy.y), text: rs.map(r => hoverText(r, t)),
      hovertemplate: '%{text}<extra></extra>', customdata: rs.map(r => String(r.provider)),
      marker: { color: col[s], size: sel.map(v => v ? 14 : (s === 'as_expected' ? 6 : 9)), opacity: s === 'as_expected' ? .55 : .95, line: { width: sel.map(v => v ? 2 : 0), color: cssVar('--ink') } } }, xy.extra || {});
  });
}
function clickSelect(div) {
  div.on('plotly_click', ev => { const p = ev.points && ev.points[0]; if (p && p.customdata) { state.pid = String(p.customdata); drawExplorer(); } });
}
function plotFunnel(div, t) {
  const col = COLOR();
  let rows, xy, limits = [], yref = 1, xtitle, ytitle;
  if (t.cfg.scale === 'direct') {
    rows = t.rows.filter(r => r.se != null && r.se > 0 && r.measure != null);
    xy = { x: r => 1 / (r.se * r.se), y: r => r.measure };
    yref = rows.length ? rows[0].ref : null;
    xtitle = 'precision (1 / se²)'; ytitle = 'directly standardized rate';
    if (t.cfg.reference === 'theoretical') {
      const s = rows.slice().sort((a, b) => xy.x(a) - xy.x(b));
      limits = [{ x: s.map(xy.x), y: s.map(r => r.ref + 1.96 * r.se) }, { x: s.map(xy.x), y: s.map(r => r.ref - 1.96 * r.se) }];
    }
  } else {
    rows = t.rows.filter(r => r.funnel_estimate != null && r.funnel_precision != null);
    xy = { x: r => r.funnel_precision, y: r => r.funnel_estimate };
    xtitle = 'precision (expected events)'; ytitle = 'observed / expected';
    const s = rows.filter(r => r.funnel_upper != null).sort((a, b) => a.funnel_precision - b.funnel_precision);
    limits = [{ x: s.map(r => r.funnel_precision), y: s.map(r => r.funnel_upper) }, { x: s.map(r => r.funnel_precision), y: s.map(r => r.funnel_lower) }];
  }
  const traces = limits.map((l, i) => ({ type: 'scatter', mode: 'lines', name: i ? 'lower limit' : 'upper limit', x: l.x, y: l.y, line: { color: cssVar('--muted'), width: 1, dash: 'dot' }, hoverinfo: 'skip', showlegend: i === 0 }));
  traces.push(...byStatus(t, rows, xy));
  const layout = plotLayout({ height: 420, xaxis: { title: { text: xtitle }, type: 'log', gridcolor: cssVar('--line'), color: cssVar('--muted') }, yaxis: { title: { text: ytitle }, gridcolor: cssVar('--line'), color: cssVar('--muted') },
    shapes: yref != null ? [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: yref, y1: yref, line: { color: cssVar('--accent'), width: 1.2 } }] : [] });
  Plotly.newPlot(div, traces, layout, PLOT_CONFIG).then(() => clickSelect(div));
}
function plotCaterpillar(div, t) {
  const direct = t.cfg.scale === 'direct';
  const est = r => direct ? r.measure : r.estimate, lo = r => direct ? r.lower : r.ci_lower, hi = r => direct ? r.upper : r.ci_upper;
  const rows = t.rows.filter(r => est(r) != null && isFinite(est(r))).sort((a, b) => est(a) - est(b));
  const rank = new Map(rows.map((r, i) => [String(r.provider), i + 1]));
  const hasCI = rows.some(r => lo(r) != null && hi(r) != null);
  const xy = { x: r => rank.get(String(r.provider)), y: est, extra: hasCI ? {} : {} };
  const traces = byStatus(t, rows, xy);
  if (hasCI) for (const tr of traces) {
    const rs = rows.filter(r => t.status[String(r.provider)] === ({ 'as expected': 'as_expected', 'worse than expected': 'worse', 'better than expected': 'better' })[tr.name]);
    tr.error_y = { type: 'data', symmetric: false, array: rs.map(r => hi(r) != null ? hi(r) - est(r) : 0), arrayminus: rs.map(r => lo(r) != null ? est(r) - lo(r) : 0), color: tr.marker.color, thickness: 1, width: 0 };
  }
  const yref = direct ? (rows.length ? rows[0].ref : null) : 1;
  Plotly.newPlot(div, traces, plotLayout({ height: 320, xaxis: { title: { text: 'providers in order of their estimate' }, gridcolor: cssVar('--line'), color: cssVar('--muted') },
    yaxis: { title: { text: direct ? 'directly standardized rate' : 'standardized ratio' }, gridcolor: cssVar('--line'), color: cssVar('--muted') },
    shapes: yref != null ? [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: yref, y1: yref, line: { color: cssVar('--accent'), width: 1.2 } }] : [],
    annotations: hasCI ? [] : [{ xref: 'paper', yref: 'paper', x: 1, y: 1.06, showarrow: false, text: 'no interval for this test', font: { color: cssVar('--muted'), size: 11 } }] }), PLOT_CONFIG).then(() => clickSelect(div));
}
function plotStability(div) {
  const b = state.bundle, cfgs = allConfigs(b), rep = reportedCfg(b);
  const pids = new Set();
  for (const t of cfgs) for (const p in t.status) if (t.status[p] === 'worse' || t.status[p] === 'better') pids.add(p);
  const score = p => { let w = 0, bt = 0; for (const t of cfgs) { if (t.status[p] === 'worse') w++; if (t.status[p] === 'better') bt++; } return [w, bt]; };
  const rows = Array.from(pids).sort((a, c) => { const [wa, ba] = score(a), [wc, bc] = score(c); return (wc - wa) || (bc - ba) || a.localeCompare(c); });
  if (!rows.length) { div.textContent = 'no provider is flagged under any configuration'; return; }
  const val = { worse: 1, as_expected: 0, better: -1 };
  const z = rows.map(p => cfgs.map(t => t.status[p] == null ? null : val[t.status[p]]));
  const text = rows.map(p => cfgs.map(t => `${p} · ${t.label}<br>${STATUS_WORD[t.status[p]] || 'not tested'}`));
  const col = COLOR();
  const xlabels = cfgs.map(t => (sameCfg(t.cfg, rep) ? '★ ' : '') + t.label.replace(' · ', '<br>'));
  Plotly.newPlot(div, [{ type: 'heatmap', z, x: xlabels, y: rows, text, hovertemplate: '%{text}<extra></extra>', zmin: -1, zmax: 1, showscale: false, xgap: 2, ygap: 2,
    colorscale: [[0, col.better], [0.5, col.as_expected], [1, col.worse]] }],
    plotLayout({ height: Math.max(300, 22 * rows.length + 120), margin: { l: 70, r: 10, t: 10, b: 110 }, xaxis: { side: 'bottom', tickangle: -35, tickfont: { size: 10 }, color: cssVar('--muted') },
      yaxis: { autorange: 'reversed', tickfont: { family: cssVar('--mono'), size: 10 }, color: cssVar('--muted') } }), PLOT_CONFIG)
    .then(() => div.on('plotly_click', ev => { const p = ev.points && ev.points[0]; if (p) { state.pid = String(p.y); drawExplorer(); } }));
}
function plotCalTests(div) {
  const b = state.bundle, rows = b.tables.calibration_tests || [];
  const tests = Array.from(new Set(rows.map(r => r.test)));
  const col = COLOR(), tailRaw = consequentialTailRaw(b.specification);
  const traces = [];
  for (const ref of ['theoretical', 'empirical']) for (const tail of ['higher', 'lower']) {
    const rs = tests.map(t => rows.find(r => r.test === t && r.reference === ref && r.tail === tail));
    if (!rs.some(Boolean)) continue;
    const tailWord = tail === tailRaw ? `${b.specification.consequential_tail} tail (consequential)` : `${tail === tailRaw ? '' : 'other'} tail`;
    traces.push({ type: 'bar', name: `${ref} · ${tailWord}`, x: tests.map(words), y: rs.map(r => r ? r.rate_pct : null),
      error_y: { type: 'data', array: rs.map(r => r ? r.se_pct : 0), thickness: 1 },
      marker: { color: ref === 'empirical' ? col.accent : col.as_expected, opacity: tail === tailRaw ? .95 : .5 }, hovertemplate: '%{x}: %{y:.1f}% type I error<extra>' + ref + ' · ' + tail + '</extra>' });
  }
  const nominal = rows.length ? rows[0].nominal_pct : 2.5;
  Plotly.newPlot(div, traces, plotLayout({ height: 320, barmode: 'group', yaxis: { title: { text: 'type I error, % per tail' }, gridcolor: cssVar('--line'), color: cssVar('--muted') }, xaxis: { color: cssVar('--muted') },
    shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: nominal, y1: nominal, line: { color: col.worse, width: 1.5, dash: 'dash' } }],
    annotations: [{ xref: 'paper', x: 1, y: nominal, text: `nominal ${nominal}%`, showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { color: col.worse, size: 11 } }] }), PLOT_CONFIG);
}
function engroups(div) {
  const b = state.bundle, t = (b.recommended || {}).test, rows = b.tables[`en_groups_${t}`];
  div.innerHTML = '';
  if (!rows) { div.append(el('p', { class: 'empty', text: 'no empirical null fitted for the reported test' })); return; }
  const en = b.results.find(r => r.id === `en:${t}`) || {};
  div.append(el('p', { class: 'sub', text: `${words(t)}: the statistics regressed on nothing but size group, with a robust location and scale per group; ${((b.recommended || {}).reasons || {}).reference || ''}` }));
  div.append(table(rows, [['group', 'group'], ['n', 'providers'], ['min_size', 'smallest'], ['median_size', 'median size'], ['max_size', 'largest'], ['location', 'location'], ['scale', 'scale']]));
}
function modelCard(div, plot) {
  const b = state.bundle, ev = b.results.find(r => r.id === 'evaluation'), fit = b.results.find(r => r.id === 'fit');
  div.innerHTML = '';
  const s = (ev || {}).summary || {}, fs = (fit || {}).summary || {};
  div.append(el('div', { class: 'tiles' }, [
    el('div', { class: 'tile' }, [el('div', { class: 'v num', text: fmt(s.c_overall != null ? s.c_overall : fs.auc, 3) }), el('div', { class: 'k', text: 'C statistic, overall' })]),
    el('div', { class: 'tile' }, [el('div', { class: 'v num', text: fmt(s.c_within_provider, 3) }), el('div', { class: 'k', text: 'C statistic within provider' })]),
    el('div', { class: 'tile' }, [el('div', { class: 'v num', text: fmt((b.specification.adjusters || []).length) }), el('div', { class: 'k', text: 'adjusters' })]),
  ]));
  const rows = b.tables.calibration || [];
  if (!rows.length) { plot.textContent = ''; return; }
  const col = COLOR();
  const scales = Array.from(new Set(rows.map(r => r.scale)));
  const traces = scales.map((sc, i) => { const rs = rows.filter(r => r.scale === sc); return { type: 'scatter', mode: 'lines+markers', name: words(sc), x: rs.map(r => r.mean_predicted), y: rs.map(r => r.observed_rate), marker: { color: i ? col.as_expected : col.accent }, hovertemplate: 'decile %{text}: predicted %{x:.3f}, observed %{y:.3f}<extra></extra>', text: rs.map(r => r.decile) }; });
  const lo = Math.min(...rows.map(r => Math.min(r.mean_predicted, r.observed_rate))), hi = Math.max(...rows.map(r => Math.max(r.mean_predicted, r.observed_rate)));
  Plotly.newPlot(plot, traces, plotLayout({ height: 240, margin: { l: 50, r: 10, t: 10, b: 40 }, xaxis: { title: { text: 'mean predicted rate (decile)' }, gridcolor: cssVar('--line'), color: cssVar('--muted') }, yaxis: { title: { text: 'observed rate' }, gridcolor: cssVar('--line'), color: cssVar('--muted') },
    shapes: [{ type: 'line', x0: lo, y0: lo, x1: hi, y1: hi, line: { color: cssVar('--muted'), width: 1, dash: 'dot' } }], showlegend: scales.length > 1 }), PLOT_CONFIG);
}
function table(rows, cols, max) {
  const t = el('table', { class: 'data' });
  t.append(el('thead', {}, [el('tr', {}, cols.map(([k, h]) => el('th', { class: typeof (rows[0] || {})[k] === 'number' ? 'num' : '', text: h })))]));
  const tb = el('tbody');
  for (const r of rows.slice(0, max || 500)) tb.append(el('tr', {}, cols.map(([k]) => el('td', { class: typeof r[k] === 'number' ? 'num' : '', text: typeof r[k] === 'number' ? fmt(r[k]) : (r[k] == null ? '—' : String(r[k])) }))));
  t.append(tb);
  return el('div', { class: 'scroll-x' }, [t]);
}
function drawer() {
  const b = state.bundle, d = $('#drawer');
  if (!d) return;
  d.innerHTML = '';
  const pid = state.pid;
  if (!pid) {
    const rep = tableFor(b, reportedCfg(b));
    const worst = rep ? Object.keys(rep.status).filter(p => rep.status[p] === 'worse') : [];
    d.append(el('div', { class: 'card' }, [el('h3', { text: 'A provider' }), el('p', { class: 'sub', text: 'click a point in the funnel or caterpillar, or a row of the stability matrix' }),
      worst.length ? el('div', { class: 'chips' }, worst.slice(0, 12).map(p => el('button', { class: 'chip worse', type: 'button', text: p, onclick: () => { state.pid = p; drawExplorer(); } }))) : null]));
    return;
  }
  const g = (b.tables.gamma || []).find(r => String(r.provider) === pid) || {};
  const cfgs = allConfigs(b), rep = tableFor(b, reportedCfg(b)), cur = tableFor(b, state.cfg);
  const repRow = rep ? rep.rows.find(r => String(r.provider) === pid) || {} : {};
  const card = el('div', { class: 'card' });
  card.append(el('div', { style: 'display:flex;justify-content:space-between;align-items:baseline;gap:8px' }, [el('div', { class: 'pid', text: pid }), el('button', { class: 'btn small', type: 'button', text: 'clear', onclick: () => { state.pid = null; drawExplorer(); } })]));
  const st = rep ? rep.status[pid] : null;
  card.append(el('div', { class: 'chips', style: 'margin:8px 0' }, [el('span', { class: 'chip ' + (st || 'expected'), text: (STATUS_WORD[st] || 'not tested') + ' (reported)' }),
    cur && cur.key !== (rep || {}).key ? el('span', { class: 'chip ' + (cur.status[pid] || 'expected'), text: (STATUS_WORD[cur.status[pid]] || 'not tested') + ' (selected)' }) : null]));
  card.append(el('div', { class: 'tiles' }, [
    el('div', { class: 'tile' }, [el('div', { class: 'v num', text: fmt(g.n_patients) }), el('div', { class: 'k', text: 'patients' })]),
    el('div', { class: 'tile' }, [el('div', { class: 'v num', text: fmt(repRow.observed) }), el('div', { class: 'k', text: 'observed events' })]),
    el('div', { class: 'tile' }, [el('div', { class: 'v num', text: fmt(repRow.expected, 1) }), el('div', { class: 'k', text: 'expected events' })]),
    el('div', { class: 'tile' }, [el('div', { class: 'v num', text: fmt(repRow.estimate, 3) }), el('div', { class: 'k', text: 'observed / expected' })]),
  ]));
  const tr = truthOf(b, pid);
  if (tr) card.append(el('p', { class: 'truth', style: 'margin-top:10px' }, [el('span', { class: 'eyebrow', text: 'planted truth ' }), el('span', { class: 'chip ' + tr.kind, text: tr.text })]));
  const w = cfgs.filter(t => t.status[pid] === 'worse').length, bt = cfgs.filter(t => t.status[pid] === 'better').length;
  card.append(el('p', { class: 'note', style: 'margin-top:10px', text: `Flagged worse under ${w} and better under ${bt} of the ${cfgs.length} computed configurations.` }));
  card.append(el('div', { class: 'chips' }, cfgs.map(t => el('span', { class: 'chip ' + (t.status[pid] || 'expected'), title: STATUS_WORD[t.status[pid]] || 'not tested', text: t.label.replace(' null', '') }))));
  d.append(card);
  const sub = b.tables[`subset_${pid}`];
  if (sub && sub.length) {
    const rows = sub.filter(r => r.group !== 'all rows' && r.n >= 1);
    const plot = el('div', { class: 'plot js-plot', style: 'min-height:260px' });
    const bd = el('div', { class: 'card' }, [el('h3', { text: 'Why? The breakdown' }), el('p', { class: 'sub', text: 'observed over expected within each patient group of this provider; a ratio far from one in every group is a uniform departure, not a case-mix artifact' }), plot]);
    d.append(bd);
    const col = COLOR(), ed = b.specification.event_direction;
    const worseSide = r => ed === 'higher_is_worse' ? r.ratio > 1 : r.ratio < 1;
    plot._draw = () => Plotly.newPlot(plot, [{ type: 'bar', orientation: 'h', x: rows.map(r => r.ratio), y: rows.map(r => `${r.group} ${levelWord(r.level)}`), text: rows.map(r => `${fmt(r.observed)} / ${fmt(r.expected, 1)} (n ${fmt(r.n)})`),
      hovertemplate: '%{y}: ratio %{x:.2f}<br>%{text}<extra></extra>', marker: { color: rows.map(r => worseSide(r) ? col.worse : col.better), opacity: .85 } }],
      plotLayout({ height: Math.max(240, 18 * rows.length + 60), margin: { l: 120, r: 10, t: 6, b: 36 }, xaxis: { title: { text: 'observed / expected' }, gridcolor: cssVar('--line'), color: cssVar('--muted') }, yaxis: { autorange: 'reversed', tickfont: { size: 10 }, color: cssVar('--muted') },
        shapes: [{ type: 'line', yref: 'paper', y0: 0, y1: 1, x0: 1, x1: 1, line: { color: cssVar('--accent'), width: 1.2 } }], showlegend: false }), PLOT_CONFIG);
    plot._draw();
    const res = b.results.find(r => r.id === `subset:${pid}`);
    const cm = res && res.summary && res.summary.case_mix;
    if (cm && typeof cm === 'object') {
      const cmRows = Object.keys(cm).filter(k => cm[k] && typeof cm[k] === 'object').map(k => ({ variable: k, provider: cm[k].provider_mean, population: cm[k].population_mean,
        diff: cm[k].population_mean ? (cm[k].provider_mean - cm[k].population_mean) / Math.abs(cm[k].population_mean) : null }));
      d.append(el('div', { class: 'card' }, [el('h3', { text: 'Case mix against the population' }), el('p', { class: 'sub', text: 'means of the adjusters; the expected events already reflect them' }),
        table(cmRows.map(r => ({ variable: r.variable, provider: r.provider, population: r.population, 'relative difference': r.diff == null ? null : Math.round(100 * r.diff) + '%' })), [['variable', 'adjuster'], ['provider', 'this provider'], ['population', 'population'], ['relative difference', 'difference']])]));
    }
  }
  if (!(sub && sub.length) && live.sid && b.run === live.sid) {
    d.append(el('div', { class: 'card' }, [el('h3', { text: 'Why? The breakdown' }), el('p', { class: 'sub', text: 'not computed yet for this provider' }),
      el('button', { class: 'btn primary', type: 'button', text: 'Compute the breakdown (engine call)', onclick: async ev => { ev.target.disabled = true; ev.target.textContent = 'computing…'; await liveSubset(pid); } })]));
  }
  const said = (b.transcript || []).filter(m => m.role === 'agent' && m.text && m.text.includes(pid) && !/^Done\./.test(m.text));
  if (said.length) d.append(el('div', { class: 'card' }, [el('h3', { text: 'What the agent said' }), ...said.map(m => el('div', { class: 'bubble', style: 'margin-top:8px', html: bubbleHtml(m.text) }))]));
}
function levelWord(l) { const s = String(l); if (s === '1' || s === '1.0') return 'yes'; if (s === '0' || s === '0.0') return 'no'; return s; }

// ------------------------------------------------------------------ the report
function resolveRefs(text, refs, withNames) {
  return esc(text).replace(/\[([a-z_0-9:]+)\]/g, (m, k) => {
    if (!(k in refs)) return `<span class="ref" title="unknown reference">[${esc(k)}]</span>`;
    let v = refs[k];
    let s = typeof v === 'number' ? (Number.isInteger(v) ? v.toLocaleString() : fmt(v)) + (k.endsWith('_pct') ? '%' : '') : (v == null ? 'not available' : String(v));
    return `<span class="ref" title="[${esc(k)}] filled by the harness">${esc(s)}${withNames ? `<span class="nm">${esc(k)}</span>` : ''}</span>`;
  });
}
function mdToHtml(md) {
  const out = [];
  const lines = String(md || '').split('\n');
  let i = 0, inList = false, para = [];
  const flushP = () => { if (para.length) { out.push('<p>' + inline(para.join(' ')) + '</p>'); para = []; } };
  const closeList = () => { if (inList) { out.push('</ul>'); inList = false; } };
  function inline(s) {
    let t = esc(s);
    t = t.replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    return t;
  }
  while (i < lines.length) {
    const l = lines[i];
    const img = l.match(/^!\[(.*?)\]\((.*?)\)\s*$/);
    if (img) { flushP(); closeList(); const name = img[2].split('/').pop(); const f = (state.bundle.figures || {})[name];
      out.push(f ? `<div class="figwrap">${f.svg}<div class="note">${esc(img[1] || f.caption || '')}</div></div>` : `<p class="note">[figure ${esc(name)}]</p>`); i++; continue; }
    if (/^#{1,3} /.test(l)) { flushP(); closeList(); const lvl = l.match(/^(#+)/)[1].length; const tag = lvl === 1 ? 'h2' : lvl === 2 ? 'h2' : 'h3'; out.push(`<${tag}>${inline(l.replace(/^#+\s*/, ''))}</${tag}>`); i++; continue; }
    if (/^\s*[-*] /.test(l)) { flushP(); if (!inList) { out.push('<ul>'); inList = true; } out.push('<li>' + inline(l.replace(/^\s*[-*]\s*/, '')) + '</li>'); i++; continue; }
    if (/^\|/.test(l)) {
      flushP(); closeList();
      const rows = [];
      while (i < lines.length && /^\|/.test(lines[i])) { if (!/^\|\s*-+/.test(lines[i])) rows.push(lines[i].trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim())); i++; }
      if (rows.length) out.push('<div class="scroll-x"><table>' + rows.map((r, k) => '<tr>' + r.map(c => `<${k ? 'td' : 'th'}>${inline(c)}</${k ? 'td' : 'th'}>`).join('') + '</tr>').join('') + '</table></div>');
      continue;
    }
    if (!l.trim()) { flushP(); closeList(); i++; continue; }
    para.push(l.trim()); i++;
  }
  flushP(); closeList();
  return out.join('\n');
}
function renderReport() {
  const b = state.bundle, view = el('div', { class: 'view' });
  const prose = b.prose || {}, order = ['declared', 'model', 'measures', 'flags', 'alternatives'];
  const titles = { declared: 'The measure as declared', model: 'The risk-adjustment model', measures: 'Standardized measures', flags: 'Flags', alternatives: 'Flags under the other admissible choices' };
  const has = order.some(k => prose[k]);
  const proseCard = el('div', { class: 'card prose' });
  const body = el('div');
  const draw = () => { body.innerHTML = ''; for (const k of order) if (prose[k]) { body.append(el('h3', { text: titles[k] }), el('p', { html: resolveRefs(prose[k], b.references || {}, state.showRefNames) })); } };
  if (has) {
    proseCard.append(el('h2', { text: 'What the model wrote' }), el('p', { class: 'sub', text: 'the model writes with reference names where numbers belong; the harness fills every number from the engine\'s tables. Highlighted values are those references.' }));
    const tg = el('label', { class: 'toggle' }, [el('input', { type: 'checkbox', id: 'refnames', onchange: e => { state.showRefNames = e.target.checked; draw(); } }), 'show the reference names']);
    if (state.showRefNames) tg.firstChild.checked = true;
    proseCard.append(tg, body);
    draw();
    view.append(proseCard);
  }
  const rep = el('div', { class: 'card report' }, [el('h2', { text: 'The report' }), el('p', { class: 'sub', text: 'rendered by the harness from the engine\'s tables; the figures are the ones the model chose from the catalogue' })]);
  const r = el('div', { html: mdToHtml(b.report_md || '_no report_') });
  rep.append(r);
  view.append(rep);
  if (b.repro_r) view.append(el('div', { class: 'card' }, [el('h2', { text: 'The reproducing script' }), el('p', { class: 'sub', text: 'replays every table from the recorded payloads without the model' }),
    el('details', {}, [el('summary', { text: 'repro.R' }), el('pre', { class: 'mono', style: 'white-space:pre-wrap;font-size:.8rem;max-height:420px;overflow:auto', text: b.repro_r })])]));
  return view;
}

// ------------------------------------------------------------------ the audit trail
function renderAudit() {
  const b = state.bundle, view = el('div', { class: 'view' });
  const calls = b.model_calls || [];
  const byAct = {};
  for (const c of calls) { const a = byAct[c.name] = byAct[c.name] || { name: c.name, calls: 0, prompt: 0, completion: 0, seconds: 0 }; a.calls++; a.prompt += c.prompt_tokens || 0; a.completion += c.completion_tokens || 0; a.seconds += c.seconds || 0; }
  view.append(el('div', { class: 'grid two' }, [
    el('div', { class: 'card' }, [el('h2', { text: 'Provenance' }), el('dl', { class: 'kv' }, [
      el('dt', { text: 'configuration hash' }), el('dd', { class: 'mono', text: b.config_hash || '—' }),
      el('dt', { text: 'data file' }), el('dd', { class: 'mono', text: b.dataset.data_file || '' }),
      el('dt', { text: 'run' }), el('dd', { class: 'mono', text: b.run }),
      el('dt', { text: 'steps' }), el('dd', { text: `${b.steps} (cost ${b.cost_used} of ${b.budget})` }),
      el('dt', { text: 'phase' }), el('dd', { text: b.phase }),
    ])]),
    el('div', { class: 'card' }, [el('h2', { text: 'Language model calls' }), el('p', { class: 'sub', text: 'every call is one constrained generation: a policy as the system prompt, this turn\'s material as the user turn, a JSON schema the decoder must obey' }),
      table(Object.values(byAct).map(a => ({ act: a.name, calls: a.calls, 'prompt tokens': a.prompt, 'completion tokens': a.completion, seconds: Math.round(a.seconds) })), [['act', 'act'], ['calls', 'calls'], ['prompt tokens', 'prompt tokens'], ['completion tokens', 'completion tokens'], ['seconds', 'seconds']])]),
  ]));
  view.append(el('div', { class: 'card' }, [el('h2', { text: 'Notes and refusals recorded in this run' }), (b.refusals || []).length ? el('ul', {}, b.refusals.map(x => el('li', { text: x }))) : el('p', { class: 'empty', text: 'none' })]));
  view.append(el('div', { class: 'grid two' }, [
    el('div', { class: 'card' }, [el('h2', { text: 'Rule thresholds' }), el('dl', { class: 'kv' }, Object.keys(b.thresholds || {}).flatMap(k => [el('dt', { class: 'mono', text: k }), el('dd', { text: String(b.thresholds[k]) })]))]),
    el('div', { class: 'card' }, [el('h2', { text: 'The closed action set' }), el('ul', {}, (b.actions || []).map(a => el('li', {}, [el('b', { class: 'mono', text: a.name }), ` (cost ${a.cost}): ${a.purpose}`])))]),
  ]));
  const spec = b.specification || {};
  view.append(el('div', { class: 'card' }, [el('h2', { text: 'Specification as verified' }), el('pre', { class: 'mono', style: 'white-space:pre-wrap;font-size:.8rem', text: JSON.stringify(spec, null, 1) })]));
  return view;
}

// ------------------------------------------------------------------ live mode (server.py on the same origin)
const live = { api: null, sid: null, timer: null, lastLog: 0 };
async function detectLive() {
  try {
    const r = await fetch('api/datasets', { cache: 'no-store' });
    if (!r.ok) return;
    live.api = await r.json();
    const badge = $('#mode');
    badge.textContent = live.api.model ? `live · ${live.api.model}` : 'live · harness only';
    badge.classList.add('live');
    $('#live-controls').hidden = false;
  } catch (e) { /* static hosting: replay only */ }
}
function liveForm() {
  const main = $('#main');
  main.innerHTML = '';
  const sel = el('select', { id: 'live-dataset' }, (live.api.datasets || []).map(d => el('option', { value: d.name, text: d.name })));
  const file = el('input', { type: 'file', id: 'live-file', accept: '.csv,.parquet', class: 'input' });
  const card = el('div', { class: 'card' }, [
    el('h2', { text: 'A new live analysis' }),
    el('p', { class: 'sub', text: 'the agent profiles the file, asks what the data cannot settle, and runs the analysis with the model as planner; every step is shown as it happens' }),
    el('div', { class: 'controls' }, [el('label', {}, ['example data set', sel]), el('label', {}, ['or upload a measure file', file]),
      el('button', { class: 'btn primary', type: 'button', text: 'Start', onclick: async () => { try { await startLive(sel.value, file.files[0]); } catch (e) { showError(e); } } })]),
  ]);
  main.append(card);
}
async function startLive(dataset, upload) {
  let r;
  if (upload) { const fd = new FormData(); fd.append('file', upload); r = await fetch('api/sessions/upload', { method: 'POST', body: fd }); }
  else r = await fetch('api/sessions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ dataset }) });
  if (!r.ok) throw new Error(await r.text());
  const j = await r.json();
  live.sid = j.id; live.lastLog = 0;
  const opt = el('option', { value: `api/sessions/${j.id}/bundle`, text: `live · ${j.id}` });
  $('#run').append(opt); $('#run').value = opt.value;
  await refreshLive();
  showTab('conversation', true);
}
async function refreshLive() {
  const r = await fetch(`api/sessions/${live.sid}/bundle`, { cache: 'no-store' });
  if (!r.ok) throw new Error(await r.text());
  state.bundle = await r.json();
  state.bundle.run = live.sid;
  state.cfg = state.bundle.recommended && state.bundle.recommended.test ? reportedCfg(state.bundle) : state.cfg;
  state.rendered = {};
}
function composer() {
  const ta = el('textarea', { id: 'composer', class: 'input', rows: 3, placeholder: 'Declare the measure in a sentence, answer by number, or ask about the result…', style: 'width:100%;resize:vertical' });
  const status = el('div', { class: 'note', id: 'live-status' });
  const logbox = el('pre', { class: 'mono', id: 'live-log', style: 'white-space:pre-wrap;font-size:.75rem;max-height:160px;overflow:auto;margin:0;color:var(--muted)' });
  const send = el('button', { class: 'btn primary', type: 'button', text: 'Send', onclick: async () => {
    const text = ta.value.trim(); if (!text) return;
    ta.value = ''; send.disabled = true;
    const chat = $('.chat'); chat.append(messageNode({ role: 'analyst', text, figures: [] }));
    const typing = el('div', { class: 'msg agent' }, [el('div', { class: 'who', text: 'pprof agent' }), el('div', { class: 'bubble' }, [el('span', { class: 'typing' }, [el('i'), el('i'), el('i')])])]);
    chat.append(typing);
    try {
      const r = await fetch(`api/sessions/${live.sid}/messages`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text }) });
      if (!r.ok) throw new Error(await r.text());
      pollLive(typing, send, status, logbox);
    } catch (e) { typing.remove(); status.textContent = e.message; send.disabled = false; }
  } });
  const card = el('div', { class: 'card' }, [el('div', { style: 'display:grid;gap:8px' }, [ta, el('div', { style: 'display:flex;gap:10px;align-items:center;flex-wrap:wrap' }, [send, status]), logbox])]);
  if (live.api && live.api.datasets) {
    const d = live.api.datasets.find(x => x.name === (state.bundle.dataset || {}).name);
    if (d && d.scenarios && d.scenarios.demo) {
      const lines = d.scenarios.demo.split('\n').map(l => l.trim()).filter(l => l && !l.startsWith('#'));
      card.append(el('div', { class: 'chips', style: 'margin-top:10px' }, lines.map(l => el('button', { class: 'chip', type: 'button', title: l, text: l.length > 60 ? l.slice(0, 57) + '…' : l, onclick: () => { ta.value = l; ta.focus(); } }))));
    }
  }
  return card;
}
function pollLive(typing, send, status, logbox) {
  if (live.timer) clearTimeout(live.timer);
  const tick = async () => {
    try {
      const r = await fetch(`api/sessions/${live.sid}`, { cache: 'no-store' });
      const st = await r.json();
      const log = st.log || [];
      if (log.length > live.lastLog) { logbox.textContent = log.slice(-12).join('\n'); live.lastLog = log.length; logbox.scrollTop = logbox.scrollHeight; }
      status.textContent = st.busy ? `working… ${st.results} results, ${st.model_calls} model calls, ${st.elapsed} s` : (st.error ? `error: ${st.error}` : 'ready');
      if (st.busy) { live.timer = setTimeout(tick, 2000); return; }
      await refreshLive();
      showTab(state.tab, true);
    } catch (e) { status.textContent = e.message; if (typing) typing.remove(); send.disabled = false; }
  };
  tick();
}
async function liveSubset(pid) {
  const r = await fetch(`api/sessions/${live.sid}/subset/${encodeURIComponent(pid)}`, { method: 'POST' });
  if (!r.ok) { showError(new Error(await r.text())); return; }
  await refreshLive();
  showTab('explorer', true);
}

// ------------------------------------------------------------------ boot
async function boot() {
  for (const t of $$('.tab')) t.addEventListener('click', () => showTab(t.dataset.tab));
  window.addEventListener('hashchange', () => { const h = location.hash.slice(1); if (TABS.includes(h) && h !== state.tab) showTab(h); });
  $('#dataset').addEventListener('change', () => { fillRuns(); loadRun($('#run').value).catch(showError); });
  $('#run').addEventListener('change', () => { const v = $('#run').value; if (v.startsWith('api/')) { refreshLive().then(() => showTab(state.tab, true)).catch(showError); } else loadRun(v).catch(showError); });
  $('#live-new').addEventListener('click', liveForm);
  window.addEventListener('resize', () => { for (const p of $$('.js-plot')) if (p.data) Plotly.Plots.resize(p); });
  const h = location.hash.slice(1);
  if (TABS.includes(h)) state.tab = h;
  await detectLive();
  try {
    await loadIndex();
    await loadRun($('#run').value);
  } catch (e) { if (live.api) liveForm(); else showError(e); }
}
function showError(e) { $('#main').innerHTML = `<div class="card"><h2>Could not load</h2><p>${esc(e.message || e)}</p><p class="note">Serve the folder over HTTP (for example <code>python -m http.server</code>) so the app can fetch its data files.</p></div>`; }
document.addEventListener('DOMContentLoaded', boot);
})();
