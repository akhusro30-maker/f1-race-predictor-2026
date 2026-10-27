"""Build a standalone website: site/index.html with the data and d3 built in.

    python -m f1dash.site

Open site/index.html in a browser (double-click works, no server needed), or publish
the site/ folder with GitHub Pages. Run after `f1dash.build` and `f1dash.export`.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "dashboard"
SITE = ROOT / "site"

TOKENS = """
:root {
  color-scheme: light dark;
  --font-anthropic-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-anthropic-serif: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
  --cds-font-size-caption: 0.8125rem; --cds-font-size-body: 0.9375rem; --cds-font-size-heading: 1.0625rem; --cds-font-size-title: 1.75rem;
  --cds-font-weight-medium: 600;
  --cds-gap-xs: 4px; --cds-gap-sm: 8px; --cds-gap-md: 16px; --cds-gap-lg: 24px; --cds-gap-xl: 32px;
  --cds-pad-xs: 4px; --cds-pad-sm: 8px; --cds-pad-md: 12px; --cds-pad-lg: 18px; --cds-pad-xl: 28px;
  --cds-radius: 10px; --cds-dur-slow: 450ms; --cds-ease-out: cubic-bezier(.2,.7,.2,1);
  --color-bg: #f6f5f1; --color-panel: #ffffff; --color-fg: #1d1d1b; --color-fg-muted: #66655f; --color-border-line: #e3e1da;
  --color-ok: #17803d; --color-warn: #a16207; --color-bad: #c42b1c;
  --cds-chart-muted: #c9c7bf; --cds-chart-grid: #ecebe5; --cds-chart-axis: #b5b3ab; --cds-chart-reference: #8f8d86;
  --cds-surface-popover: #ffffff; --cds-border: #dcdad2; --cds-shadow-popover: 0 4px 16px rgba(0,0,0,.12); --cds-text-secondary: #66655f;
  --s1: #2563c9; --s2: #c2410c; --s3: #0f8a6a; --s4: #7c4dcc; --s5: #c0267a; --s6: #0e7c9a; --s7: #6b8e1e; --s8: #8a6d3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --color-bg: #141413; --color-panel: #1e1e1c; --color-fg: #ecebe6; --color-fg-muted: #a3a29b; --color-border-line: #33322e;
    --color-ok: #4ade80; --color-warn: #facc15; --color-bad: #f87171;
    --cds-chart-muted: #55544e; --cds-chart-grid: #2a2a27; --cds-chart-axis: #4a4944; --cds-chart-reference: #85847d;
    --cds-surface-popover: #262624; --cds-border: #3a3935; --cds-shadow-popover: 0 4px 16px rgba(0,0,0,.5); --cds-text-secondary: #a3a29b;
    --s1: #5b8ee6; --s2: #de7432; --s3: #25a67f; --s4: #9a7ae4; --s5: #e0587e; --s6: #4cc0de; --s7: #a6c95a; --s8: #c9a46a;
  }
}
html { background: var(--color-bg); }
body { margin: 0; background: var(--color-bg); color: var(--color-fg); font-family: var(--font-anthropic-sans); -webkit-font-smoothing: antialiased; }
#dash-root { max-width: 1240px; margin-inline: auto; padding: 28px 16px 8px; box-sizing: border-box; }
#dash-root *, #dash-root *::before, #dash-root *::after { box-sizing: border-box; }
.site-foot { max-width: 1240px; margin: 8px auto 32px; padding: 0 16px; color: var(--color-fg-muted); font-size: 0.8125rem; }
.site-foot a { color: inherit; }
.dash-skeleton { color: transparent !important; background: var(--color-border-line); border-radius: 6px; }
"""

SHIM = """
(function () {
  // Stand-in for the hosted dashboard runtime: same `dash` API, data built into the page.
  var D = window.F1_DATA, calcs = {}, listeners = [], defaults = D.params, params = Object.assign({}, defaults);
  function ok(rows) { return { status: 'ok', data: rows, rows: rows, columns: Object.keys(rows[0] || {}), refreshing: false, meta: [] }; }
  function data(id) {
    if (Object.prototype.hasOwnProperty.call(D.sets, id)) return ok(D.sets[id]);
    var c = calcs[id];
    if (c) { if (!c.done) { c.value = c.fn.apply(null, c.inputs.map(function (i) { return data(i).data; })); c.done = true; } return ok(c.value); }
    return { status: 'missing', data: [], rows: [], columns: [], refreshing: false, meta: [] };
  }
  function fire() { listeners.forEach(function (f) { try { f(); } catch (e) { console.error(e); } }); }
  window.dash = {
    data: data,
    calc: function (id, o) { calcs[id] = { inputs: o.inputs, fn: o.fn }; },
    loader: function () {},
    onData: function (f) { listeners.push(f); try { f(); } catch (e) { console.error(e); } },
    params: function () { return Object.assign({}, params); },
    setParams: function (p) { Object.assign(params, p); fire(); },
    resetParams: function () { params = Object.assign({}, defaults); fire(); },
    setLink: function (n) { try { history.replaceState(null, '', '#' + n); } catch (e) {} },
    refresh: function () {},
    colors: ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', 'var(--s6)', 'var(--s7)', 'var(--s8)'],
  };
  var w = window.innerWidth, t;
  window.addEventListener('resize', function () {
    clearTimeout(t);
    t = setTimeout(function () { if (window.innerWidth !== w) { w = window.innerWidth; fire(); } }, 150);
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', fire);
})();
"""


def _js_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def main():
    sets = {p.stem: json.loads(p.read_text()) for p in sorted((DASH / "datasets").glob("*.json"))}
    meta = json.loads((DASH / "data.json").read_text())
    page = (DASH / "index.html").read_text()
    d3 = (SITE / "vendor" / "d3.min.js").read_text()
    payload = {"sets": sets, "params": {"round": sets["races"][0]["round"]}}
    generated = meta["generated"][:10]
    through = meta["data_through"]
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>F1 Race Predictor {meta['season']}</title>
<meta name="description" content="Machine-learning predictions for the remaining {meta['season']} Formula 1 races: win, pole and podium chances, strategy and title maths.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' rx='3' fill='%23c42b1c'/%3E%3Cpath d='M3 11h10M3 8h7M3 5h10' stroke='white' stroke-width='1.6'/%3E%3C/svg%3E">
<style>{TOKENS}</style>
<script>{d3}</script>
<script>window.F1_DATA = {_js_json(payload)};</script>
<script>{SHIM}</script>
</head>
<body>
<div id="dash-root">
{page}
</div>
<p class="site-foot">Data through round {through['round']} of {meta['season']}, built {generated}. Results data:
<a href="https://github.com/f1db/f1db">F1DB</a> (CC BY 4.0). Charts: <a href="https://d3js.org">d3</a>.
Not affiliated with Formula 1 or the FIA.</p>
</body>
</html>
"""
    SITE.mkdir(exist_ok=True)
    out = SITE / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB). Open it in a browser.")


if __name__ == "__main__":
    main()
