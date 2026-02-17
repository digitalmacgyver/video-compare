#!/usr/bin/env python3
"""Generate a cross-clip comparison HTML report from multiple quality report JSONs.

Reads two or more quality_report.json files (produced by quality_report.py) and
generates an interactive HTML report comparing device performance across clips.

Usage:
    python cross_clip_report.py <json1> <json2> [json3...] --output PATH

Examples:
    python cross_clip_report.py clip02.json clip03.json clip04.json --output comparison.html
    python cross_clip_report.py /path/to/*_quality_report.json --output /path/to/comparison.html
"""

import argparse
import json
import os
import sys
import numpy as np

# =====================================================================
# METRIC METADATA (shared with quality_report.py)
# =====================================================================

ALL_KEYS = [
    "sharpness", "edge_strength", "blocking", "detail", "texture_quality",
    "ringing", "temporal_stability", "colorfulness", "naturalness",
    "crushed_blacks", "blown_whites",
]

METRIC_INFO = {
    "sharpness":          ("Sharpness",      "Laplacian CV\u00b2",   True),
    "edge_strength":      ("Edge Strength",  "Sobel norm.",          True),
    "blocking":           ("Blocking",       "8x8 grid ratio",      None),
    "detail":             ("Detail",         "local CV median",      True),
    "texture_quality":    ("Texture Q",      "structure ratio",      True),
    "ringing":            ("Ringing",        "edge overshoot norm.", False),
    "temporal_stability": ("Temporal",       "frame diff norm.",     False),
    "colorfulness":       ("Colorfulness",   "Hasler-S. M",          True),
    "naturalness":        ("Naturalness",    "MSCN kurtosis",        True),
    "crushed_blacks":     ("Crushed Blacks", "shadow headroom",     False),
    "blown_whites":       ("Blown Whites",   "highlight headroom",  False),
}

COLORS_7  = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#42d4f4", "#f58231", "#911eb4"]
COLORS_14 = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
             "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324", "#800000"]

CLIP_COLORS = ["#58a6ff", "#3fb950", "#f58231", "#e6194b", "#911eb4",
               "#42d4f4", "#ffe119", "#f032e6", "#bfef45", "#dcbeff"]

# =====================================================================
# COMPOSITE SCORING
# =====================================================================

def compute_composites(data):
    """Compute rank-based overall composite scores.

    For each metric, clips are ranked 1 (best) to N (worst). The overall
    score is the average rank across all metrics — lower is better.
    Z-scores are also computed for heatmap cell coloring (positive = good).
    """
    clip_names = sorted(data.keys())
    n = len(clip_names)
    # Only use metrics present in all clips (backward compat with older JSONs)
    available = set(ALL_KEYS)
    for c in clip_names:
        available &= set(data[c].keys())
    keys = [k for k in ALL_KEYS if k in available]

    zscores = {}
    for key in keys:
        arr = np.array([data[c][key]["mean"] for c in clip_names])
        mu, sigma = np.mean(arr), np.std(arr)
        zs = (arr - mu) / sigma if sigma > 1e-15 else np.zeros(n)
        _, _, hb = METRIC_INFO[key]
        if hb is False:
            zs = -zs
        elif hb is None:
            dist = np.abs(arr - 1.0)
            d_mu, d_sigma = np.mean(dist), np.std(dist)
            zs = -(dist - d_mu) / d_sigma if d_sigma > 1e-15 else np.zeros(n)
        zscores[key] = zs

    ranks = {}
    for key in keys:
        arr = np.array([data[c][key]["mean"] for c in clip_names])
        _, _, hb = METRIC_INFO[key]
        if hb is True:
            order = np.argsort(-arr)
        elif hb is False:
            order = np.argsort(arr)
        else:
            order = np.argsort(np.abs(arr - 1.0))
        rank_arr = np.empty(n, dtype=float)
        for r, idx in enumerate(order, 1):
            rank_arr[idx] = r
        ranks[key] = rank_arr

    avg_ranks = np.mean([ranks[k] for k in keys], axis=0)

    return {c: {"overall": float(avg_ranks[i]),
                "ranks": {k: int(ranks[k][i]) for k in keys},
                "zscores": {k: float(zscores[k][i]) for k in keys}}
            for i, c in enumerate(clip_names)}


# =====================================================================
# CSS (shared dark theme)
# =====================================================================

HTML_CSS = """
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #e6edf3; --text-dim: #8b949e; --accent: #58a6ff;
    --good: #3fb950; --bad: #f85149; --mid: #d29922;
    --accent2: #d2a8ff;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6; padding: 24px; max-width: 1400px; margin: 0 auto;
  }
  h1 { font-size: 1.8em; margin-bottom: 8px; }
  h2 { font-size: 1.3em; margin: 40px 0 16px; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 8px; }
  h2.alt { color: var(--accent2); }
  h3 { font-size: 1.1em; margin: 20px 0 12px; color: var(--text-dim); }
  .subtitle { color: var(--text-dim); margin-bottom: 24px; font-size: 0.95em; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 24px; }
  .chart-container { position: relative; width: 100%; }
  .chart-wide { height: 520px; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }
  .heatmap { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  .heatmap th { background: #21262d; padding: 6px 5px; text-align: center; font-weight: 600;
    border: 1px solid var(--border); white-space: nowrap; position: sticky; top: 0; z-index: 2;
    cursor: pointer; user-select: none; }
  .heatmap th:hover { background: #2d333b; }
  .heatmap th::after { content: ' \\2195'; opacity: 0.3; font-size: 0.8em; }
  .heatmap th.sort-asc::after { content: ' \\2191'; opacity: 0.8; }
  .heatmap th.sort-desc::after { content: ' \\2193'; opacity: 0.8; }
  .heatmap th:first-child { text-align: left; min-width: 140px; }
  .heatmap th { border-bottom: 2px solid var(--accent); }
  .heatmap td { padding: 5px 6px; text-align: center; border: 1px solid var(--border);
    font-variant-numeric: tabular-nums; white-space: nowrap; font-size: 0.92em; }
  .heatmap td:first-child { text-align: left; font-weight: 500; }
  .heatmap tr:hover { outline: 2px solid var(--accent); }
  .legend-note { font-size: 0.85em; color: var(--text-dim); margin-top: 12px; }
  .clip-tag { display: inline-block; font-size: 0.75em; padding: 2px 8px; border-radius: 4px;
    font-weight: 600; letter-spacing: 0.5px; margin-right: 8px; }
"""


# =====================================================================
# HTML GENERATION
# =====================================================================

def generate_crossclip_html(all_data, clip_labels):
    """Generate a cross-clip comparison HTML report.

    Args:
        all_data: dict mapping clip_label -> {device_name -> metric_data}
        clip_labels: list of clip labels in order
    """
    # Validate: all clips must have the same device names
    device_sets = [set(all_data[c].keys()) for c in clip_labels]
    common_devices = device_sets[0]
    for ds in device_sets[1:]:
        common_devices &= ds
    if not common_devices:
        print("ERROR: No common device names found across JSON files.")
        sys.exit(1)

    all_devices = set()
    for c in clip_labels:
        all_devices |= set(all_data[c].keys())

    # Warn about non-common devices
    non_common = all_devices - common_devices
    if non_common:
        print(f"WARNING: {len(non_common)} device(s) not in all clips, using only common: {sorted(common_devices)}")

    device_names = sorted(common_devices)
    n_clips = len(clip_labels)
    n_devices = len(device_names)

    # Only use metrics present in all devices across all clips
    avail_keys = set(ALL_KEYS)
    for c in clip_labels:
        for d in device_names:
            avail_keys &= set(all_data[c][d].keys())
    active_keys = [k for k in ALL_KEYS if k in avail_keys]

    # Assign colors to clips and devices
    clip_colors = {c: CLIP_COLORS[i % len(CLIP_COLORS)] for i, c in enumerate(clip_labels)}

    # Compute composites per clip
    all_composites = {}
    for clip in clip_labels:
        # Filter to common devices only for fair comparison
        filtered = {d: all_data[clip][d] for d in device_names}
        all_composites[clip] = compute_composites(filtered)

    # Average overall score per device
    avg_overall = {}
    for dev in device_names:
        scores = [all_composites[clip][dev]["overall"] for clip in clip_labels]
        avg_overall[dev] = float(np.mean(scores))
    ranked_devices = sorted(device_names, key=lambda d: avg_overall[d])

    palette = COLORS_7 if n_devices <= 7 else COLORS_14
    dev_colors = {d: palette[i % len(palette)] for i, d in enumerate(ranked_devices)}

    # Bar chart height scales with device count
    bar_height = max(320, n_devices * 36 + 80)

    # Grouped bar chart datasets
    datasets_grouped = []
    for clip in clip_labels:
        datasets_grouped.append({
            "label": clip,
            "data": [all_composites[clip][d]["overall"] for d in ranked_devices],
            "backgroundColor": clip_colors[clip] + "cc",
            "borderColor": clip_colors[clip],
            "borderWidth": 1, "borderRadius": 3
        })

    # Consistency table: average rank per metric across clips
    avg_ranks = {}
    for dev in device_names:
        avg_ranks[dev] = {}
        for key in active_keys:
            _, _, higher_better = METRIC_INFO[key]
            ranks = []
            for clip in clip_labels:
                items = [(d, all_data[clip][d][key]["mean"]) for d in device_names]
                if higher_better is None:
                    items.sort(key=lambda x: abs(x[1] - 1.0))
                elif higher_better:
                    items.sort(key=lambda x: -x[1])
                else:
                    items.sort(key=lambda x: x[1])
                rank = [d for d, _ in items].index(dev) + 1
                ranks.append(rank)
            avg_ranks[dev][key] = float(np.mean(ranks))

    consistency_data = []
    for dev in ranked_devices:
        cells = [{"rank": avg_ranks[dev][key], "key": key} for key in active_keys]
        consistency_data.append({"name": dev, "avg_overall": avg_overall[dev], "cells": cells})

    # Build HTML
    clip_list_str = ", ".join(clip_labels)

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cross-Clip Quality Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>{css}</style>
</head>
<body>

<h1>Cross-Clip Quality Comparison</h1>
<p class="subtitle">
  Performance of {n_dev} capture devices across {n_clip} clips ({clip_list}).
  Each clip was independently normalized and evaluated on 9 no-reference metrics.
</p>

<h2>Overall Ranking by Clip</h2>
<div class="card">
  <div class="chart-container" style="height:{bar_h}px;">
    <canvas id="crossOverall"></canvas>
  </div>
  <p class="legend-note">Grouped bar chart showing each device's average rank per clip (lower = better). Devices sorted by best average rank.</p>
</div>

<h2>Average Rank (Across All Clips)</h2>
<div class="card">
  <div class="chart-container" style="height:{bar_h}px;">
    <canvas id="avgOverall"></canvas>
  </div>
  <p class="legend-note">Mean of average ranks across {clip_list}. Lower = better.</p>
</div>

<h2>Metric Rankings — Consistency Across Clips</h2>
<div class="card" style="overflow-x:auto;">
  <table class="heatmap" id="consistencyTable">
    <thead>
      <tr>
        <th>Device</th><th>Avg Rank</th>
        {th_metrics}
      </tr>
    </thead>
    <tbody></tbody>
  </table>
  <p class="legend-note">Cells show average rank across all {n_clip} clips (1 = best). Color: green = consistently good, red = consistently poor. Click any column header to sort.</p>
</div>

<script>
function hbar(id, labels, scores, clrs, xlabel) {{
  new Chart(document.getElementById(id), {{
    type:'bar', data:{{ labels, datasets:[{{ data:scores, backgroundColor:clrs, borderColor:clrs, borderWidth:1, borderRadius:4 }}] }},
    options:{{ indexAxis:'y', responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{display:false}}, tooltip:{{callbacks:{{label:c=>'avg rank: '+c.parsed.x.toFixed(1)}}}} }},
      scales:{{ x:{{grid:{{color:'#30363d'}},ticks:{{color:'#8b949e'}},title:{{display:!!xlabel,text:xlabel||'',color:'#8b949e'}}}}, y:{{grid:{{display:false}},ticks:{{color:'#e6edf3',font:{{size:11}}}}}} }}
    }}
  }});
}}

new Chart(document.getElementById('crossOverall'), {{
  type: 'bar',
  data: {{
    labels: {ranked_dev_json},
    datasets: {datasets_grouped_json}
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ labels: {{ color: '#e6edf3' }} }},
      tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': avg rank ' + ctx.parsed.y.toFixed(1) }} }}
    }},
    scales: {{
      x: {{ grid: {{ display: false }}, ticks: {{ color: '#e6edf3' }} }},
      y: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }}, title: {{ display: true, text: 'Average Rank (lower = better)', color: '#8b949e' }} }}
    }}
  }}
}});

hbar('avgOverall', {ranked_dev_json}, {avg_scores_json}, {dev_colors_json}, 'Average Rank (lower = better)');

const consistData = {consistency_json};
const ctbody = document.querySelector('#consistencyTable tbody');
consistData.forEach(row => {{
  const tr = document.createElement('tr');
  let td = document.createElement('td');
  td.textContent = row.name;
  tr.appendChild(td);
  td = document.createElement('td');
  td.textContent = row.avg_overall.toFixed(1);
  const mid = ({n_dev} + 1) / 2;
  const goodness = (mid - row.avg_overall) / mid;
  td.style.background = goodness > 0 ? 'rgba(63,185,80,' + (Math.abs(goodness)*0.5) + ')' : 'rgba(248,81,73,' + (Math.abs(goodness)*0.5) + ')';
  td.style.fontWeight = '600';
  tr.appendChild(td);
  row.cells.forEach(cell => {{
    const td = document.createElement('td');
    td.textContent = cell.rank.toFixed(1);
    const goodness = ({n_dev} + 1 - cell.rank) / {n_dev};
    td.style.background = goodness > 0.5 ? 'rgba(63,185,80,' + ((goodness-0.5)*0.6) + ')' : 'rgba(248,81,73,' + ((0.5-goodness)*0.6) + ')';
    tr.appendChild(td);
  }});
  ctbody.appendChild(tr);
}});

document.querySelectorAll('.heatmap th').forEach((th,colIdx)=>{{
  th.addEventListener('click',()=>{{
    const table=th.closest('table'),tbody=table.querySelector('tbody');
    const rows=Array.from(tbody.querySelectorAll('tr'));
    const asc=!th.classList.contains('sort-asc');
    table.querySelectorAll('th').forEach(h=>h.classList.remove('sort-asc','sort-desc'));
    th.classList.add(asc?'sort-asc':'sort-desc');
    rows.sort((a,b)=>{{
      const av=a.cells[colIdx].textContent,bv=b.cells[colIdx].textContent;
      const an=parseFloat(av),bn=parseFloat(bv);
      if(!isNaN(an)&&!isNaN(bn)) return asc?an-bn:bn-an;
      return asc?av.localeCompare(bv):bv.localeCompare(av);
    }});
    rows.forEach(r=>tbody.appendChild(r));
  }});
}});
</script>
</body>
</html>""".format(
        css=HTML_CSS,
        n_dev=n_devices,
        n_clip=n_clips,
        clip_list=clip_list_str,
        bar_h=bar_height,
        th_metrics="".join(
            '<th>{}</th>'.format(METRIC_INFO[k][0]) for k in active_keys
        ),
        ranked_dev_json=json.dumps(ranked_devices),
        datasets_grouped_json=json.dumps(datasets_grouped),
        avg_scores_json=json.dumps([avg_overall[d] for d in ranked_devices]),
        dev_colors_json=json.dumps([dev_colors[d] for d in ranked_devices]),
        consistency_json=json.dumps(consistency_data),
    )

    return html


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-clip comparison HTML from multiple quality report JSONs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("jsons", nargs="+", help="Two or more quality_report.json files")
    parser.add_argument("--output", "-o", required=True, help="Output HTML file path")
    args = parser.parse_args()

    if len(args.jsons) < 2:
        print("ERROR: Need at least 2 JSON files for cross-clip comparison.")
        sys.exit(1)

    # Load JSON files, using filename stem as clip label
    all_data = {}
    clip_labels = []
    for json_path in args.jsons:
        if not os.path.exists(json_path):
            print(f"ERROR: File not found: {json_path}")
            sys.exit(1)
        label = os.path.splitext(os.path.basename(json_path))[0]
        # Strip common suffixes for cleaner labels
        for suffix in ["_quality_report", "_report", "_normalized"]:
            label = label.replace(suffix, "")
        with open(json_path) as f:
            data = json.load(f)
        # Remove n_frames key from device data (not a metric)
        for dev in data:
            data[dev].pop("n_frames", None)
        all_data[label] = data
        clip_labels.append(label)

    print(f"Loaded {len(clip_labels)} reports: {', '.join(clip_labels)}")

    # Check device overlap
    for label in clip_labels:
        devices = sorted(all_data[label].keys())
        print(f"  {label}: {len(devices)} devices — {', '.join(devices)}")

    html = generate_crossclip_html(all_data, clip_labels)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
