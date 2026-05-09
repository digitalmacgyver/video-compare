"""HTML, text, and screenshot rendering for the quality report.

Generates the self-contained HTML report (with embedded CSS, Chart.js
references, comparison frames, and sample screenshots) and the
optional plain-text summary. Frame extraction via ffmpeg lives here
because it's only used for HTML output.

Pure presentation — no metric algorithms.
"""

import base64
import json
import re
import subprocess
import numpy as np

from common import ALL_KEYS, METRIC_INFO, COLORS_7, COLORS_14, compute_composites
from metrics import PCT_KEYS


CMP_ICON_SVG = ('<svg viewBox="0 0 24 24" width="20" height="20" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round">'
    '<rect x="2" y="3" width="20" height="18" rx="2"/>'
    '<line x1="12" y1="3" x2="12" y2="21"/>'
    '<text x="7" y="15.5" text-anchor="middle" fill="currentColor" stroke="none" '
    'font-size="8" font-family="sans-serif" font-weight="bold">A</text>'
    '<text x="17" y="15.5" text-anchor="middle" fill="currentColor" stroke="none" '
    'font-size="8" font-family="sans-serif" font-weight="bold">B</text></svg>')


METRIC_GUIDE_TEXT = {
    "sharpness": "Laplacian variance — blur vs. sharpness",
    "edge_strength": "Sobel gradient — edge definition",
    "blocking": "8x8 DCT boundary ratio",
    "detail": "Local variance — micro-detail",
    "detail_perceptual": "Perceptual detail tuned for SD analog VHS (not validated for HD digital sources)",
    "detail_tenengrad": "Robust Tenengrad — edge detail (noise-suppressed)",
    "detail_sml": "Robust modified Laplacian — fine transition clarity",
    "detail_blur_inv": "Inverse blur effect — directional blur resistance",
    "texture_quality": "Structure/noise ratio — detail quality",
    "ringing": "Edge overshoot / haloing",
    "temporal_stability": "Frame-to-frame diff — flicker",
    "colorfulness": "Hasler-Süstrunk — color vibrancy",
    "naturalness": "MSCN kurtosis — natural signal statistics",
    "crushed_blacks": "Shadow headroom — clipped near-black fraction",
    "blown_whites": "Highlight headroom — clipped near-white fraction",
}


# =====================================================================
# FORMATTING
# =====================================================================

def short_metric_header(key):
    """Return compact table header text for a metric key."""
    label = METRIC_INFO[key][0]
    compact = re.sub(r"[^A-Za-z0-9]+", "", label)
    return compact[:8] if compact else key[:8]


def fmt_metric_value(key, value):
    """Format one metric value for text/HTML tables."""
    if key in PCT_KEYS:
        return f"{value*100:.1f}%"
    if key == "blocking":
        return f"{value:.4f}"
    if key == "colorfulness":
        return f"{value:.1f}"
    if key == "naturalness":
        return f"{value:.3f}"
    return f"{value:.6f}"


# =====================================================================
# FRAME EXTRACTION (for HTML comparison galleries)
# =====================================================================

def extract_frame_jpeg(filepath, frame_num):
    """Extract a single frame as JPEG bytes using ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", filepath,
         "-vf", f"select=eq(n\\,{frame_num})", "-vframes", "1",
         "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "2", "pipe:1"],
        capture_output=True
    )
    return proc.stdout if proc.returncode == 0 else b""


def find_comparison_frames(clip_paths, clip_names, all_perframe, all_results, skip_offsets, metric_keys):
    """Find a representative frame for each metric and extract screenshots.

    For each metric M:
      a) Find the clip V whose overall M score is best.
      b) From V, find the frame where M is near its 90th percentile — a frame
         where this metric is clearly demonstrated (not just average).
      c) Extract that frame from ALL clips for side-by-side comparison.

    skip_offsets: dict {clip_name: int} — frames skipped at start (added to ffmpeg frame number).
    metric_keys: ordered list of metrics to include in comparison galleries.

    Returns dict: {metric_key: {"frame_num": int, "best_clip": str,
                                 "frames": {clip_name: {"jpeg_b64": str, "value": float}}}}
    """
    comparisons = {}
    for key in metric_keys:
        _, _, higher_better = METRIC_INFO[key]

        # (a) Find the clip with the best overall score for this metric
        best_clip = None
        best_mean = None
        for name in clip_names:
            val = all_results[name][key]["mean"]
            if best_mean is None:
                best_clip, best_mean = name, val
            elif higher_better is True and val > best_mean:
                best_clip, best_mean = name, val
            elif higher_better is False and val < best_mean:
                best_clip, best_mean = name, val
            elif higher_better is None and abs(val - 1.0) < abs(best_mean - 1.0):
                best_clip, best_mean = name, val

        # (b) From the best clip, find the frame closest to p90 of the raw value.
        # This picks frames where the metric is clearly demonstrated, not just average.
        # For "lower is better" metrics (noise, etc.) p90 shows a challenging frame
        # from the best clip — more revealing for quality comparison.
        arr = all_perframe[best_clip][key]
        target_val = float(np.percentile(arr, 90))
        frame_idx = int(np.argmin(np.abs(arr - target_val)))

        # (c) Extract that frame from all clips
        frames = {}
        for name in clip_names:
            path = clip_paths[name]
            # Add back the skip offset so ffmpeg extracts the correct source frame
            actual_frame = frame_idx + skip_offsets.get(name, 0)
            jpeg_data = extract_frame_jpeg(path, actual_frame)
            pf_arr = all_perframe[name][key]
            frame_val = float(pf_arr[frame_idx]) if frame_idx < len(pf_arr) else 0.0
            frames[name] = {
                "jpeg_b64": base64.b64encode(jpeg_data).decode("ascii") if jpeg_data else "",
                "value": frame_val,
            }
        comparisons[key] = {
            "frame_num": frame_idx,
            "best_clip": best_clip,
            "frames": frames,
        }
        print(f"  {METRIC_INFO[key][0]}: frame {frame_idx} (p90 in {best_clip})")

    return comparisons


def extract_sample_screenshots(clip_paths, clip_names, n_screenshots, skip_offsets):
    """Extract N evenly-spaced screenshots from each clip.

    skip_offsets: dict {clip_name: int} — frames skipped at start.
    Returns dict: {clip_name: [{"frame_num": int, "jpeg_b64": str}, ...]}
    """
    if n_screenshots <= 0:
        return {}

    samples = {}
    for name in clip_names:
        path = clip_paths[name]
        skip = skip_offsets.get(name, 0)
        # Get frame count via ffprobe
        cmd = ["ffprobe", "-v", "error", "-count_frames",
               "-select_streams", "v:0",
               "-show_entries", "stream=nb_read_frames",
               "-of", "csv=p=0", path]
        try:
            total = int(subprocess.check_output(cmd, text=True).strip())
        except (ValueError, subprocess.CalledProcessError):
            total = 500  # fallback

        usable = total - skip
        indices = [int((i + 1) * usable / (n_screenshots + 1)) for i in range(n_screenshots)]
        clip_samples = []
        for fi in indices:
            actual_frame = fi + skip
            jpeg_data = extract_frame_jpeg(path, actual_frame)
            clip_samples.append({
                "frame_num": fi,
                "jpeg_b64": base64.b64encode(jpeg_data).decode("ascii") if jpeg_data else "",
            })
        samples[name] = clip_samples
        print(f"  {name}: {len(clip_samples)} screenshots")

    return samples


# =====================================================================
# TEXT REPORT
# =====================================================================

def format_text_report(all_results, src_dir, metric_keys):
    """Format results into a readable text report with auto-sized columns."""
    clip_names = sorted(all_results.keys())
    col_w = max(len(n) for n in clip_names) + 2

    lines = []
    lines.append("=" * 120)
    lines.append("NO-REFERENCE VIDEO QUALITY METRICS REPORT")
    lines.append("=" * 120)
    lines.append(f"Source: {src_dir}")
    lines.append("All metrics are brightness-agnostic (no upstream normalization required).")
    lines.append("")
    lines.append(f"--- Quality Metrics ({len(metric_keys)}) ---")

    headers = [short_metric_header(k) for k in metric_keys]
    header = f"{'Clip':<{col_w}}" + "".join(f" {h:>10}" for h in headers)
    lines.append(header)
    lines.append("-" * len(header))
    for name in clip_names:
        row = f"{name:<{col_w}}"
        for k in metric_keys:
            v = all_results[name][k]["mean"]
            row += f" {fmt_metric_value(k, v):>10}"
        lines.append(row)
    lines.append("")

    # Rankings
    lines.append("=" * 120)
    lines.append("RANKINGS (best to worst per metric)")
    lines.append("=" * 120)

    for key in metric_keys:
        label, _, higher_better = METRIC_INFO[key]
        if higher_better is True:
            dir_str = "(higher = better)"
        elif higher_better is False:
            dir_str = "(lower = better)"
        else:
            dir_str = "(closer to 1.0 = better)"
        lines.append(f"\n  {label} {dir_str}:")

        items = [(n, all_results[n][key]["mean"]) for n in clip_names]
        if higher_better is None:
            items.sort(key=lambda x: abs(x[1] - 1.0))
        elif higher_better:
            items.sort(key=lambda x: -x[1])
        else:
            items.sort(key=lambda x: x[1])

        for rank, (name, val) in enumerate(items, 1):
            lines.append(f"    {rank:>2}. {name:<{col_w}} {fmt_metric_value(key, val)}")

    # Composite rankings
    composites = compute_composites(all_results, metric_keys=metric_keys)

    lines.append("")
    lines.append("=" * 120)
    lines.append("OVERALL COMPOSITE RANKING")
    lines.append("=" * 120)
    lines.append("")
    ranked = sorted(clip_names, key=lambda c: composites[c]["overall"])
    for rank, name in enumerate(ranked, 1):
        lines.append(f"  {rank:>2}. {name:<{col_w}} avg rank: {composites[name]['overall']:.1f}")

    lines.append("")
    lines.append("=" * 120)
    return "\n".join(lines)


# =====================================================================
# HTML REPORT
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
  .chart-radar { height: 600px; max-width: 900px; margin: 0 auto; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }
  .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 20px; }
  .metric-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .metric-card canvas { height: 300px !important; }
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
  .rank { color: var(--text-dim); font-size: 0.85em; margin-right: 4px; }
  .legend-note { font-size: 0.85em; color: var(--text-dim); margin-top: 12px; }
  .metric-guide { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 8px 24px; }
  .metric-guide div { font-size: 0.88em; padding: 4px 0; }
  .metric-guide .mg-label { font-weight: 600; }
  .dir { font-size: 0.78em; padding: 2px 6px; border-radius: 4px; margin-left: 6px; }
  .dir-up { background: #1a3a2a; color: var(--good); }
  .dir-down { background: #3a1a1a; color: var(--bad); }
  .dir-mid { background: #3a2e1a; color: var(--mid); }
  /* Lightbox — CSS-only click-to-enlarge */
  .lb-toggle { display: none; }
  .lb-thumb { cursor: zoom-in; display: block; }
  .lb-thumb img { transition: opacity 0.15s; }
  .lb-thumb:hover img { opacity: 0.85; }
  .lb-overlay { display: none; position: fixed; inset: 0;
    background: #000; z-index: 9999; cursor: zoom-out;
    justify-content: center; align-items: center; flex-direction: column; padding: 20px; }
  .lb-overlay img { max-width: 95vw; max-height: 85vh; object-fit: contain; border-radius: 4px;
    transform-origin: 0 0; user-select: none; -webkit-user-drag: none; }
  .lb-overlay .lb-caption { color: #e6edf3; font-size: 0.9em; margin-top: 12px;
    text-align: center; max-width: 90vw; }
  .lb-toggle:checked + label.lb-overlay { display: flex; }
  .lb-hint { color: #555; font-size: 0.75em; margin-top: 6px; }
  /* A/B Comparison Slider */
  .cmp-card { position: relative; }
  .cmp-btn {
    position: absolute; top: 8px; right: 8px; width: 36px; height: 36px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(0,0,0,0.6); color: #ccc; border-radius: 6px;
    cursor: pointer; z-index: 10; user-select: none; line-height: 0;
    transition: background 0.15s, color 0.15s;
  }
  .cmp-btn:hover { background: rgba(30,90,200,0.8); color: #fff; }
  .cmp-card.cmp-selected { outline: 2px solid var(--accent); }
  .cmp-card.cmp-selected .cmp-btn { background: var(--accent); color: #fff; }
  .cmp-toast {
    display: none; position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: #1f6feb; color: #fff; padding: 10px 24px; border-radius: 8px;
    font-size: 0.95em; z-index: 10000; box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    pointer-events: none;
  }
  .ab-overlay {
    display: none; position: fixed; inset: 0; background: #000; z-index: 9999;
    flex-direction: column; align-items: center; justify-content: center;
  }
  .ab-viewport {
    position: relative; max-width: 95vw; max-height: 85vh;
    overflow: hidden; cursor: col-resize; touch-action: none;
    transform-origin: 0 0;
  }
  .ab-viewport img { display: block; max-width: 95vw; max-height: 85vh; object-fit: contain;
    user-select: none; -webkit-user-drag: none; }
  .ab-img-b {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    overflow: hidden; clip-path: inset(0 0 0 50%);
  }
  .ab-img-b img { display: block; max-width: 95vw; max-height: 85vh; object-fit: contain; }
  .ab-divider {
    position: absolute; top: 0; left: 50%; width: 3px; height: 100%;
    background: #fff; pointer-events: none; z-index: 2;
  }
  .ab-divider::after {
    content: ''; position: absolute; top: 50%; left: 50%;
    transform: translate(-50%,-50%); width: 28px; height: 28px;
    border: 2px solid #fff; border-radius: 50%; background: rgba(0,0,0,0.5);
  }
  .ab-labels {
    display: flex; gap: 20px; margin-bottom: 8px; font-size: 0.85em;
    color: #aaa; z-index: 2;
  }
  .ab-labels span { padding: 2px 10px; border-radius: 4px; background: rgba(255,255,255,0.1); }
  .ab-close-hint { margin-top: 10px; font-size: 0.8em; color: #666; z-index: 2; }
"""


def generate_html(data, metric_keys, title="Video Quality Report", comparisons=None,
                  samples=None, json_filename=None):
    """Generate a self-contained HTML quality report with Chart.js visualizations."""
    clip_names = sorted(data.keys())
    n = len(clip_names)
    m = len(metric_keys)
    palette = COLORS_7 if n <= 7 else COLORS_14
    colors = {c: palette[i % len(palette)] for i, c in enumerate(clip_names)}

    composites = compute_composites(data, metric_keys=metric_keys)
    ranked_overall = sorted(clip_names, key=lambda c: composites[c]["overall"])

    # Radar: normalize each metric to 0-100
    radar_scores = {}
    for c in clip_names:
        sc = []
        for key in metric_keys:
            vals = [data[d][key]["mean"] for d in clip_names]
            mn, mx = min(vals), max(vals)
            rng = mx - mn if mx - mn > 1e-15 else 1e-15
            raw = data[c][key]["mean"]
            _, _, hb = METRIC_INFO[key]
            if hb is True:
                sc.append(((raw - mn) / rng) * 100)
            elif hb is False:
                sc.append((1 - (raw - mn) / rng) * 100)
            else:
                max_dist = max(abs(v - 1.0) for v in vals) or 1e-15
                sc.append((1 - abs(raw - 1.0) / max_dist) * 100)
        radar_scores[c] = sc
    radar_labels = [METRIC_INFO[k][0] for k in metric_keys]

    # Heatmap data
    hm_data = []
    for c in ranked_overall:
        cells = [{"raw": data[c][k]["mean"],
                  "raw_fmt": fmt_metric_value(k, data[c][k]["mean"]),
                  "z": composites[c]["zscores"][k], "key": k}
                 for k in metric_keys]
        hm_data.append({"name": c, "overall": composites[c]["overall"], "cells": cells})

    # Per-metric rankings
    per_metric = {}
    for key in metric_keys:
        _, _, hb = METRIC_INFO[key]
        items = [(c, data[c][key]["mean"]) for c in clip_names]
        if hb is None:
            items.sort(key=lambda x: abs(x[1] - 1.0))
        elif hb:
            items.sort(key=lambda x: -x[1])
        else:
            items.sort(key=lambda x: x[1])
        is_pct = key in PCT_KEYS
        per_metric[key] = {
            "labels": [c for c, _ in items],
            "values": [v * 100 for _, v in items] if is_pct else [v for _, v in items],
            "colors": [colors[c] for c, _ in items],
            "label": METRIC_INFO[key][0], "unit": METRIC_INFO[key][1],
            "higher_better": hb, "is_pct": is_pct,
        }

    # Radar datasets
    radar_datasets = [{"label": c, "data": radar_scores[c], "borderColor": colors[c],
                       "backgroundColor": colors[c] + "20", "borderWidth": 2, "pointRadius": 3,
                       "hidden": i >= 5} for i, c in enumerate(clip_names)]

    # Chart height scales with clip count
    bar_height = max(320, n * 36 + 80)
    metric_guide_html = []
    for key in metric_keys:
        label, _, hb = METRIC_INFO[key]
        if hb is True:
            dir_class, dir_text = "dir-up", "higher"
        elif hb is False:
            dir_class, dir_text = "dir-down", "lower"
        else:
            dir_class, dir_text = "dir-mid", "~1.0"
        desc = METRIC_GUIDE_TEXT.get(key, "")
        metric_guide_html.append(
            f'<div><span class="mg-label">{label}</span> <span class="dir {dir_class}">{dir_text}</span>'
            f"<br>{desc}</div>"
        )
    heatmap_headers = "".join(f"<th>{short_metric_header(k)}</th>" for k in metric_keys)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>{HTML_CSS}</style>
</head>
<body>

<h1>{title}</h1>
<p class="subtitle">
  {n} analog video captures evaluated on {m} no-reference quality metrics
  (all brightness-agnostic — no upstream normalization required).
</p>

<div class="card">
  <h3>Metric Guide</h3>
  <div class="metric-guide">
    {"".join(metric_guide_html)}
  </div>
</div>

<h2>Overall Composite Ranking</h2>
<div class="card">
  <div class="chart-container" style="height:{bar_height}px;"><canvas id="overallChart"></canvas></div>
  <p class="legend-note">Average rank across all {m} metrics (1 = best per metric). Lower average rank = better overall quality.</p>
</div>

<h2>{m}-Metric Radar Comparison</h2>
<div class="card">
  <div class="chart-container chart-radar"><canvas id="radarChart"></canvas></div>
  <p class="legend-note">All metrics on 0–100 quality scale (higher = better on all axes). Click legend to toggle.{" Top 5 shown by default." if n > 5 else ""}</p>
</div>

<h2>Detailed Heatmap — All Metrics</h2>
<div class="card" style="overflow-x:auto;">
  <table class="heatmap" id="heatmapTable">
    <thead><tr>
      <th>Clip</th><th>Avg Rank</th>
      {heatmap_headers}
    </tr></thead>
    <tbody></tbody>
  </table>
  <p class="legend-note">Ranked by average rank across all {m} metrics (lower = better). Cell color: green = good, red = poor. Click any column header to sort.</p>
</div>

<h2>Individual Metric Rankings</h2>
<div class="metric-grid" id="metricGrid"></div>

<script>
new Chart(document.getElementById('overallChart'), {{
  type:'bar',
  data:{{ labels:{json.dumps([c for c in ranked_overall])},
    datasets:[{{ data:{json.dumps([round(composites[c]["overall"], 2) for c in ranked_overall])},
      backgroundColor:{json.dumps([colors[c] for c in ranked_overall])},
      borderColor:{json.dumps([colors[c] for c in ranked_overall])}, borderWidth:1, borderRadius:4 }}] }},
  options:{{ indexAxis:'y', responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{display:false}},
      tooltip:{{callbacks:{{label:c=>'avg rank: '+c.raw}}}} }},
    scales:{{ x:{{reverse:true,min:1,max:{n},grid:{{color:'#30363d'}},ticks:{{color:'#8b949e'}},title:{{display:true,text:'Average Rank (lower = better)',color:'#8b949e'}}}}, y:{{grid:{{display:false}},ticks:{{color:'#e6edf3',font:{{size:11}}}}}} }}
  }}
}});

new Chart(document.getElementById('radarChart'),{{
  type:'radar',
  data:{{labels:{json.dumps(radar_labels)},datasets:{json.dumps(radar_datasets)}}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{position:'right',labels:{{color:'#e6edf3',font:{{size:11}},boxWidth:12,padding:6}}}}}},
    scales:{{r:{{min:0,max:100,grid:{{color:'#30363d'}},angleLines:{{color:'#30363d'}},pointLabels:{{color:'#e6edf3',font:{{size:11}}}},ticks:{{display:false}}}}}}
  }}
}});

function escHtml(s){{var d=document.createElement('div');d.textContent=s;return d.innerHTML;}}
const hmData={json.dumps(hm_data)};
const tbody=document.querySelector('#heatmapTable tbody');
hmData.forEach((row,idx)=>{{
  const tr=document.createElement('tr');
  let td=document.createElement('td');
  td.innerHTML='<span class="rank">#'+(idx+1)+'</span> '+escHtml(row.name);
  tr.appendChild(td);
  td=document.createElement('td');td.textContent=row.overall.toFixed(1);
  const mid=({n}+1)/2, ov=Math.max(-1,Math.min(1,(mid-row.overall)/mid));
  td.style.background=ov>0?'rgba(63,185,80,'+(Math.abs(ov)*0.4)+')':'rgba(248,81,73,'+(Math.abs(ov)*0.4)+')';
  td.style.fontWeight='600';tr.appendChild(td);
  row.cells.forEach(cell=>{{
    td=document.createElement('td');
    td.textContent=cell.raw_fmt;
    const z=cell.z,n=Math.max(-1,Math.min(1,z/2.5));
    td.style.background=n>0?'rgba(63,185,80,'+(Math.abs(n)*0.35)+')':'rgba(248,81,73,'+(Math.abs(n)*0.35)+')';
    tr.appendChild(td);
  }});
  tbody.appendChild(tr);
}});

const amd={json.dumps(per_metric)};
const grid=document.getElementById('metricGrid');
Object.entries(amd).forEach(([key,md])=>{{
  const card=document.createElement('div');
  card.className='metric-card';
  const canvas=document.createElement('canvas');card.appendChild(canvas);grid.appendChild(card);
  const dir=md.higher_better===true?'(higher = better)':md.higher_better===false?'(lower = better)':'(closer to 1.0)';
  new Chart(canvas,{{
    type:'bar',data:{{labels:md.labels,datasets:[{{data:md.values,backgroundColor:md.colors.map(c=>c+'cc'),borderColor:md.colors,borderWidth:1,borderRadius:3}}]}},
    options:{{indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},title:{{display:true,text:md.label+' '+dir,color:'#e6edf3',font:{{size:14}}}},tooltip:{{callbacks:{{label:c=>md.is_pct?c.parsed.x.toFixed(1)+'%':md.unit+': '+c.parsed.x.toFixed(6)}}}}}},
      scales:{{x:{{grid:{{color:'#30363d'}},ticks:{{color:'#8b949e'}}}},y:{{grid:{{display:false}},ticks:{{color:'#e6edf3',font:{{size:10}}}}}}}}
    }}
  }});
}});

document.querySelectorAll('.heatmap th').forEach((th,colIdx)=>{{
  th.addEventListener('click',()=>{{
    const table=th.closest('table'),tbody=table.querySelector('tbody');
    const rows=Array.from(tbody.querySelectorAll('tr'));
    const asc=!th.classList.contains('sort-asc');
    table.querySelectorAll('th').forEach(h=>h.classList.remove('sort-asc','sort-desc'));
    th.classList.add(asc?'sort-asc':'sort-desc');
    rows.sort((a,b)=>{{
      const av=a.cells[colIdx].textContent.replace(/^#\\d+\\s*/,'');
      const bv=b.cells[colIdx].textContent.replace(/^#\\d+\\s*/,'');
      const an=parseFloat(av),bn=parseFloat(bv);
      if(!isNaN(an)&&!isNaN(bn)) return asc?an-bn:bn-an;
      return asc?av.localeCompare(bv):bv.localeCompare(av);
    }});
    rows.forEach((r,i)=>{{
      const fc=r.cells[0],name=fc.textContent.replace(/^#\\d+\\s*/,'');
      fc.innerHTML='<span class="rank">#'+(i+1)+'</span> '+escHtml(name);
      tbody.appendChild(r);
    }});
  }});
}});
</script>
"""

    # Collect lightbox overlays to emit at document root level (avoids stacking context clipping)
    lb_id = 0
    lb_overlays = []  # list of (id, img_src, caption) tuples

    # Append metric comparison frames if available
    if comparisons:
        html += """<h2>Visual Metric Comparisons</h2>
<p class="subtitle">For each metric, a strong example frame from the best-scoring clip (near its 90th percentile)
is shown. All clips are shown at the same frame number for direct comparison.
Click any image to enlarge; use the <span style="display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;background:rgba(0,0,0,0.55);border-radius:4px;vertical-align:middle;color:#ccc;">""" + CMP_ICON_SVG.replace('width="20" height="20"', 'width="16" height="16"') + """</span> icon to A/B compare two images with a slider.</p>
"""
        for key in metric_keys:
            if key not in comparisons:
                continue
            comp = comparisons[key]
            label, unit, higher_better = METRIC_INFO[key]
            if higher_better is True:
                direction = "(higher = better)"
            elif higher_better is False:
                direction = "(lower = better)"
            else:
                direction = "(closer to 1.0 = better)"
            html += f"""<h3>{label} {direction} — Frame {comp["frame_num"]} (p90 in {comp["best_clip"]})</h3>
<div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:24px;">
"""
            # Sort by metric value (best first)
            if higher_better is True:
                frame_items = sorted(comp["frames"].items(), key=lambda x: -x[1]["value"])
            elif higher_better is False:
                frame_items = sorted(comp["frames"].items(), key=lambda x: x[1]["value"])
            else:
                frame_items = sorted(comp["frames"].items(), key=lambda x: abs(x[1]["value"] - 1.0))
            is_pct = key in ("crushed_blacks", "blown_whites")
            for rank, (name, fdata) in enumerate(frame_items, 1):
                if fdata["jpeg_b64"]:
                    img_src = f"data:image/jpeg;base64,{fdata['jpeg_b64']}"
                    vfmt = f"{fdata['value']*100:.1f}%" if is_pct else f"{fdata['value']:.6f}"
                    caption = f"#{rank} {name} &mdash; {label}: {vfmt} &mdash; Frame {comp['frame_num']}"
                    lb_overlays.append((lb_id, img_src, caption))
                    html += f"""  <div class="card cmp-card" style="flex:1;min-width:300px;max-width:48%;">
    <div class="cmp-btn" data-cmp-caption="#{rank} {name} &mdash; {label}: {vfmt}">{CMP_ICON_SVG}</div>
    <div style="font-weight:600;margin-bottom:4px;"><span class="rank">#{rank}</span> {name}</div>
    <div style="font-size:0.9em;color:var(--text-dim);margin-bottom:8px;">{label}: {vfmt}</div>
    <label for="lb{lb_id}" class="lb-thumb"><img src="{img_src}" style="width:100%;border-radius:4px;"></label>
  </div>
"""
                    lb_id += 1
            html += "</div>\n"

    # Append sample screenshots if available
    if samples:
        html += """<h2>Appendix: Sample Frames</h2>
<p class="subtitle">Evenly-spaced frames from each clip for visual reference and decode verification. Click any image to enlarge.</p>
"""
        for name in sorted(samples.keys()):
            frames = samples[name]
            html += f'<h3>{name}</h3>\n<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;">\n'
            for sf in frames:
                if sf["jpeg_b64"]:
                    img_src = f"data:image/jpeg;base64,{sf['jpeg_b64']}"
                    caption = f"{name} &mdash; Frame {sf['frame_num']}"
                    lb_overlays.append((lb_id, img_src, caption))
                    html += f"""  <div style="flex:1;min-width:200px;max-width:32%;">
    <div style="font-size:0.8em;color:var(--text-dim);margin-bottom:4px;">Frame {sf["frame_num"]}</div>
    <label for="lb{lb_id}" class="lb-thumb"><img src="{img_src}" style="width:100%;border-radius:4px;"></label>
  </div>
"""
                    lb_id += 1
            html += "</div>\n"

    # Emit lightbox overlays at document root level (outside all containers)
    if lb_overlays:
        html += '\n<!-- Lightbox overlays (document root to avoid stacking context clipping) -->\n'
        for lid, img_src, caption in lb_overlays:
            html += f'<input type="checkbox" id="lb{lid}" class="lb-toggle">'
            html += f'<label for="lb{lid}" class="lb-overlay"><img src="{img_src}"><div class="lb-caption">{caption}</div><div class="lb-hint">Scroll to zoom &middot; Drag to pan &middot; R to reset &middot; Click to close</div></label>\n'

    # Lightbox zoom/pan script (works for all lightbox images)
    if lb_overlays:
        html += """
<script>
(function(){
  var scale=1,tx=0,ty=0,dragging=false,lastX=0,lastY=0;

  function getActive(){
    var chk=document.querySelector('.lb-toggle:checked');
    if(!chk) return null;
    return chk.nextElementSibling;
  }
  function getImg(){
    var ov=getActive();
    return ov?ov.querySelector('img'):null;
  }
  function apply(img){
    img.style.transform='translate('+tx+'px,'+ty+'px) scale('+scale+')';
    img.style.cursor=scale>1?'grab':'zoom-in';
  }
  function resetZoom(){
    scale=1;tx=0;ty=0;
    var img=getImg();
    if(img){img.style.transform='';img.style.cursor='zoom-in';}
  }

  /* Wheel zoom — zoom toward cursor */
  document.addEventListener('wheel',function(e){
    var img=getImg();
    if(!img) return;
    e.preventDefault();
    var rect=img.getBoundingClientRect();
    var oldS=scale;
    scale=Math.max(1,Math.min(15,scale*(e.deltaY<0?1.15:1/1.15)));
    if(scale===1){resetZoom();return;}
    var ratio=scale/oldS;
    tx=(1-ratio)*(e.clientX-rect.left)+tx;
    ty=(1-ratio)*(e.clientY-rect.top)+ty;
    apply(img);
  },{passive:false});

  /* Drag to pan */
  document.addEventListener('mousedown',function(e){
    var img=getImg();
    if(!img||scale<=1||e.target!==img) return;
    dragging=true;lastX=e.clientX;lastY=e.clientY;
    img.style.cursor='grabbing';
    e.preventDefault();
  });
  document.addEventListener('mousemove',function(e){
    if(!dragging) return;
    var img=getImg();
    if(!img) return;
    tx+=e.clientX-lastX;ty+=e.clientY-lastY;
    lastX=e.clientX;lastY=e.clientY;
    apply(img);
  });
  document.addEventListener('mouseup',function(){
    if(dragging){
      dragging=false;
      var img=getImg();
      if(img&&scale>1) img.style.cursor='grab';
    }
  });

  /* Prevent close when zoomed */
  document.querySelectorAll('.lb-overlay').forEach(function(ov){
    ov.addEventListener('click',function(e){
      if(scale>1){e.preventDefault();e.stopPropagation();}
    });
  });

  /* Prevent native image drag */
  document.querySelectorAll('.lb-overlay img').forEach(function(img){
    img.addEventListener('dragstart',function(e){e.preventDefault();});
  });

  /* Keyboard: r=reset, Escape=close+reset */
  document.addEventListener('keydown',function(e){
    if((e.key==='r'||e.key==='R')&&getImg()){e.preventDefault();resetZoom();}
    if(e.key==='Escape') resetZoom();
  });

  /* Reset zoom when lightbox closes */
  document.querySelectorAll('.lb-toggle').forEach(function(chk){
    chk.addEventListener('change',function(){if(!chk.checked) resetZoom();});
  });

  /* Touch: pinch-to-zoom + drag */
  var lastDist=0,lastMid=null;
  document.addEventListener('touchstart',function(e){
    var img=getImg();if(!img) return;
    if(e.touches.length===2){
      var dx=e.touches[0].clientX-e.touches[1].clientX;
      var dy=e.touches[0].clientY-e.touches[1].clientY;
      lastDist=Math.sqrt(dx*dx+dy*dy);
      lastMid={x:(e.touches[0].clientX+e.touches[1].clientX)/2,
               y:(e.touches[0].clientY+e.touches[1].clientY)/2};
      e.preventDefault();
    }else if(e.touches.length===1&&scale>1){
      dragging=true;lastX=e.touches[0].clientX;lastY=e.touches[0].clientY;
      e.preventDefault();
    }
  },{passive:false});
  document.addEventListener('touchmove',function(e){
    var img=getImg();if(!img) return;
    if(e.touches.length===2&&lastDist>0){
      e.preventDefault();
      var dx=e.touches[0].clientX-e.touches[1].clientX;
      var dy=e.touches[0].clientY-e.touches[1].clientY;
      var dist=Math.sqrt(dx*dx+dy*dy);
      var mid={x:(e.touches[0].clientX+e.touches[1].clientX)/2,
               y:(e.touches[0].clientY+e.touches[1].clientY)/2};
      var oldS=scale;
      scale=Math.max(1,Math.min(15,scale*(dist/lastDist)));
      if(scale===1){resetZoom();lastDist=dist;return;}
      var rect=img.getBoundingClientRect();
      var ratio=scale/oldS;
      tx=(1-ratio)*(mid.x-rect.left)+tx;
      ty=(1-ratio)*(mid.y-rect.top)+ty;
      tx+=mid.x-lastMid.x;ty+=mid.y-lastMid.y;
      lastDist=dist;lastMid=mid;
      apply(img);
    }else if(e.touches.length===1&&dragging&&scale>1){
      e.preventDefault();
      tx+=e.touches[0].clientX-lastX;ty+=e.touches[0].clientY-lastY;
      lastX=e.touches[0].clientX;lastY=e.touches[0].clientY;
      apply(img);
    }
  },{passive:false});
  document.addEventListener('touchend',function(e){
    if(e.touches.length<2) lastDist=0;
    if(e.touches.length===0) dragging=false;
  });
})();
</script>
"""

    # A/B comparison slider (only when comparison frames exist)
    if comparisons:
        html += """
<!-- A/B Comparison Slider -->
<div class="cmp-toast" id="cmpToast">Select another image to compare</div>
<div class="ab-overlay" id="abOverlay">
  <div class="ab-labels"><span id="abLabelA">A</span><span id="abLabelB">B</span></div>
  <div class="ab-viewport" id="abViewport">
    <img id="abImgA" src="" alt="A">
    <div class="ab-img-b" id="abClipB"><img id="abImgB" src="" alt="B"></div>
    <div class="ab-divider" id="abDivider"></div>
  </div>
  <div class="ab-close-hint">Scroll to zoom &middot; Drag to pan &middot; R to reset &middot; Escape to close</div>
</div>
<script>
(function(){
  var selectedA = null;
  var toast = document.getElementById('cmpToast');
  var overlay = document.getElementById('abOverlay');
  var viewport = document.getElementById('abViewport');
  var imgA = document.getElementById('abImgA');
  var imgB = document.getElementById('abImgB');
  var clipB = document.getElementById('abClipB');
  var divider = document.getElementById('abDivider');
  var labelA = document.getElementById('abLabelA');
  var labelB = document.getElementById('abLabelB');

  /* Zoom state for A/B overlay */
  var zs=1, ztx=0, zty=0, zDrag=false, zLastX=0, zLastY=0, zDidDrag=false;

  function clearSelection() {
    if (selectedA) selectedA.el.classList.remove('cmp-selected');
    selectedA = null;
    toast.style.display = 'none';
  }

  function getSrc(btn) {
    return btn.closest('.cmp-card').querySelector('img').src;
  }

  function updateSlider(pct) {
    clipB.style.clipPath = 'inset(0 0 0 ' + pct + '%)';
    divider.style.left = pct + '%';
  }

  function applyZoom() {
    viewport.style.transform = 'translate('+ztx+'px,'+zty+'px) scale('+zs+')';
    viewport.style.cursor = zs > 1 ? 'grab' : 'col-resize';
  }

  function resetABZoom() {
    zs=1; ztx=0; zty=0;
    viewport.style.transform = '';
    viewport.style.cursor = 'col-resize';
  }

  function openAB(srcA, captA, srcB, captB) {
    imgA.src = srcA; imgB.src = srcB;
    labelA.textContent = 'A: ' + captA;
    labelB.textContent = 'B: ' + captB;
    updateSlider(50);
    resetABZoom();
    overlay.style.display = 'flex';
    clearSelection();
  }

  function closeAB() { resetABZoom(); overlay.style.display = 'none'; }

  document.querySelectorAll('.cmp-btn').forEach(function(btn) {
    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      var src = getSrc(btn);
      var caption = btn.getAttribute('data-cmp-caption');
      if (!selectedA) {
        selectedA = { el: btn.closest('.cmp-card'), src: src, caption: caption };
        selectedA.el.classList.add('cmp-selected');
        toast.style.display = 'block';
      } else if (selectedA.el === btn.closest('.cmp-card')) {
        clearSelection();
      } else {
        openAB(selectedA.src, selectedA.caption, src, caption);
      }
    });
  });

  /* Wheel zoom on A/B viewport */
  overlay.addEventListener('wheel', function(e) {
    if (overlay.style.display !== 'flex') return;
    e.preventDefault();
    var rect = viewport.getBoundingClientRect();
    var oldS = zs;
    zs = Math.max(1, Math.min(15, zs * (e.deltaY < 0 ? 1.15 : 1/1.15)));
    if (zs === 1) { resetABZoom(); return; }
    var ratio = zs / oldS;
    ztx = (1-ratio) * (e.clientX - rect.left) + ztx;
    zty = (1-ratio) * (e.clientY - rect.top) + zty;
    applyZoom();
  }, {passive: false});

  /* Mouse drag to pan when zoomed, slider when not */
  viewport.addEventListener('mousedown', function(e) {
    if (zs > 1) {
      zDrag = true; zDidDrag = false;
      zLastX = e.clientX; zLastY = e.clientY;
      viewport.style.cursor = 'grabbing';
      e.preventDefault();
    }
  });
  viewport.addEventListener('mousemove', function(e) {
    if (zDrag) {
      ztx += e.clientX - zLastX; zty += e.clientY - zLastY;
      zLastX = e.clientX; zLastY = e.clientY;
      zDidDrag = true;
      applyZoom();
      return;
    }
    /* Slider — works at any zoom level */
    var rect = viewport.getBoundingClientRect();
    var pct = ((e.clientX - rect.left) / rect.width) * 100;
    pct = Math.max(0, Math.min(100, pct));
    updateSlider(pct);
  });
  document.addEventListener('mouseup', function() {
    if (zDrag) { zDrag = false; if (zs > 1) viewport.style.cursor = 'grab'; }
  });

  /* Touch: single-finger slider or pan; two-finger pinch zoom */
  var abTouchDist=0, abTouchMid=null;
  viewport.addEventListener('touchstart', function(e) {
    if (e.touches.length === 2) {
      var dx=e.touches[0].clientX-e.touches[1].clientX;
      var dy=e.touches[0].clientY-e.touches[1].clientY;
      abTouchDist = Math.sqrt(dx*dx+dy*dy);
      abTouchMid = {x:(e.touches[0].clientX+e.touches[1].clientX)/2,
                    y:(e.touches[0].clientY+e.touches[1].clientY)/2};
      e.preventDefault();
    } else if (e.touches.length === 1 && zs > 1) {
      zDrag = true; zDidDrag = false;
      zLastX = e.touches[0].clientX; zLastY = e.touches[0].clientY;
      e.preventDefault();
    }
  }, {passive: false});
  viewport.addEventListener('touchmove', function(e) {
    e.preventDefault();
    if (e.touches.length === 2 && abTouchDist > 0) {
      var dx=e.touches[0].clientX-e.touches[1].clientX;
      var dy=e.touches[0].clientY-e.touches[1].clientY;
      var dist=Math.sqrt(dx*dx+dy*dy);
      var mid={x:(e.touches[0].clientX+e.touches[1].clientX)/2,
               y:(e.touches[0].clientY+e.touches[1].clientY)/2};
      var oldS=zs;
      zs=Math.max(1,Math.min(15,zs*(dist/abTouchDist)));
      if(zs===1){resetABZoom();abTouchDist=dist;return;}
      var rect=viewport.getBoundingClientRect();
      var ratio=zs/oldS;
      ztx=(1-ratio)*(mid.x-rect.left)+ztx;
      zty=(1-ratio)*(mid.y-rect.top)+zty;
      ztx+=mid.x-abTouchMid.x; zty+=mid.y-abTouchMid.y;
      abTouchDist=dist; abTouchMid=mid;
      applyZoom();
    } else if (e.touches.length===1 && zDrag && zs>1) {
      ztx+=e.touches[0].clientX-zLastX; zty+=e.touches[0].clientY-zLastY;
      zLastX=e.touches[0].clientX; zLastY=e.touches[0].clientY;
      zDidDrag=true;
      applyZoom();
    } else if (e.touches.length===1 && zs<=1) {
      var rect=viewport.getBoundingClientRect();
      var pct=((e.touches[0].clientX-rect.left)/rect.width)*100;
      pct=Math.max(0,Math.min(100,pct));
      updateSlider(pct);
    }
  }, {passive: false});
  viewport.addEventListener('touchend', function(e) {
    if(e.touches.length<2) abTouchDist=0;
    if(e.touches.length===0){
      zDrag=false;
      if(!zDidDrag && zs<=1) closeAB();
      zDidDrag=false;
    }
  });

  /* Click: close only at 1x zoom and if not after a drag */
  viewport.addEventListener('click', function() {
    if (zDidDrag) { zDidDrag = false; return; }
    if (zs > 1) return;
    closeAB();
  });

  document.addEventListener('keydown', function(e) {
    if (overlay.style.display === 'flex') {
      if (e.key === 'r' || e.key === 'R') { e.preventDefault(); resetABZoom(); return; }
      if (e.key === 'Escape') { closeAB(); return; }
    }
    if (e.key === 'Escape') clearSelection();
  });
})();
</script>
"""

    if json_filename:
        html += f'\n<p style="text-align:center; color:#888; font-size:0.85em; margin-top:2em;">Data source: <code>{json_filename}</code></p>\n'
        html += f'<!-- data-source: {json_filename} -->\n'

    html += "</body>\n</html>"

    return html
