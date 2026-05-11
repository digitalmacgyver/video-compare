#!/usr/bin/env python3
"""tp_compare: render multi-capture SW2 measurement comparison HTML.

CLI:
    python tp_compare.py cap1.json cap2.json ... --output report.html
"""

from __future__ import annotations
import html as _h
import json
import os
from typing import Any, Dict, List

import tp_chart


def _basename(path: str) -> str:
    return os.path.basename(path)


def render_registration_summary(captures: List[Dict[str, Any]]) -> str:
    rows = []
    for c in captures:
        m = c["_meta"]
        reg = m["registration"]
        affine = reg["affine"]
        if affine is not None:
            tx = affine[0][2]
            ty = affine[1][2]
            sx = affine[0][0]
            sy = affine[1][1]
            shear = affine[0][1]
            transform = (
                f"tx={tx:+.2f} ty={ty:+.2f} "
                f"sx={sx:.4f} sy={sy:.4f} shear={shear:+.4f}"
            )
        else:
            transform = "<i>registration failed</i>"
        flag = reg.get("quality_flag", "?")
        flag_class = {"ok": "ok", "warn": "warn", "failed": "bad"}.get(flag, "")
        residuals = reg["residuals_px"]
        warn_prog = " (progressive!)" if m.get("progressive_warning") else ""
        rows.append(
            f"<tr>"
            f"<td>{_h.escape(_basename(m['capture']))}{warn_prog}</td>"
            f"<td>{m['raster_in'][0]}x{m['raster_in'][1]}</td>"
            f"<td>{m.get('field_order', 'unknown')}</td>"
            f"<td class='{flag_class}'>{flag}</td>"
            f"<td>{residuals['mean']:.2f}</td>"
            f"<td>{residuals['max']:.2f}</td>"
            f"<td>{reg['inliers']}/{reg['total']}</td>"
            f"<td><code>{transform}</code></td>"
            f"</tr>"
        )
    table_body = "\n".join(rows)
    return f"""
<section class="registration">
  <h2>Registration Summary</h2>
  <table class="data">
    <thead>
      <tr>
        <th>Capture</th><th>Raster</th><th>Field</th>
        <th>Quality</th><th>Mean&nbsp;px</th><th>Max&nbsp;px</th>
        <th>Inliers</th><th>Affine</th>
      </tr>
    </thead>
    <tbody>
{table_body}
    </tbody>
  </table>
</section>
"""


def _swatch(rgb_tuple, role: str) -> str:
    """role is 'ideal' or 'measured' — drives styling and the hover tooltip."""
    r, g, b = rgb_tuple
    label = "ideal" if role == "ideal" else "measured"
    return (
        f"<span class='swatch swatch-{role}' "
        f"style='background-color: rgb({r},{g},{b});' "
        f"title='{label} RGB ({r},{g},{b})'></span>"
    )


def _delta_class(d: float) -> str:
    a = abs(d)
    if a < 5:
        return "delta-good"
    if a < 15:
        return "delta-warn"
    return "delta-bad"


def render_tartan_deltas(captures: List[Dict[str, Any]]) -> str:
    # Use TARTAN_REGIONS as the canonical column ordering.
    region_ids = [r["id"] for r in tp_chart.TARTAN_REGIONS]

    head = "<tr><th>Capture</th>" + "".join(
        f"<th>{rid}</th>" for rid in region_ids
    ) + "</tr>"

    body_rows = []
    for c in captures:
        cap_name = _basename(c["_meta"]["capture"])
        cells = [f"<td>{_h.escape(cap_name)}</td>"]
        by_id = {p["id"]: p for p in c["tartan"]}
        for rid in region_ids:
            p = by_id.get(rid)
            if p is None:
                cells.append("<td>-</td>")
                continue
            ideal_rgb = tp_chart.yuv10_to_rgb8(*p["ideal_yuv10"])
            meas_rgb = tp_chart.yuv10_to_rgb8(*p["measured_yuv10"])
            dy, du, dv = p["delta_yuv10"]
            cls = _delta_class(dy)
            cells.append(
                f"<td class='{cls}'>"
                f"<div class='swatch-pair'>"
                f"<div class='swatch-cell'>{_swatch(ideal_rgb, 'ideal')}"
                f"<div class='swatch-label'>ref</div></div>"
                f"<div class='swatch-cell'>{_swatch(meas_rgb, 'measured')}"
                f"<div class='swatch-label'>cap</div></div>"
                f"</div>"
                f"<div class='delta'>"
                f"&Delta;Y={dy:+.1f}<br>"
                f"&Delta;U={du:+.1f}<br>"
                f"&Delta;V={dv:+.1f}"
                f"</div></td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    body = "\n".join(body_rows)
    return f"""
<section class="tartan">
  <h2>Tartan Deltas (measured vs ideal)</h2>
  <p class="legend">
    Each cell:
    <span class='swatch swatch-ideal' style='background:#888;'></span>&nbsp;<b>ref</b>
    (synthesized ideal from Rec.601 math, dashed border)&nbsp;&nbsp;
    <span class='swatch swatch-measured' style='background:#888;'></span>&nbsp;<b>cap</b>
    (sampled from the capture)
    &nbsp;&nbsp;then the YUV10 deltas of capture vs ref.
  </p>
  <table class="data tartan-table">
    <thead>{head}</thead>
    <tbody>
{body}
    </tbody>
  </table>
</section>
"""


def render_gray_deltas(captures: List[Dict[str, Any]]) -> str:
    region_ids = [r["id"] for r in tp_chart.GRAY_REGIONS]
    head = "<tr><th>Capture</th>" + "".join(
        f"<th>{rid}</th>" for rid in region_ids
    ) + "</tr>"

    body_rows = []
    for c in captures:
        cap_name = _basename(c["_meta"]["capture"])
        cells = [f"<td>{_h.escape(cap_name)}</td>"]
        by_id = {g["id"]: g for g in c["grays"]}
        for rid in region_ids:
            g = by_id.get(rid)
            if g is None:
                cells.append("<td>-</td>")
                continue
            cls = _delta_class(g["delta_y10"])
            ideal_rgb = tp_chart.yuv10_to_rgb8(g["ideal_y10"], 512, 512)
            meas_rgb = tp_chart.yuv10_to_rgb8(
                g["measured_y10"],
                g.get("u10", 512),
                g.get("v10", 512),
            )
            cells.append(
                f"<td class='{cls}'>"
                f"<div class='swatch-pair'>"
                f"<div class='swatch-cell'>{_swatch(ideal_rgb, 'ideal')}"
                f"<div class='swatch-label'>ref</div>"
                f"<div class='small muted'>Y={g['ideal_y10']:.1f}</div></div>"
                f"<div class='swatch-cell'>{_swatch(meas_rgb, 'measured')}"
                f"<div class='swatch-label'>cap</div>"
                f"<div class='small muted'>Y={g['measured_y10']:.1f}</div></div>"
                f"</div>"
                f"<div class='delta'>&Delta;Y={g['delta_y10']:+.2f}</div>"
                f"</td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    # Chart.js linearity plot: x = ideal Y10, y = measured Y10, one dataset per capture.
    ideals = [tp_chart.GRAY_IDEAL_Y10[i] for i in range(4)]
    datasets = []
    for c in captures:
        by_id = {g["id"]: g for g in c["grays"]}
        cap_name = _basename(c["_meta"]["capture"])
        ys = [by_id.get(rid, {}).get("measured_y10", None) for rid in region_ids]
        datasets.append({"label": cap_name, "data": ys})
    chart_data = {
        "labels": [f"{v:.1f}" for v in ideals],
        "datasets": datasets,
        "ideal": ideals,
    }
    chart_json = json.dumps(chart_data)

    return f"""
<section class="gray">
  <h2>Gray Step Deltas + Linearity</h2>
  <p class="legend">
    Each cell:
    <span class='swatch swatch-ideal' style='background:#888;'></span>&nbsp;<b>ref</b>
    (ideal gray level)&nbsp;&nbsp;
    <span class='swatch swatch-measured' style='background:#888;'></span>&nbsp;<b>cap</b>
    (sampled from the capture; small chroma deviations show up as a tint).
  </p>
  <table class="data gray-table">
    <thead>{head}</thead>
    <tbody>
{''.join(body_rows)}
    </tbody>
  </table>
  <div class="chart-wrap">
    <canvas id="grayLinearity" width="640" height="320"></canvas>
  </div>
  <script>
    (function() {{
      const data = {chart_json};
      const datasets = data.datasets.map(function(ds) {{
        return {{
          label: ds.label,
          data: ds.data,
          fill: false,
          tension: 0.0,
        }};
      }});
      // Add ideal as the reference line.
      datasets.unshift({{label: "ideal", data: data.ideal, borderDash: [5, 5], fill: false}});
      const ctx = document.getElementById("grayLinearity").getContext("2d");
      new Chart(ctx, {{
        type: "line",
        data: {{labels: data.labels, datasets: datasets}},
        options: {{responsive: false, scales: {{y: {{title: {{display: true, text: "measured Y10"}}}}, x: {{title: {{display: true, text: "ideal Y10"}}}}}}}}
      }});
    }})();
  </script>
</section>
"""


_CSS = """
body { background: #181a1f; color: #d8dde6; font-family: system-ui, sans-serif; margin: 24px; }
h1 { color: #fff; }
h2 { color: #fff; border-bottom: 1px solid #2a2e36; padding-bottom: 4px; }
p.legend { font-size: 12px; color: #b8c0cc; line-height: 1.6; }
table.data { border-collapse: collapse; margin: 12px 0; }
table.data th, table.data td { border: 1px solid #2a2e36; padding: 6px 10px; vertical-align: top; }
table.data th { background: #21252b; color: #fff; }
.swatch { display: inline-block; width: 22px; height: 22px; vertical-align: middle; }
.swatch-ideal    { border: 2px dashed #c5d1e0; box-sizing: border-box; }
.swatch-measured { border: 2px solid  #f0b450; box-sizing: border-box; }
.swatch-pair   { display: flex; gap: 4px; }
.swatch-cell   { display: flex; flex-direction: column; align-items: center; }
.swatch-label  { font-size: 9px; color: #b8c0cc; line-height: 1.2; margin-top: 1px; }
.delta { font-size: 11px; margin-top: 4px; color: #b8c0cc; }
.delta-good { background: rgba(80,200,120,0.10); }
.delta-warn { background: rgba(240,180,80,0.15); }
.delta-bad  { background: rgba(220,80,80,0.18); }
.ok   { color: #61c08f; font-weight: 600; }
.warn { color: #f0b450; font-weight: 600; }
.bad  { color: #e26464; font-weight: 600; }
.small { font-size: 11px; color: #b8c0cc; }
.muted { color: #8a929f; }
code { color: #c5d1e0; }
.geometry-panel { margin: 12px 0; padding: 8px 12px; background: #1d2026; border: 1px solid #2a2e36; }
.geometry-panel h3 { margin: 4px 0 8px 0; font-size: 14px; }
.geometry-panel h4 { margin: 8px 0 4px 0; font-size: 12px; color: #c5d1e0; }
.geo-table { border-collapse: collapse; }
.geo-table th, .geo-table td { border: 1px solid #2a2e36; padding: 3px 6px; font-size: 12px; }
"""


def _get_luma_scale(c: Dict[str, Any]):
    """Prefer the pre-computed luma_scale block from tp_measure's JSON. Fall
    back to computing it on the fly for older JSONs that lack the field."""
    fit = c.get("luma_scale")
    if fit is not None:
        return fit
    # Older JSON -- recompute via tp_measure.fit_gray_ramp.
    import tp_measure
    return tp_measure.fit_gray_ramp(c.get("grays") or [])


def render_luma_scale_analysis(captures: List[Dict[str, Any]]) -> str:
    rows = []
    for c in captures:
        fit = _get_luma_scale(c)
        cap_name = _basename(c["_meta"]["capture"])
        if fit is None:
            rows.append(
                f"<tr><td>{_h.escape(cap_name)}</td>"
                f"<td colspan='7' class='muted'>insufficient gray data</td></tr>"
            )
            continue
        pred_black = fit.get("predicted_y10_at_black") or fit.get("pred_at_black")
        pred_white = fit.get("predicted_y10_at_white") or fit.get("pred_at_white")
        rms = {
            "linear gain": fit["rms_linear"],
            "pedestal A (no-setup decoder, signal has setup)": fit["rms_pedestal_a"],
            "pedestal B (setup-decoder, signal has none)": fit["rms_pedestal_b"],
        }
        best = min(rms, key=rms.get)
        if fit["rms_linear"] < 2.0 and abs(fit["gain_pct_loss"]) < 1.0:
            verdict = "<span class='ok'>essentially correct</span>"
        elif best.startswith("linear"):
            verdict = (
                f"<span class='warn'>luma gain {fit['gain_pct_loss']:+.1f}%</span>"
                f"<div class='small'>black ≈ correct, white compressed</div>"
            )
        else:
            verdict = f"<span class='bad'>pedestal issue: {best}</span>"
        rows.append(
            f"<tr>"
            f"<td>{_h.escape(cap_name)}</td>"
            f"<td>{fit['slope']:.4f}</td>"
            f"<td>{fit['intercept']:+.2f}</td>"
            f"<td>{pred_black:+.1f}<div class='small muted'>vs 64</div></td>"
            f"<td>{pred_white:+.1f}<div class='small muted'>vs 940</div></td>"
            f"<td>{fit['rms_linear']:.2f}</td>"
            f"<td class='muted'>{fit['rms_pedestal_a']:.1f} / {fit['rms_pedestal_b']:.1f}</td>"
            f"<td>{verdict}</td>"
            f"</tr>"
        )
    return f"""
<section class="luma">
  <h2>Luma Scale Analysis</h2>
  <p class="legend">
    Linear fit of measured Y10 against ideal Y10 across the 4 gray steps:
    <code>Y_meas = slope · Y_ideal + intercept</code>.
    A pure luma-gain error gives slope ≠ 1 with intercept near 0.
    A classical NTSC pedestal mismatch would instead produce large residuals
    against the linear model and a closer match to one of the pedestal hypotheses
    — these residuals are reported alongside for comparison.
  </p>
  <table class="data luma-table">
    <thead>
      <tr>
        <th>Capture</th>
        <th>Slope</th>
        <th>Intercept</th>
        <th>Predicted at black (Y10=64)</th>
        <th>Predicted at white (Y10=940)</th>
        <th>Linear-fit RMS</th>
        <th>Pedestal&nbsp;A / B RMS</th>
        <th>Verdict</th>
      </tr>
    </thead>
    <tbody>
{''.join(rows)}
    </tbody>
  </table>
</section>
"""


def _delta_class_offset(value, green_lt, yellow_lt):
    """Color class for an absolute offset/skew value."""
    a = abs(value) if value is not None else 0
    if a < green_lt:
        return "delta-good"
    if a < yellow_lt:
        return "delta-warn"
    return "delta-bad"


def render_geometry_section(captures: List[Dict[str, Any]]) -> str:
    panels = []
    for c in captures:
        cap_name = _basename(c["_meta"]["capture"])
        g = c.get("geometry")
        if g is None:
            panels.append(
                f"<div class='geometry-panel'><h3>{_h.escape(cap_name)}</h3>"
                f"<p class='muted'>no geometry block (older JSON)</p></div>"
            )
            continue
        d = g["derived"]
        flag = g.get("quality_flag", "?")
        flag_class = {"ok": "ok", "warn": "warn", "partial": "warn",
                      "failed": "bad"}.get(flag, "")
        box = d.get("active_picture_box")
        if box is not None:
            offset = d.get("picture_offset_from_ideal", {})
            skew = d.get("corner_skew_px", {})
            box_html = (
                f"<table class='geo-table'>"
                f"<tr><th>top</th><th>left</th><th>bottom</th><th>right</th>"
                f"<th>w&times;h</th></tr>"
                f"<tr><td>{box['top']:.1f}</td><td>{box['left']:.1f}</td>"
                f"<td>{box['bottom']:.1f}</td><td>{box['right']:.1f}</td>"
                f"<td>{d['picture_extent_px']['width']:.1f}&times;"
                f"{d['picture_extent_px']['height']:.1f}</td></tr></table>"
                f"<div class='small'>"
                f"offset <span class='{_delta_class_offset(offset.get('dx', 0), 2, 5)}'>"
                f"dx={offset.get('dx', 0):+.1f}</span> "
                f"<span class='{_delta_class_offset(offset.get('dy', 0), 2, 5)}'>"
                f"dy={offset.get('dy', 0):+.1f}</span>"
                f" &nbsp; skew "
                f"<span class='{_delta_class_offset(skew.get('top_vs_bottom_width_diff', 0), 2, 5)}'>"
                f"w_diff={skew.get('top_vs_bottom_width_diff', 0):.1f}</span> "
                f"<span class='{_delta_class_offset(skew.get('left_vs_right_height_diff', 0), 2, 5)}'>"
                f"h_diff={skew.get('left_vs_right_height_diff', 0):.1f}</span>"
                f"</div>"
            )
        else:
            box_html = "<p class='muted'>picture box not derivable</p>"

        clip = d.get("clip_detected") or {}
        clip_rows = []
        for tid in ("TL", "TR", "BL", "BR"):
            entry = clip.get(tid, {})
            visible = entry.get("apex_visible", False)
            interp = entry.get("interpretation", "?")
            cls = "ok" if visible else "bad"
            clip_rows.append(
                f"<tr><td>{tid}</td>"
                f"<td class='{cls}'>{'visible' if visible else 'clipped'}</td>"
                f"<td>{_h.escape(interp)}</td></tr>"
            )
        clip_html = (
            f"<table class='geo-table'><tr><th>Corner</th>"
            f"<th>Apex</th><th>Interpretation</th></tr>"
            + "".join(clip_rows) + "</table>"
        )

        cross_off = d.get("cross_offset_from_ideal") or [None, None]
        aperture = d.get("aperture_symmetry")
        if cross_off[0] is not None and aperture is not None:
            cross_html = (
                f"<div class='small'>"
                f"offset dx={cross_off[0]:+.2f}, dy={cross_off[1]:+.2f}"
                f" &nbsp; aperture_symmetry={aperture:.3f}"
                f"</div>"
            )
        else:
            cross_html = "<p class='muted'>cross missing</p>"

        aspect = d.get("aspect_ratio_check")
        dvp = d.get("diameter_vs_picture_height")
        circle_fit_rms = d.get("circle_fit_rms")
        if aspect is not None:
            dvp_str = f"{dvp:.3f}" if dvp is not None else "n/a"
            rms_str = f"{circle_fit_rms:.2f}px" if circle_fit_rms is not None else "n/a"
            circle_html = (
                f"<div class='small'>"
                f"aspect_ratio_check={aspect:.4f} &nbsp; "
                f"diameter_vs_picture_height={dvp_str} &nbsp; "
                f"fit_rms={rms_str}"
                f"</div>"
            )
        else:
            circle_html = "<p class='muted'>circle missing</p>"

        refit = g.get("registration_refit", {})
        mean_res = refit.get("final_residuals_px", {}).get("mean", float("nan"))
        refit_html = (
            f"<div class='small'>"
            f"inliers: {refit.get('inlier_count_initial', '?')} "
            f"&rarr; <b>{refit.get('inlier_count_final', '?')}</b> "
            f" &nbsp; mean_residual={mean_res:.2f}px"
            f"</div>"
        )

        panels.append(
            f"<div class='geometry-panel'>"
            f"<h3>{_h.escape(cap_name)} "
            f"<span class='{flag_class}'>[{flag}]</span></h3>"
            f"<h4>Picture box</h4>{box_html}"
            f"<h4>Clip detection</h4>{clip_html}"
            f"<h4>Registration cross</h4>{cross_html}"
            f"<h4>Black circle</h4>{circle_html}"
            f"<h4>Refit benefit</h4>{refit_html}"
            f"</div>"
        )

    return f"""
<section class="geometry">
  <h2>Geometry</h2>
  <p class="legend">
    Picture-in-raster geometry from the SW2 boundary triangles, picture-
    centered registration cross, and black-ring circle. Active picture box
    is bounded by the 4 triangle apexes (inferred from their back corners
    when the apex is clipped). Aspect / aperture / fit_rms surface
    decoder-side geometry artifacts; clip detection flags overscan or
    letterboxing.
  </p>
  {''.join(panels)}
</section>
"""


def render_page(captures: List[Dict[str, Any]]) -> str:
    title = f"SW2 Comparison — {len(captures)} captures"
    sections = (
        render_registration_summary(captures)
        + render_geometry_section(captures)
        + render_tartan_deltas(captures)
        + render_gray_deltas(captures)
        + render_luma_scale_analysis(captures)
    )
    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{_h.escape(title)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>{_CSS}</style>
</head><body>
<h1>{_h.escape(title)}</h1>
{sections}
</body></html>
"""


def _main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("inputs", nargs="+", help="per-capture JSON files from tp_measure")
    p.add_argument("--output", required=True, help="output HTML path")
    args = p.parse_args()
    captures = []
    for path in args.inputs:
        with open(path) as f:
            captures.append(json.load(f))
    html = render_page(captures)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"wrote {args.output} ({len(captures)} captures)")


if __name__ == "__main__":
    _main()
