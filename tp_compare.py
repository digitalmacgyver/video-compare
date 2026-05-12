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
.diag-panel { margin: 16px 0; padding: 8px; background: #1d2026; border: 1px solid #2a2e36; }
.diag-panel h3 { margin: 4px 0 8px 0; font-size: 14px; }
.diag-zoom-readout { font-size: 11px; color: #8a929f; font-weight: 400; margin-left: 6px; }
.diag-zoom-container { position: relative; overflow: hidden; cursor: grab; outline: none;
    border: 1px solid #2a2e36; background: #000; width: 100%; height: 720px; }
.diag-zoom-container:focus { outline: 1px solid #61c08f; }
.diag-img { display: block; max-width: none; transform-origin: 0 0; image-rendering: pixelated;
    user-select: none; -webkit-user-drag: none; }
.diag-coord-readout { position: absolute; top: 4px; right: 4px; background: rgba(0,0,0,0.75);
    color: #fff; padding: 2px 8px; font-family: monospace; font-size: 12px;
    pointer-events: none; border-radius: 3px; }
kbd { background: #2a2e36; padding: 1px 6px; border-radius: 3px; font-family: monospace;
    font-size: 11px; border: 1px solid #3a3e46; }
ul.legend { font-size: 12px; color: #b8c0cc; line-height: 1.7; margin: 6px 0 12px 18px; }
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


def render_sample_diagnostics(captures: List[Dict[str, Any]]) -> str:
    """Appendix: per-capture overlay PNG showing where each region was
    sampled in capture coords. Embeds the PNG as a base64 data URL if a
    sibling `<json_stem>_overlay.png` exists next to the source JSON.

    Each capture dict may carry a `_source_json_path` attribute set by
    the CLI; if absent, the section is skipped for that capture.
    """
    import base64
    panels = []
    for c in captures:
        src = c.get("_source_json_path")
        cap_name = _basename(c["_meta"]["capture"])
        if not src:
            continue
        stem = os.path.splitext(src)[0]
        png_path = stem + "_overlay.png"
        if not os.path.exists(png_path):
            panels.append(
                f"<div class='diag-panel'><h3>{_h.escape(cap_name)}</h3>"
                f"<p class='muted'>no overlay PNG at {_h.escape(png_path)}</p></div>"
            )
            continue
        with open(png_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        panels.append(
            f"<div class='diag-panel'>"
            f"<h3>{_h.escape(cap_name)} "
            f"<span class='diag-zoom-readout'></span></h3>"
            f"<div class='diag-zoom-container' tabindex='0'>"
            f"<img class='diag-img' src='data:image/png;base64,{b64}' draggable='false'/>"
            f"<div class='diag-coord-readout'>(–, –)</div>"
            f"</div>"
            f"</div>"
        )
    if not panels:
        return ""
    return f"""
<section class="diagnostics">
  <h2>Appendix: Sampling Diagnostics</h2>
  <p class="legend">
    Each frame is the actual captured image after raster padding, with
    overlays at the positions where measurements were taken. Use this
    to spot-check whether a sample window straddles the wrong chart
    region (which would explain anomalous deltas in the tables above).
    <br><b>Controls:</b> hover the image, then mouse-wheel to zoom
    centered on the cursor, click-and-drag to pan, press
    <kbd>R</kbd> to reset.
  </p>
  <ul class="legend">
    <li><span style='color:#0ff'>cyan</span> boxes: tartan sample windows (with region ID).</li>
    <li><span style='color:#ff0'>yellow</span> boxes: gray-step sample windows.</li>
    <li><span style='color:#f0f'>magenta</span> crosses: grid-landmark expected positions; bright magenta = used in final fit, dim = detected but rejected by RANSAC.</li>
    <li><span style='color:#0f0'>green</span> crosses: detected Stage 2 fiducials (boundary triangle back corners, registration cross, circle ellipse).</li>
    <li><span style='color:rgb(255,150,0)'>orange</span> crosses: detected triangle apex (vs the green inferred apex).</li>
  </ul>
  {''.join(panels)}
</section>
<script>
(function() {{
    function initZoom(container) {{
        var img = container.querySelector('.diag-img');
        var readout = container.parentElement.querySelector('.diag-zoom-readout');
        var coordReadout = container.querySelector('.diag-coord-readout');
        var scale = 1, tx = 0, ty = 0;
        var panning = false, lastX = 0, lastY = 0;
        function apply() {{
            img.style.transform = 'translate(' + tx + 'px, ' + ty + 'px) scale(' + scale + ')';
            if (readout) readout.textContent = '[zoom ' + scale.toFixed(2) + 'x]';
        }}
        function reset() {{ scale = 1; tx = 0; ty = 0; apply(); }}
        function updateCoord(e) {{
            if (!coordReadout) return;
            var rect = container.getBoundingClientRect();
            var mx = e.clientX - rect.left;
            var my = e.clientY - rect.top;
            // Inverse of: container_x = image_x * scale + tx
            var imgX = (mx - tx) / scale;
            var imgY = (my - ty) / scale;
            if (imgX >= 0 && imgX < img.naturalWidth && imgY >= 0 && imgY < img.naturalHeight) {{
                coordReadout.textContent = '(' + Math.round(imgX) + ', ' + Math.round(imgY) + ')';
            }} else {{
                coordReadout.textContent = '(–, –)';
            }}
        }}
        container.addEventListener('wheel', function(e) {{
            e.preventDefault();
            var rect = container.getBoundingClientRect();
            var mx = e.clientX - rect.left;
            var my = e.clientY - rect.top;
            var oldScale = scale;
            var factor = e.deltaY < 0 ? 1.25 : 1.0 / 1.25;
            scale = Math.max(0.25, Math.min(40, scale * factor));
            tx = mx - (mx - tx) * (scale / oldScale);
            ty = my - (my - ty) * (scale / oldScale);
            apply();
        }}, {{ passive: false }});
        container.addEventListener('mousedown', function(e) {{
            panning = true; lastX = e.clientX; lastY = e.clientY;
            container.style.cursor = 'grabbing';
            container.focus();
            e.preventDefault();
        }});
        window.addEventListener('mousemove', function(e) {{
            if (panning) {{
                tx += e.clientX - lastX;
                ty += e.clientY - lastY;
                lastX = e.clientX; lastY = e.clientY;
                apply();
            }}
        }});
        container.addEventListener('mousemove', updateCoord);
        container.addEventListener('mouseleave', function() {{
            if (coordReadout) coordReadout.textContent = '(–, –)';
        }});
        window.addEventListener('mouseup', function() {{
            if (panning) {{ panning = false; container.style.cursor = 'grab'; }}
        }});
        container.addEventListener('mouseenter', function() {{ container.focus(); }});
        container.addEventListener('keydown', function(e) {{
            if (e.key === 'r' || e.key === 'R') {{ reset(); e.preventDefault(); }}
        }});
        // Double-click toggles zoom: identity <-> 4x at cursor.
        container.addEventListener('dblclick', function(e) {{
            var rect = container.getBoundingClientRect();
            var mx = e.clientX - rect.left;
            var my = e.clientY - rect.top;
            if (scale > 1.05) {{
                reset();
            }} else {{
                var oldScale = scale;
                scale = 4;
                tx = mx - (mx - tx) * (scale / oldScale);
                ty = my - (my - ty) * (scale / oldScale);
                apply();
            }}
        }});
        apply();
    }}
    document.querySelectorAll('.diag-zoom-container').forEach(initZoom);
}})();
</script>
"""


def render_frequency_response(captures: List[Dict[str, Any]]) -> str:
    if not any(c.get("frequency_response") for c in captures):
        return ""
    rows = ['<section class="freq-response"><h2>Frequency response (Stage 3)</h2>']
    rows.append('<table class="freq-table"><thead><tr><th>Clip</th>'
                '<th>−3 dB (MHz)</th><th>−6 dB (MHz)</th></tr></thead><tbody>')
    for c in captures:
        fr = c.get("frequency_response") or {}
        s = (fr.get("summary") or {}) if isinstance(fr, dict) else {}
        m3 = s.get("minus_3db_freq_MHz")
        m6 = s.get("minus_6db_freq_MHz")
        name = _basename(c["_meta"].get("capture_path", "?"))
        m3_str = f"{m3:.2f}" if m3 is not None else "—"
        m6_str = f"{m6:.2f}" if m6 is not None else "—"
        rows.append(f"<tr><td>{_h.escape(name)}</td>"
                    f"<td>{m3_str}</td><td>{m6_str}</td></tr>")
    rows.append("</tbody></table>")
    datasets = []
    for c in captures:
        fr = c.get("frequency_response") or {}
        if not isinstance(fr, dict):
            continue
        curve = (fr.get("summary") or {}).get("luma_response_curve", []) or []
        if not curve:
            continue
        name = _basename(c["_meta"].get("capture_path", "?"))
        datasets.append({
            "label": name,
            "data": [{"x": float(f), "y": float(d)} for f, d in curve],
        })
    rows.append('<canvas id="freqChart" height="160"></canvas>')
    rows.append("<script>")
    rows.append(f"const FREQ_DATA = {json.dumps(datasets)};")
    rows.append("""new Chart(document.getElementById('freqChart').getContext('2d'), {
        type: 'line',
        data: {datasets: FREQ_DATA.map(d => Object.assign({}, d,
                                                          {fill:false, tension:0.2, parsing:false}))},
        options: {
            parsing: false,
            scales: {x: {type: 'linear',
                         title:{display:true, text:'Frequency (MHz)'}},
                     y: {title:{display:true, text:'Modulation (dB)'}}},
            plugins: {legend:{position:'top'}}
        }
    });""")
    rows.append("</script></section>")
    return "\n".join(rows)


def render_artifacts(captures: List[Dict[str, Any]]) -> str:
    if not any(c.get("artifacts") for c in captures):
        return ""
    rows = ['<section class="artifacts"><h2>Decoder artifacts (Stage 3)</h2>']
    rows.append('<table class="artifact-table"><thead><tr>'
                '<th>Clip</th>'
                '<th>Hanging dots (Y pp)</th>'
                '<th>Dot crawl (chroma RMS)</th>'
                '<th>Cross-color (chroma RMS)</th>'
                '<th>Cross-luma (Y pp)</th>'
                '<th>Zone-plate chroma RMS</th>'
                '<th>ZP chroma present?</th>'
                '</tr></thead><tbody>')
    for c in captures:
        a = c.get("artifacts") or {}
        s = (a.get("summary") or {}) if isinstance(a, dict) else {}
        zp_present = s.get("zone_plate_chroma_present", False)
        name = _basename(c["_meta"].get("capture_path", "?"))
        rows.append(
            "<tr>"
            f"<td>{_h.escape(name)}</td>"
            f"<td>{s.get('max_hanging_dots_y10_pp', 0):.1f}</td>"
            f"<td>{s.get('max_dot_crawl_chroma_rms', 0):.2f}</td>"
            f"<td>{s.get('max_cross_color_chroma_rms', 0):.2f}</td>"
            f"<td>{s.get('max_cross_luma_y10_pp', 0):.1f}</td>"
            f"<td>{s.get('zone_plate_chroma_rms', 0):.2f}</td>"
            f"<td>{'YES' if zp_present else 'no'}</td>"
            "</tr>"
        )
    rows.append("</tbody></table></section>")
    return "\n".join(rows)


def render_decoder_class(captures: List[Dict[str, Any]]) -> str:
    if not any(c.get("decoder_class") for c in captures):
        return ""
    rows = ['<section class="decoder-class"><h2>Decoder class (Stage 3)</h2>']
    rows.append('<table class="decoder-table"><thead><tr>'
                '<th>Clip</th><th>Class</th><th>Confidence</th>'
                '<th>Candidate confidences</th>'
                '</tr></thead><tbody>')
    for c in captures:
        dc = c.get("decoder_class") or {}
        cls = dc.get("decoder_class", "—")
        conf = float(dc.get("confidence", 0.0))
        ev = (dc.get("evidence") or {})
        cands = ev.get("candidate_confidences", {}) or {}
        cands_str = ", ".join(f"{k}: {v:.2f}" for k, v in cands.items())
        name = _basename(c["_meta"].get("capture_path", "?"))
        rows.append(
            "<tr>"
            f"<td>{_h.escape(name)}</td>"
            f"<td><strong>{_h.escape(str(cls))}</strong></td>"
            f"<td>{conf:.2f}</td>"
            f"<td>{_h.escape(cands_str)}</td>"
            "</tr>"
        )
    rows.append("</tbody></table></section>")
    return "\n".join(rows)


def render_page(captures: List[Dict[str, Any]]) -> str:
    title = f"SW2 Comparison — {len(captures)} captures"
    sections = (
        render_registration_summary(captures)
        + render_geometry_section(captures)
        + render_tartan_deltas(captures)
        + render_gray_deltas(captures)
        + render_luma_scale_analysis(captures)
        + render_frequency_response(captures)
        + render_artifacts(captures)
        + render_decoder_class(captures)
        + render_sample_diagnostics(captures)
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
            cap = json.load(f)
        cap["_source_json_path"] = path
        captures.append(cap)
    html = render_page(captures)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"wrote {args.output} ({len(captures)} captures)")


if __name__ == "__main__":
    _main()
