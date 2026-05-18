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


def _th(label: str, tip: str) -> str:
    """Column header with a hover tooltip explaining what the column means.

    Uses a CSS-driven popover (see `.tip:hover::after` in _CSS) for instant,
    reliable display instead of the native `title=` attribute, which most
    browsers show only after a ~1 s delay and dismiss after a few seconds.
    aria-label is still set so screen readers get the explanation.
    """
    tip_esc = _h.escape(tip)
    return f'<th class="tip" data-tip="{tip_esc}" aria-label="{tip_esc}">{label}</th>'


# Friendly tooltips for tartan and gray region IDs. Keys are region ids,
# values are short tooltip strings.
_REGION_TOOLTIPS = {
    "YEL":  "Top-row 75% yellow tartan patch (cell 1,1)",
    "CYN":  "Top-row 75% cyan tartan patch (cell 1,1)",
    "BLU":  "Top-row 75% blue tartan patch (cell 1,1)",
    "RED":  "Top-row 75% red tartan patch (cell 1,1)",
    "MAG":  "Bottom-row 75% magenta tartan patch (cell 1,1)",
    "GRN":  "Bottom-row 75% green tartan patch (cell 1,1)",
    "RED2": "Bottom-row 75% red tartan patch — duplicate color, position used for vertical-transition test",
    "CYN2": "Bottom-row 75% cyan tartan patch — duplicate color",
    "G1":   "20% IRE gray step (chart spec Y10≈239)",
    "G2":   "40% IRE gray step (Y10≈414)",
    "G3":   "60% IRE gray step (Y10≈590)",
    "G4":   "80% IRE gray step (Y10≈765)",
}


def _region_th(rid: str) -> str:
    tip = _REGION_TOOLTIPS.get(rid, f"Region {rid}")
    return _th(rid, tip)


def _render_fiducial_crops(capture: Dict[str, Any]) -> str:
    """Embed the sibling _fiducials.png inline as a base64 data URL.

    The PNG is produced by tp_measure (which calls tp_fiducial_crops.build)
    next to the source JSON: <stem>_fiducials.png. Falls back to a muted
    'not available' note if the file is missing — older JSONs predate the
    fiducial-crops feature.
    """
    import base64
    src = capture.get("_source_json_path")
    if not src:
        return ""
    stem = os.path.splitext(src)[0]
    png_path = stem + "_fiducials.png"
    if not os.path.exists(png_path):
        return (
            "<h4>Fiducial crops</h4>"
            f"<p class='muted small'>no fiducial-crops PNG at {_h.escape(png_path)}"
            " — re-run tp_measure to generate.</p>"
        )
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return (
        "<h4>Fiducial crops "
        "<span class='small muted'>(green = back corner; orange = inferred apex; "
        "yellow = detected apex; cyan = circle ring sample)</span></h4>"
        f"<img class='fiducial-crops' "
        f"src='data:image/png;base64,{b64}' alt='fiducial crops'/>"
    )


# ---------------------------------------------------------------------
# Top-of-report overview tables (sortable). Each row is one capture; the
# columns are the small handful of headline metrics a reader is most
# likely to want to compare across processors.
# ---------------------------------------------------------------------

def _sortable_th(label: str, tip: str, kind: str = "num") -> str:
    tip_esc = _h.escape(tip)
    return (
        f'<th class="tip sortable" data-tip="{tip_esc}" data-sort="{kind}" '
        f'aria-label="{tip_esc}">{label}</th>'
    )


_OVERVIEW_SORT_JS = """
<script>
(function () {
  document.querySelectorAll('table.overview-table').forEach(function (tbl) {
    var ths = tbl.querySelectorAll('th.sortable');
    ths.forEach(function (th, idx) {
      th.addEventListener('click', function () {
        var asc = !th.classList.contains('sort-asc');
        ths.forEach(function (other) {
          other.classList.remove('sort-asc');
          other.classList.remove('sort-desc');
        });
        th.classList.toggle('sort-asc',  asc);
        th.classList.toggle('sort-desc', !asc);
        var kind = th.getAttribute('data-sort') || 'num';
        var rows = Array.prototype.slice.call(
          tbl.querySelectorAll('tbody tr'));
        rows.sort(function (a, b) {
          var av = a.children[idx].getAttribute('data-v');
          var bv = b.children[idx].getAttribute('data-v');
          if (kind === 'num') {
            av = parseFloat(av); bv = parseFloat(bv);
            if (isNaN(av)) av = -Infinity;
            if (isNaN(bv)) bv = -Infinity;
            return asc ? av - bv : bv - av;
          }
          av = (av || '').toLowerCase();
          bv = (bv || '').toLowerCase();
          return asc ? av.localeCompare(bv) : bv.localeCompare(av);
        });
        var tbody = tbl.querySelector('tbody');
        rows.forEach(function (r) { tbody.appendChild(r); });
      });
    });
  });
})();
</script>
"""


def _td_num(value, fmt: str = "{:.2f}", cls: str = "") -> str:
    if value is None or (isinstance(value, float)
                         and (value != value)):  # NaN
        return (f'<td class="numeric {cls}" data-v="">'
                f'<span class="muted">&mdash;</span></td>')
    return (f'<td class="numeric {cls}" data-v="{value}">'
            f'{fmt.format(value)}</td>')


def _td_name(name: str) -> str:
    safe = _h.escape(name)
    return f'<td class="name" data-v="{safe}">{safe}</td>'


def _sum_geom_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a per-capture geometry.derived.summary to the headline
    numbers shown in the overview table."""
    out = {
        "max_arrow_delta_px": None,
        "center_offset_mag_px": None,
        "h_scale_pct":  None,
        "v_scale_pct":  None,
        "keystone_max_px": None,
        "displayed_circularity": None,
    }
    if not summary:
        return out
    sp = summary.get("arrow_spacings_px") or {}
    if sp:
        out["max_arrow_delta_px"] = max(abs(sp[k]["delta"]) for k in sp)
    off = summary.get("picture_center_offset_px") or {}
    if "dx" in off and "dy" in off:
        out["center_offset_mag_px"] = (off["dx"] ** 2 + off["dy"] ** 2) ** 0.5
    sc = summary.get("picture_scale_pct") or {}
    out["h_scale_pct"] = sc.get("horizontal")
    out["v_scale_pct"] = sc.get("vertical")
    ks = summary.get("keystone_px") or {}
    if ks:
        out["keystone_max_px"] = max(
            abs(ks.get("horizontal_top_minus_bottom", 0.0)),
            abs(ks.get("vertical_left_minus_right",   0.0)),
        )
    circ = summary.get("circle") or {}
    out["displayed_circularity"] = circ.get("displayed_circularity")
    return out


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _score_geometry(c: Dict[str, Any]) -> float:
    """0-100 composite. 100 = no detectable picture-in-raster distortion.
    Penalties: arrowhead spacing error, scale error, keystone, circularity."""
    g = c.get("geometry") or {}
    summary = (g.get("derived") or {}).get("summary")
    m = _sum_geom_metrics(summary)
    if m["max_arrow_delta_px"] is None:
        return float("nan")
    pen = 0.0
    pen += _clamp(m["max_arrow_delta_px"] * 4.0, 0, 40)
    if m["h_scale_pct"] is not None:
        pen += _clamp(abs(m["h_scale_pct"] - 100.0) * 3.0, 0, 15)
    if m["v_scale_pct"] is not None:
        pen += _clamp(abs(m["v_scale_pct"] - 100.0) * 3.0, 0, 15)
    if m["keystone_max_px"] is not None:
        pen += _clamp(m["keystone_max_px"] * 2.0, 0, 20)
    if m["displayed_circularity"] is not None:
        pen += _clamp(abs(m["displayed_circularity"] - 1.0) * 200.0, 0, 30)
    return max(0.0, 100.0 - pen)


def _score_color(c: Dict[str, Any]) -> float:
    """0-100. 100 = every tartan patch matches its spec exactly."""
    patches = c.get("tartan") or []
    if not patches:
        return float("nan")
    mean_de = sum(_color_delta_e(p) for p in patches) / len(patches)
    return max(0.0, 100.0 - _clamp(mean_de * 1.5, 0, 100))


def _score_grayscale(c: Dict[str, Any]) -> float:
    """0-100. 100 = grayscale lands exactly on the chart-spec curve with
    no chroma cast."""
    grays = c.get("grays") or []
    if not grays:
        return float("nan")
    mean_abs_dy = sum(abs(g["delta_y10"]) for g in grays) / len(grays)
    max_cast = max(_gray_chroma_cast(g) for g in grays)
    pen = _clamp(mean_abs_dy * 2.5, 0, 80) + _clamp(max_cast * 0.6, 0, 20)
    return max(0.0, 100.0 - pen)


def _score_class(value) -> str:
    """Class for a 0-100 score where higher is better."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if value >= 85:
        return "delta-good"
    if value >= 60:
        return "delta-warn"
    return "delta-bad"


def render_overall_summary(captures: List[Dict[str, Any]]) -> str:
    """Composite scoreboard at the very top of the report. One row per
    capture, with per-category 0-100 scores and an overall (mean of the
    available categories). Sortable so you can rank processors on any
    dimension."""
    head = (
        "<tr>"
        + _sortable_th("Capture",
                       "Capture file name.", kind="text")
        + _sortable_th("Geometry",
                       "0-100 score combining arrowhead-spacing error, "
                       "picture scale, keystone, and PAR-aware "
                       "circularity. Higher = closer to a perfect raster.")
        + _sortable_th("Color",
                       "0-100 score from the mean Euclidean YUV10 distance "
                       "between the 8 tartan patches and chart spec. "
                       "Higher = better color match.")
        + _sortable_th("Grayscale",
                       "0-100 score from mean |ΔY| across the 4 IRE steps, "
                       "with an extra penalty for chroma cast on neutral "
                       "patches.")
        + _sortable_th("Overall",
                       "Mean of the available category scores.")
        + "</tr>"
    )
    rows = []
    for c in captures:
        name = _basename(c["_meta"]["capture"])
        g  = _score_geometry(c)
        co = _score_color(c)
        gs = _score_grayscale(c)
        vals = [v for v in (g, co, gs)
                if v is not None and not (isinstance(v, float) and v != v)]
        overall = sum(vals) / len(vals) if vals else float("nan")
        cells = [
            _td_name(name),
            _td_num(g,       "{:.1f}", cls=_score_class(g)),
            _td_num(co,      "{:.1f}", cls=_score_class(co)),
            _td_num(gs,      "{:.1f}", cls=_score_class(gs)),
            _td_num(overall, "{:.1f}", cls=_score_class(overall)),
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<section class="overview overall">
  <h2>Overall Summary</h2>
  <p class="legend">
    Composite at-a-glance scores per processor. Each category is on a
    0-100 scale where 100 = no measurable error. Click any column to
    rank. See the per-section overviews below for the underlying
    numbers.
  </p>
  <table class="overview-table">
    <thead>{head}</thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def render_geometry_overview(captures: List[Dict[str, Any]]) -> str:
    head = (
        "<tr>"
        + _sortable_th("Capture",
                       "Capture file name.", kind="text")
        + _sortable_th("Max arrow Δ (px)",
                       "Largest absolute deviation of any of the four "
                       "arrowhead spacings (top/bottom/left/right) vs the "
                       "chart spec. Lower = closer to a faithful raster.")
        + _sortable_th("Center offset (px)",
                       "Euclidean distance from the four-apex midpoint to "
                       "the ideal raster center.")
        + _sortable_th("H scale %",
                       "Mean horizontal arrowhead spacing / ideal × 100. "
                       "100% = correct.")
        + _sortable_th("V scale %",
                       "Mean vertical arrowhead spacing / ideal × 100. "
                       "100% = correct.")
        + _sortable_th("Keystone (px)",
                       "Max of |top−bottom width| and |left−right height|. "
                       "0 = no keystone.")
        + _sortable_th("Circle disp.",
                       "Displayed circularity (PAR-aware). 1.00 = round in "
                       "display; >1 = horizontally stretched; <1 = "
                       "vertically stretched.")
        + "</tr>"
    )
    rows = []
    for c in captures:
        name = _basename(c["_meta"]["capture"])
        g = c.get("geometry") or {}
        summary = (g.get("derived") or {}).get("summary")
        m = _sum_geom_metrics(summary)
        cells = [
            _td_name(name),
            _td_num(m["max_arrow_delta_px"], "{:.1f}",
                    cls=_delta_class_offset(m["max_arrow_delta_px"] or 0,
                                            2, 5)),
            _td_num(m["center_offset_mag_px"], "{:.2f}",
                    cls=_delta_class_offset(m["center_offset_mag_px"] or 0,
                                            2, 5)),
            _td_num(m["h_scale_pct"], "{:.2f}",
                    cls=_delta_class_offset(((m["h_scale_pct"] or 100.0)
                                             - 100.0), 1, 3)),
            _td_num(m["v_scale_pct"], "{:.2f}",
                    cls=_delta_class_offset(((m["v_scale_pct"] or 100.0)
                                             - 100.0), 1, 3)),
            _td_num(m["keystone_max_px"], "{:.1f}",
                    cls=_delta_class_offset(m["keystone_max_px"] or 0,
                                            2, 5)),
            _td_num(m["displayed_circularity"], "{:.3f}",
                    cls=_delta_class_offset(((m["displayed_circularity"]
                                              or 1.0) - 1.0) * 100,
                                            2, 5)),
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<section class="overview">
  <h2>Geometry Overview</h2>
  <p class="legend">
    Click any column header to sort. Color-coded against the same
    thresholds the per-source panels below use.
  </p>
  <table class="overview-table">
    <thead>{head}</thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def _color_delta_e(p: Dict[str, Any]) -> float:
    dy, du, dv = p["delta_yuv10"]
    return float((dy * dy + du * du + dv * dv) ** 0.5)


def _color_class(delta_e: float) -> str:
    if delta_e < 10:
        return "delta-good"
    if delta_e < 30:
        return "delta-warn"
    return "delta-bad"


def _gray_chroma_cast(g: Dict[str, Any]) -> float:
    """Magnitude of chroma offset for a gray step (0 = neutral)."""
    du = (g.get("u10", 512) - 512)
    dv = (g.get("v10", 512) - 512)
    return float((du * du + dv * dv) ** 0.5)


def render_tartan_overview(captures: List[Dict[str, Any]]) -> str:
    head = (
        "<tr>"
        + _sortable_th("Capture",
                       "Capture file name.", kind="text")
        + _sortable_th("Mean ΔE",
                       "Average Euclidean YUV10 distance from chart spec "
                       "across all 8 tartan colors. Lower = better.")
        + _sortable_th("Max ΔE",
                       "Largest YUV10 distance from spec across the 8 "
                       "colors.")
        + _sortable_th("Worst color",
                       "Tartan color with the largest YUV10 distance.",
                       kind="text")
        + _sortable_th("Mean sat %",
                       "Mean of sat_pct_vs_ideal across the 8 colors. "
                       "100% = correct saturation, <100% = desaturated, "
                       ">100% = oversaturated.")
        + "</tr>"
    )
    rows = []
    for c in captures:
        name = _basename(c["_meta"]["capture"])
        patches = c.get("tartan") or []
        if not patches:
            rows.append(f"<tr>{_td_name(name)}"
                        + _td_num(None) * 3
                        + '<td class="muted" data-v="">—</td>'
                        + _td_num(None) + "</tr>")
            continue
        des = [(p["id"], _color_delta_e(p)) for p in patches]
        mean_de = sum(d for _, d in des) / len(des)
        worst_id, max_de = max(des, key=lambda r: r[1])
        sats = [p.get("sat_pct_vs_ideal", 100.0) for p in patches]
        mean_sat = sum(sats) / len(sats)
        cells = [
            _td_name(name),
            _td_num(mean_de, "{:.1f}", cls=_color_class(mean_de)),
            _td_num(max_de,  "{:.1f}", cls=_color_class(max_de)),
            f'<td class="name" data-v="{_h.escape(worst_id)}">'
            f'{_h.escape(worst_id)}</td>',
            _td_num(mean_sat, "{:.1f}",
                    cls=_delta_class_offset(mean_sat - 100.0, 5, 15)),
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<section class="overview">
  <h2>Color (Tartan) Overview</h2>
  <p class="legend">
    Per-capture summary of the 8 SMPTE 75% tartan patches. ΔE here is
    a plain Euclidean YUV10 distance — useful for ranking but not a
    perceptually-uniform color score.
  </p>
  <table class="overview-table">
    <thead>{head}</thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def render_grayscale_overview(captures: List[Dict[str, Any]]) -> str:
    head = (
        "<tr>"
        + _sortable_th("Capture",
                       "Capture file name.", kind="text")
        + _sortable_th("Mean |ΔY|",
                       "Mean absolute Y10 deviation across the 4 gray "
                       "steps (20/40/60/80 % IRE). Lower = better.")
        + _sortable_th("Black floor ΔY",
                       "Signed Y10 deviation at the 20 % IRE step "
                       "(G1). Positive = lifted blacks, negative = "
                       "crushed blacks.")
        + _sortable_th("White ceiling ΔY",
                       "Signed Y10 deviation at the 80 % IRE step "
                       "(G4). Positive = whites brighter than spec; "
                       "negative = rolled-off whites.")
        + _sortable_th("Max chroma cast",
                       "Largest U/V offset from neutral (512) across "
                       "the 4 gray steps. 0 = perfectly neutral grays.")
        + "</tr>"
    )
    rows = []
    for c in captures:
        name = _basename(c["_meta"]["capture"])
        grays = c.get("grays") or []
        if len(grays) < 4:
            rows.append(f"<tr>{_td_name(name)}"
                        + _td_num(None) * 4 + "</tr>")
            continue
        by_id = {g["id"]: g for g in grays}
        steps_in_order = ["G1", "G2", "G3", "G4"]
        deltas = [by_id[s]["delta_y10"] for s in steps_in_order
                  if s in by_id]
        mean_abs = sum(abs(d) for d in deltas) / len(deltas)
        black_floor = by_id["G1"]["delta_y10"]
        white_ceil  = by_id["G4"]["delta_y10"]
        max_cast = max(_gray_chroma_cast(by_id[s]) for s in steps_in_order
                       if s in by_id)
        cells = [
            _td_name(name),
            _td_num(mean_abs, "{:.1f}",
                    cls=_delta_class_offset(mean_abs, 5, 15)),
            _td_num(black_floor, "{:+.1f}",
                    cls=_delta_class_offset(black_floor, 5, 15)),
            _td_num(white_ceil, "{:+.1f}",
                    cls=_delta_class_offset(white_ceil, 5, 15)),
            _td_num(max_cast, "{:.1f}",
                    cls=_delta_class_offset(max_cast, 5, 15)),
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<section class="overview">
  <h2>Grayscale Overview</h2>
  <p class="legend">
    Per-capture summary of the 4-step grayscale (20 / 40 / 60 / 80 % IRE).
    All deltas are in 10-bit luma codes — 10 codes ≈ 1 % luma. The
    chroma-cast column flags non-neutral grays (a comb-decoder smell).
  </p>
  <table class="overview-table">
    <thead>{head}</thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


# ---------------------------------------------------------------------
# Per-video layperson panels for tartan + grayscale (matched in style to
# the geometry panel).
# ---------------------------------------------------------------------

_TARTAN_COLOR_LABELS = {
    "YEL":  "yellow (top row)",
    "CYN":  "cyan (top row)",
    "BLU":  "blue (top row)",
    "RED":  "red (top row)",
    "MAG":  "magenta (bottom row)",
    "GRN":  "green (bottom row)",
    "RED2": "red (bottom row)",
    "CYN2": "cyan (bottom row)",
}


def _swatch_inline(rgb_tuple, role: str) -> str:
    r, g, b = rgb_tuple
    return (f"<span class='swatch-inline {role}' "
            f"style='background-color: rgb({r},{g},{b});' "
            f"title='RGB ({r},{g},{b})'></span>")


def _tartan_verdict(delta_e: float, sat_pct: float) -> str:
    parts = []
    if delta_e < 10:
        parts.append("match")
    elif delta_e < 30:
        parts.append("close")
    else:
        parts.append("off")
    if abs(sat_pct - 100.0) >= 5:
        if sat_pct < 100:
            parts.append(f"{100.0 - sat_pct:.0f}% under-saturated")
        else:
            parts.append(f"{sat_pct - 100.0:.0f}% over-saturated")
    return ", ".join(parts)


def _render_tartan_panel(c: Dict[str, Any]) -> str:
    cap_name = _basename(c["_meta"]["capture"])
    patches = c.get("tartan") or []
    if not patches:
        return (f"<div class='color-panel'><h3>{_h.escape(cap_name)}</h3>"
                f"<p class='muted'>no tartan measurements</p></div>")
    region_ids = [r["id"] for r in tp_chart.TARTAN_REGIONS]
    by_id = {p["id"]: p for p in patches}
    rows = []
    for rid in region_ids:
        p = by_id.get(rid)
        if p is None:
            continue
        ideal_rgb = tp_chart.yuv10_to_rgb8(*p["ideal_yuv10"])
        meas_rgb  = tp_chart.yuv10_to_rgb8(*p["measured_yuv10"])
        dy, du, dv = p["delta_yuv10"]
        de = _color_delta_e(p)
        sat = p.get("sat_pct_vs_ideal", 100.0)
        verdict = _tartan_verdict(de, sat)
        rows.append(
            f"<tr>"
            f"<td class='name'>{rid}</td>"
            f"<td class='muted small'>{_TARTAN_COLOR_LABELS.get(rid, rid)}</td>"
            f"<td class='swatch-cell'>{_swatch_inline(ideal_rgb, 'ideal')}</td>"
            f"<td class='swatch-cell'>{_swatch_inline(meas_rgb, 'measured')}</td>"
            f"<td class='delta'>ΔY {dy:+.1f}</td>"
            f"<td class='delta'>ΔU {du:+.1f}</td>"
            f"<td class='delta'>ΔV {dv:+.1f}</td>"
            f"<td class='delta {_color_class(de)}'>ΔE {de:.1f}</td>"
            f"<td class='delta {_delta_class_offset(sat - 100.0, 5, 15)}'>"
            f"{sat:.1f}%</td>"
            f"<td class='verdict'>{_h.escape(verdict)}</td>"
            f"</tr>"
        )
    return (
        f"<div class='color-panel'>"
        f"<h3>{_h.escape(cap_name)}</h3>"
        f"<table class='color-table'>"
        f"<tr>{_th('ID', 'Region code.')}"
        f"{_th('Color', 'Plain-language label for the patch.')}"
        f"{_th('Ref', 'Ideal swatch synthesized from the chart-spec YUV.')}"
        f"{_th('Cap', 'Measured swatch sampled from the capture.')}"
        f"{_th('ΔY', 'Luma code delta, capture − ideal.')}"
        f"{_th('ΔU', 'Cb code delta, capture − ideal.')}"
        f"{_th('ΔV', 'Cr code delta, capture − ideal.')}"
        f"{_th('ΔE', 'Euclidean YUV10 distance.')}"
        f"{_th('Sat %', 'Saturation as a fraction of the ideal vector length in U/V.')}"
        f"{_th('Verdict', 'One-line plain-language summary.')}"
        f"</tr>"
        + "".join(rows) +
        f"</table></div>"
    )


def render_tartan_panels(captures: List[Dict[str, Any]]) -> str:
    panels = [_render_tartan_panel(c) for c in captures]
    return f"""
<section class="tartan-panels">
  <h2>Color (Tartan) — per capture</h2>
  <p class="legend">
    Each row is one of the 8 SMPTE 75% colors on the chart. The
    ref/cap swatches show the ideal next to the sampled color; ΔY/ΔU/ΔV
    are the underlying 10-bit code deltas; ΔE is a plain Euclidean
    YUV10 distance you can rank against. The Sat % column flags
    saturation loss or boost.
  </p>
  {''.join(panels)}
</section>
"""


def _gray_verdict(g: Dict[str, Any]) -> str:
    dy = g["delta_y10"]
    cast = _gray_chroma_cast(g)
    parts = []
    if abs(dy) < 5:
        parts.append("on spec")
    elif dy > 0:
        parts.append(f"{dy:+.0f} (lifted)")
    else:
        parts.append(f"{dy:+.0f} (compressed)")
    if cast >= 5:
        parts.append(f"chroma cast {cast:.0f}")
    return ", ".join(parts)


def _render_gray_panel(c: Dict[str, Any]) -> str:
    cap_name = _basename(c["_meta"]["capture"])
    grays = c.get("grays") or []
    if not grays:
        return (f"<div class='color-panel'><h3>{_h.escape(cap_name)}</h3>"
                f"<p class='muted'>no grayscale measurements</p></div>")
    by_id = {g["id"]: g for g in grays}
    rows = []
    labels = [("G1", "20% IRE (near black)"),
              ("G2", "40% IRE"),
              ("G3", "60% IRE"),
              ("G4", "80% IRE (near white)")]
    for rid, label in labels:
        g = by_id.get(rid)
        if g is None:
            continue
        ideal_rgb = tp_chart.yuv10_to_rgb8(g["ideal_y10"], 512, 512)
        meas_rgb  = tp_chart.yuv10_to_rgb8(
            g["measured_y10"], g.get("u10", 512), g.get("v10", 512))
        dy = g["delta_y10"]
        du = g.get("u10", 512) - 512
        dv = g.get("v10", 512) - 512
        cast = _gray_chroma_cast(g)
        rows.append(
            f"<tr>"
            f"<td class='name'>{rid}</td>"
            f"<td class='muted small'>{label}</td>"
            f"<td class='swatch-cell'>{_swatch_inline(ideal_rgb, 'ideal')}</td>"
            f"<td class='swatch-cell'>{_swatch_inline(meas_rgb, 'measured')}</td>"
            f"<td class='delta'>{g['ideal_y10']:.1f}</td>"
            f"<td class='delta'>{g['measured_y10']:.1f}</td>"
            f"<td class='delta {_delta_class_offset(dy, 5, 15)}'>{dy:+.1f}</td>"
            f"<td class='delta {_delta_class_offset(cast, 5, 15)}'>"
            f"ΔU {du:+.0f} ΔV {dv:+.0f}</td>"
            f"<td class='verdict'>{_h.escape(_gray_verdict(g))}</td>"
            f"</tr>"
        )
    return (
        f"<div class='color-panel'>"
        f"<h3>{_h.escape(cap_name)}</h3>"
        f"<table class='color-table'>"
        f"<tr>{_th('ID', 'Region code.')}"
        f"{_th('Step', 'Position on the chart 4-step grayscale.')}"
        f"{_th('Ref', 'Ideal swatch at the chart-spec gray level.')}"
        f"{_th('Cap', 'Measured swatch with any chroma cast preserved.')}"
        f"{_th('Y ideal', 'Chart-spec Y10 code.')}"
        f"{_th('Y meas.', 'Measured Y10 code.')}"
        f"{_th('ΔY', 'Capture − ideal, in 10-bit luma codes.')}"
        f"{_th('Chroma', 'U/V offsets from neutral (512). Non-zero = chroma cast on a notionally-gray patch.')}"
        f"{_th('Verdict', 'One-line plain-language summary.')}"
        f"</tr>"
        + "".join(rows) +
        f"</table></div>"
    )


# ---------------------------------------------------------------------
# Frequency response section (layperson view of the row-2 bursts).
# ---------------------------------------------------------------------

# Row-2 frequency-response bursts in display order. Each entry pairs the
# luma burst id (from BURST_REGIONS) with the matching cross-color
# artifact id (from ARTIFACT_REGIONS) so we can join luma modulation and
# chroma leak in one row.
FREQ_BURST_REGIONS = [
    {"id": "BURST_3p58",         "xc_id": "XC_BURST_3p58",
     "label": "3.58 MHz vertical",
     "blurb": "NTSC color subcarrier. Luma-only bars at the exact "
              "frequency the decoder's chroma filter is tuned to — a "
              "stress test for Y/C separation."},
    {"id": "BURST_300TVL_DIAG",  "xc_id": "XC_BURST_300TVL",
     "label": "300 TVL diagonal",
     "blurb": "300 TV-lines of resolution drawn as a diagonal grid. "
              "Tests how well the decoder preserves resolution off-axis "
              "(where 1-D line combs lose information)."},
    {"id": "BURST_400TVL_DIAG",  "xc_id": "XC_BURST_400TVL",
     "label": "400 TVL diagonal",
     "blurb": "400 TV-lines (~5.27 MHz horizontal). Past the analog "
              "decoder's design bandwidth — luma should still be visible, "
              "chroma leak should still be zero."},
    {"id": "BURST_4p43",         "xc_id": "XC_BURST_4p43",
     "label": "4.43 MHz vertical",
     "blurb": "PAL color subcarrier. Luma-only bars at the PAL chroma "
              "frequency. On an NTSC decoder this should pass through "
              "as pure luma; on a multistandard decoder it doubles as a "
              "second Y/C separation check."},
]


_SYNTH_FRAME_CACHE = None


def _synth_frame_bgr():
    """Render tp_synthesize.synthesize() once and cache the BGR8 frame
    used to crop reference burst thumbnails."""
    global _SYNTH_FRAME_CACHE
    if _SYNTH_FRAME_CACHE is None:
        import tp_synthesize
        Y, U, V = tp_synthesize.synthesize()
        _SYNTH_FRAME_CACHE = tp_synthesize._yuv422p10_to_bgr8(Y, U, V)
    return _SYNTH_FRAME_CACHE


def _bgr_crop_to_png_data_url(bgr, box, upscale: int = 4) -> str:
    """Crop bgr[y:y+h, x:x+w], upscale by integer factor, return a
    data:image/png;base64 URL."""
    import base64
    import cv2
    x, y, w, h = box
    crop = bgr[y:y + h, x:x + w]
    if upscale > 1:
        crop = cv2.resize(crop, (w * upscale, h * upscale),
                          interpolation=cv2.INTER_NEAREST)
    ok, png = cv2.imencode(".png", crop)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")


def _burst_box_for(region_id: str):
    for r in tp_chart.BURST_REGIONS:
        if r["id"] == region_id:
            return r["ideal_box"]
    return None


def _freq_burst_metrics(c: Dict[str, Any], burst_id: str, xc_id: str):
    """Pull luma modulation % + chroma leak rms for a single burst."""
    fr_regions = ((c.get("frequency_response") or {}).get("regions") or {})
    art_regions = ((c.get("artifacts") or {}).get("regions") or {})
    lu = fr_regions.get(burst_id) or {}
    xc = art_regions.get(xc_id) or {}
    return {
        "luma_modulation_pct": lu.get("modulation_pct"),
        "luma_modulation_db":  lu.get("modulation_db"),
        "chroma_leak_rms":     xc.get("chroma_rms"),
    }


def _freq_verdict(luma_pct, chroma_rms) -> str:
    parts = []
    if luma_pct is None:
        parts.append("luma not measured")
    elif luma_pct >= 60:
        parts.append("strong luma")
    elif luma_pct >= 30:
        parts.append("moderate luma")
    elif luma_pct >= 10:
        parts.append("weak luma")
    else:
        parts.append("no luma modulation")
    if chroma_rms is None:
        parts.append("chroma leak unknown")
    elif chroma_rms < 10:
        parts.append("clean Y/C")
    elif chroma_rms < 100:
        parts.append(f"some chroma leak ({chroma_rms:.0f} rms)")
    else:
        parts.append(f"heavy cross-color ({chroma_rms:.0f} rms)")
    return "; ".join(parts)


def _chroma_class(rms) -> str:
    # Empirically calibrated on real captures: cross-color from composite
    # NTSC decoding routinely hits 300-500 rms at 300/400 TVL, while
    # clean SDI sits below 5. Threshold midpoint at 100 separates the
    # two regimes cleanly.
    if rms is None:
        return ""
    if rms < 10:
        return "delta-good"
    if rms < 100:
        return "delta-warn"
    return "delta-bad"


def _luma_class(pct) -> str:
    if pct is None:
        return ""
    if pct >= 50:
        return "delta-good"
    if pct >= 20:
        return "delta-warn"
    return "delta-bad"


def render_frequency_response_overview(captures: List[Dict[str, Any]]) -> str:
    head = (
        "<tr>"
        + _sortable_th("Capture",
                       "Capture file name.", kind="text")
        + _sortable_th("3.58 MHz luma %",
                       "Luma modulation at the NTSC subcarrier burst. "
                       "Higher = decoder passes high-frequency luma. "
                       "Typical good values: 40-70 %.")
        + _sortable_th("Max chroma leak",
                       "Largest chroma RMS across the four row-2 "
                       "luma-only bursts. 0 = perfect Y/C separation; "
                       ">20 = noticeable cross-color contamination.")
        + _sortable_th("Worst burst",
                       "Burst region with the largest chroma leak.",
                       kind="text")
        + _sortable_th("300 TVL luma %",
                       "Luma modulation at the 300 TV-line diagonal burst.")
        + _sortable_th("400 TVL luma %",
                       "Luma modulation at the 400 TV-line diagonal burst.")
        + "</tr>"
    )
    rows = []
    for c in captures:
        name = _basename(c["_meta"]["capture"])
        per_burst = {b["id"]: _freq_burst_metrics(c, b["id"], b["xc_id"])
                     for b in FREQ_BURST_REGIONS}
        sc358 = per_burst["BURST_3p58"]["luma_modulation_pct"]
        sc300 = per_burst["BURST_300TVL_DIAG"]["luma_modulation_pct"]
        sc400 = per_burst["BURST_400TVL_DIAG"]["luma_modulation_pct"]
        leaks = [(bid, m["chroma_leak_rms"]) for bid, m in per_burst.items()
                 if m["chroma_leak_rms"] is not None]
        if leaks:
            worst_id, max_leak = max(leaks, key=lambda r: r[1])
            worst_label = next(b["label"] for b in FREQ_BURST_REGIONS
                               if b["id"] == worst_id)
        else:
            max_leak = None
            worst_label = "—"
        cells = [
            _td_name(name),
            _td_num(sc358,    "{:.1f}", cls=_luma_class(sc358)),
            _td_num(max_leak, "{:.1f}", cls=_chroma_class(max_leak)),
            f'<td class="name" data-v="{_h.escape(worst_label)}">'
            f'{_h.escape(worst_label)}</td>',
            _td_num(sc300, "{:.1f}", cls=_luma_class(sc300)),
            _td_num(sc400, "{:.1f}", cls=_luma_class(sc400)),
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<section class="overview">
  <h2>Frequency Response Overview</h2>
  <p class="legend">
    Headline numbers for the row-2 frequency bursts. Higher luma % = the
    decoder passes high-frequency luma; lower chroma leak = better Y/C
    separation (a luma-only burst should produce zero chroma).
  </p>
  <table class="overview-table">
    <thead>{head}</thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def _render_freq_burst_panel(c: Dict[str, Any]) -> str:
    cap_name = _basename(c["_meta"]["capture"])
    rows_html = []
    for b in FREQ_BURST_REGIONS:
        m = _freq_burst_metrics(c, b["id"], b["xc_id"])
        lu_pct = m["luma_modulation_pct"]
        lu_db  = m["luma_modulation_db"]
        xc_rms = m["chroma_leak_rms"]
        verdict = _freq_verdict(lu_pct, xc_rms)
        lu_cell = ("—" if lu_pct is None
                   else f"{lu_pct:.1f}%"
                        + (f" ({lu_db:+.1f} dB)" if lu_db is not None
                           and lu_db != float("-inf") else ""))
        rows_html.append(
            f"<tr>"
            f"<td class='name'>{_h.escape(b['label'])}</td>"
            f"<td class='delta {_luma_class(lu_pct)}'>{lu_cell}</td>"
            f"<td class='delta {_chroma_class(xc_rms)}'>"
            f"{'—' if xc_rms is None else f'{xc_rms:.1f}'}</td>"
            f"<td class='verdict'>{_h.escape(verdict)}</td>"
            f"</tr>"
        )
    return (
        f"<div class='color-panel'>"
        f"<h3>{_h.escape(cap_name)}</h3>"
        f"<table class='color-table'>"
        f"<tr>{_th('Burst', 'Frequency-response test region in row 2 of the chart.')}"
        f"{_th('Luma modulation', 'Peak luma modulation as a fraction of the chart full-contrast swing (black→white = 100%).')}"
        f"{_th('Chroma leak (rms)', 'RMS chroma deviation inside this luma-only burst. Ideally 0 — any nonzero value means the decoder is misclassifying high-frequency luma as chroma (cross-color).')}"
        f"{_th('Verdict', 'Plain-language summary.')}"
        f"</tr>"
        + "".join(rows_html) +
        f"</table></div>"
    )


def render_frequency_response_section(captures: List[Dict[str, Any]]) -> str:
    if not any(c.get("frequency_response") or c.get("artifacts")
               for c in captures):
        return ""
    # Synthetic reference thumbnails — what a clean decoder should produce.
    try:
        bgr = _synth_frame_bgr()
        ref_imgs = []
        for b in FREQ_BURST_REGIONS:
            box = _burst_box_for(b["id"])
            if box is None:
                continue
            data_url = _bgr_crop_to_png_data_url(bgr, box, upscale=4)
            ref_imgs.append(
                f"<figure class='burst-ref'>"
                f"<img src='{data_url}' alt='{_h.escape(b['label'])}'/>"
                f"<figcaption>{_h.escape(b['label'])}</figcaption>"
                f"</figure>"
            )
        ref_block = ("<div class='burst-ref-row'>" + "".join(ref_imgs)
                     + "</div>")
    except Exception:
        ref_block = ""

    intro = """
<p class="legend">
  <b>What is a TVL burst?</b> "TVL" = TV-lines, the classic broadcast
  metric for horizontal resolution. 300 TVL means the chart can resolve
  300 distinct vertical bars across the picture height (≈ 240 bars
  across the active width on a 4:3 raster); 400 TVL is a denser pattern
  past the design bandwidth of most analog composite decoders. The
  diagonal version of each burst tilts the pattern 45° to test
  off-axis resolution, where simpler line-comb decoders typically
  lose information.
</p>
<p class="legend">
  <b>What should we see?</b> Every row-2 burst is drawn as pure
  <i>luma</i> (black-and-white stripes). On a clean signal path you
  should see strong luma modulation and <b>zero chroma</b>. Any chroma
  rms above noise floor means the decoder is mistaking high-frequency
  luma for chroma — the textbook
  <a href="https://en.wikipedia.org/wiki/Chrominance"
     style="color:#9ec1ff">cross-color (Y/C separation) artifact</a>.
</p>
<p class="legend">
  3.58 MHz and 4.43 MHz are particularly diagnostic because they sit
  exactly on the NTSC and PAL color subcarriers, where most decoder
  notch filters reject them cleanly. The 300 TVL (≈ 3.95 MHz) and
  400 TVL (≈ 5.27 MHz) diagonals sit <i>outside</i> the notch — naive
  decoders happily pass them through and re-interpret them as chroma,
  producing dramatic cross-color (hundreds of rms units in real
  composite-NTSC processing chains). High chroma rms on these bursts
  isn't a measurement bug — it's the actual artifact the chart was
  designed to expose.
</p>
"""
    panels = [_render_freq_burst_panel(c) for c in captures]
    return f"""
<section class="freq-burst-panels">
  <h2>Frequency Response — row-2 bursts</h2>
  {intro}
  <h4 style="margin-bottom:4px">Reference (synthesized clean signal)</h4>
  {ref_block}
  {''.join(panels)}
</section>
"""


def render_gray_panels(captures: List[Dict[str, Any]]) -> str:
    panels = [_render_gray_panel(c) for c in captures]
    return f"""
<section class="gray-panels">
  <h2>Grayscale — per capture</h2>
  <p class="legend">
    The 4 grayscale steps at 20/40/60/80 % IRE. Y is measured against
    the chart-spec ideal; the chroma column flags any tint on what
    should be a neutral patch (comb-decoder smell).
  </p>
  {''.join(panels)}
</section>
"""


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
        {_th("Capture", "Capture file name (without directory).")}
        {_th("Raster", "Detected video raster (width x height) before pad-to-486.")}
        {_th("Field", "Field order from ffprobe (top, bottom, or progressive).")}
        {_th("Quality", "Registration quality flag from Stage 2: ok / warn / failed. Based on how many landmarks fit the affine and how tight the residuals are.")}
        {_th("Mean&nbsp;px", "Mean reprojection residual across RANSAC inliers, in pixels. Lower is better. Anything under 1 px is excellent; 1-2 px is warn; >2 is failed.")}
        {_th("Max&nbsp;px", "Maximum reprojection residual across inliers, in pixels.")}
        {_th("Inliers", "Number of grid landmarks plus Stage-2 anchors that the RANSAC affine fit accepted, vs. the total candidates available.")}
        {_th("Affine", "Estimated affine mapping ideal chart coords -> capture coords. tx,ty=translation, sx,sy=scale, shear=off-diagonal shear.")}
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

    head = (
        "<tr>"
        + _th("Capture", "Capture file name. The cells show measured YUV deltas "
                         "vs the 75% chart-spec ideal for each tartan patch.")
        + "".join(_region_th(rid) for rid in region_ids)
        + "</tr>"
    )

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
    head = (
        "<tr>"
        + _th("Capture", "Capture file name. Cells show measured Y10 vs "
                         "ideal Y10 for each 20/40/60/80% IRE gray step.")
        + "".join(_region_th(rid) for rid in region_ids)
        + "</tr>"
    )

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
th.tip { position: relative; cursor: help; border-bottom: 1px dotted #6a8aa0; }
th.tip::after {
  content: attr(data-tip);
  position: absolute;
  top: calc(100% + 6px);
  left: 50%;
  transform: translateX(-50%);
  background: #14171c;
  color: #d8dde6;
  border: 1px solid #4a5060;
  border-radius: 4px;
  padding: 8px 10px;
  font-size: 12px;
  font-weight: normal;
  font-family: system-ui, sans-serif;
  text-align: left;
  white-space: normal;
  width: max-content;
  max-width: 320px;
  line-height: 1.45;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
  visibility: hidden;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.12s ease;
  z-index: 100;
}
th.tip::before {
  content: "";
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border: 6px solid transparent;
  border-bottom-color: #4a5060;
  visibility: hidden;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.12s ease;
  z-index: 101;
}
th.tip:hover::after,
th.tip:focus::after,
th.tip:hover::before,
th.tip:focus::before { visibility: visible; opacity: 1; }
img.fiducial-crops { max-width: 100%; height: auto; margin-top: 6px;
  background: #14161a; border: 1px solid #2a2e36; image-rendering: pixelated; }
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
.geo-table { border-collapse: collapse; margin: 4px 0; }
.geo-table th, .geo-table td { border: 1px solid #2a2e36; padding: 3px 8px; font-size: 12px; }
.geo-table td:nth-child(2), .geo-table td:nth-child(4) { text-align: right; font-variant-numeric: tabular-nums; }
.geo-list { margin: 4px 0 4px 18px; padding: 0; font-size: 12px; line-height: 1.65; color: #c5d1e0; }
.geo-list li { margin: 2px 0; }
.clip-chips { display: flex; gap: 6px; flex-wrap: wrap; margin: 4px 0; }
.clip-chip { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 12px;
    font-family: monospace; border: 1px solid #2a2e36; }
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
.overview { margin: 18px 0; }
table.overview-table { border-collapse: collapse; margin: 8px 0; }
table.overview-table th, table.overview-table td { border: 1px solid #2a2e36; padding: 4px 10px;
    font-size: 12px; }
table.overview-table th { background: #21252b; color: #fff; cursor: pointer; user-select: none;
    white-space: nowrap; }
table.overview-table th.sortable::after { content: " \\2195"; opacity: 0.4; font-size: 10px; }
table.overview-table th.sort-asc::after  { content: " \\25B2"; opacity: 1; }
table.overview-table th.sort-desc::after { content: " \\25BC"; opacity: 1; }
table.overview-table td { font-variant-numeric: tabular-nums; }
table.overview-table td.numeric { text-align: right; }
table.overview-table td.name { text-align: left; max-width: 380px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
table.overview-table tbody tr:nth-child(odd) { background: rgba(255,255,255,0.015); }
.appendix { margin-top: 32px; padding: 12px 16px; background: #16181d;
    border: 1px solid #2a2e36; border-radius: 4px; }
.appendix > h2 { color: #8a929f; font-size: 14px; border-bottom: none; margin: 0 0 10px 0; }
.appendix section h2 { color: #c5d1e0; font-size: 13px; border-bottom: 1px solid #2a2e36; }
.appendix section { margin: 12px 0; }
.color-panel { margin: 12px 0; padding: 8px 12px; background: #1d2026; border: 1px solid #2a2e36; }
.color-panel h3 { margin: 4px 0 8px 0; font-size: 14px; }
.color-panel h4 { margin: 8px 0 4px 0; font-size: 12px; color: #c5d1e0; }
.color-table { border-collapse: collapse; margin: 4px 0; width: 100%; max-width: 920px; }
.color-table th, .color-table td { border: 1px solid #2a2e36; padding: 4px 8px;
    font-size: 12px; vertical-align: middle; }
.color-table th { background: #21252b; color: #fff; white-space: nowrap; }
.color-table td.name      { font-family: monospace; }
.color-table td.swatch-cell { width: 32px; text-align: center; }
.color-table td.delta     { font-variant-numeric: tabular-nums; text-align: right; }
.color-table td.verdict   { font-size: 11px; }
.swatch-inline { display: inline-block; width: 22px; height: 22px; vertical-align: middle; }
.swatch-inline.ideal    { border: 2px dashed #c5d1e0; box-sizing: border-box; }
.swatch-inline.measured { border: 2px solid  #f0b450; box-sizing: border-box; }
.burst-ref-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 6px 0 12px 0; }
.burst-ref { margin: 0; text-align: center; }
.burst-ref img { display: block; border: 1px solid #2a2e36; image-rendering: pixelated;
    background: #14161a; }
.burst-ref figcaption { font-size: 11px; color: #b8c0cc; margin-top: 4px; }
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
        {_th("Capture", "Capture file name.")}
        {_th("Slope", "Slope from a least-squares fit of measured Y10 vs ideal Y10 on the 4 gray steps. Slope=1.0 means unity luma gain.")}
        {_th("Intercept", "Intercept of the same linear fit (Y10 codes). 0 means no DC offset; positive raises black level.")}
        {_th("Predicted at black (Y10=64)", "What the linear fit predicts the measured Y10 would be at the chart's nominal black level (64).")}
        {_th("Predicted at white (Y10=940)", "What the linear fit predicts at chart white (940).")}
        {_th("Linear-fit RMS", "Root-mean-square residual of the 4 gray points against the linear fit. Lower means the gray ramp is more linear.")}
        {_th("Pedestal&nbsp;A / B RMS", "RMS residual against pedestal-mismatch hypotheses A and B (alternative explanations to a pure gain change).")}
        {_th("Verdict", "Heuristic interpretation: luma gain off, pedestal mismatch, or clean. Based on slope and the residuals above.")}
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


def _spacing_row(label: str, entry: Dict[str, Any], tip: str) -> str:
    actual = entry["actual"]
    ideal = entry["ideal"]
    delta = entry["delta"]
    cls = _delta_class_offset(delta, 2, 5)
    sign = f"{delta:+.1f}"
    return (
        f"<tr>"
        f"<td class='tip' data-tip='{_h.escape(tip)}'>{label}</td>"
        f"<td>{actual:.1f} px</td>"
        f"<td class='muted'>ideal {ideal:.0f}</td>"
        f"<td class='{cls}'>{sign}</td>"
        f"</tr>"
    )


def _dir_word_h(dx: float) -> str:
    if abs(dx) < 0.05:
        return "centered horizontally"
    return f"{abs(dx):.1f} px {'right' if dx > 0 else 'left'}"


def _dir_word_v(dy: float) -> str:
    # y axis points down; negative dy = picture sits higher than ideal center.
    if abs(dy) < 0.05:
        return "centered vertically"
    return f"{abs(dy):.1f} px {'down' if dy > 0 else 'up'}"


def _render_geometry_panel(c: Dict[str, Any]) -> str:
    cap_name = _basename(c["_meta"]["capture"])
    g = c.get("geometry")
    if g is None:
        return (
            f"<div class='geometry-panel'><h3>{_h.escape(cap_name)}</h3>"
            f"<p class='muted'>no geometry block (older JSON)</p></div>"
        )
    d = g["derived"]
    summary = d.get("summary") or {}
    flag = g.get("quality_flag", "?")
    flag_class = {"ok": "ok", "warn": "warn", "partial": "warn",
                  "failed": "bad"}.get(flag, "")

    # --- Arrowhead spacing (vs chart spec) ---
    sp = summary.get("arrow_spacings_px")
    if sp:
        rows = (
            _spacing_row("Top edge",    sp["top"],
                         "TL→TR apex horizontal spacing.")
            + _spacing_row("Bottom edge", sp["bottom"],
                           "BL→BR apex horizontal spacing.")
            + _spacing_row("Left edge",   sp["left"],
                           "TL→BL apex vertical spacing.")
            + _spacing_row("Right edge",  sp["right"],
                           "TR→BR apex vertical spacing.")
        )
        spacing_html = (
            "<table class='geo-table'>"
            f"<tr>{_th('Edge', 'Picture edge defined by the two arrowhead apexes along it.')}"
            f"{_th('Measured', 'Distance between detected arrowhead apexes, in capture pixels.')}"
            f"{_th('Chart spec', 'Distance between the two apexes in the canonical 720×486 chart.')}"
            f"{_th('Δ vs ideal', 'Measured − ideal. Positive = picture wider/taller than spec; negative = narrower/shorter.')}</tr>"
            f"{rows}</table>"
        )
    else:
        spacing_html = ("<p class='muted'>arrow spacings not derivable "
                        "(one or more triangle apexes were not detected)</p>")

    # --- Picture displacement (center offset + scale + keystone) ---
    off = summary.get("picture_center_offset_px")
    sc  = summary.get("picture_scale_pct")
    ks  = summary.get("keystone_px")
    if off and sc and ks:
        dx, dy = off["dx"], off["dy"]
        h_dev = sc["horizontal"] - 100.0
        v_dev = sc["vertical"]   - 100.0
        ks_h, ks_v = ks["horizontal_top_minus_bottom"], ks["vertical_left_minus_right"]

        if abs(ks_h) < 0.05:
            ks_h_text = "top and bottom widths match"
        else:
            ks_h_text = (
                f"top is <span class='{_delta_class_offset(ks_h, 2, 5)}'>"
                f"{abs(ks_h):.1f} px {'wider' if ks_h > 0 else 'narrower'}</span> than bottom"
            )
        if abs(ks_v) < 0.05:
            ks_v_text = "left and right heights match"
        else:
            ks_v_text = (
                f"left is <span class='{_delta_class_offset(ks_v, 2, 5)}'>"
                f"{abs(ks_v):.1f} px {'taller' if ks_v > 0 else 'shorter'}</span> than right"
            )

        disp_html = (
            "<ul class='geo-list'>"
            f"<li>Center shifted "
            f"<span class='{_delta_class_offset(dx, 2, 5)}'>{_dir_word_h(dx)}</span>, "
            f"<span class='{_delta_class_offset(dy, 2, 5)}'>{_dir_word_v(dy)}</span>.</li>"
            f"<li>Horizontal scale "
            f"<span class='{_delta_class_offset(h_dev, 1, 3)}'>{sc['horizontal']:.1f}%</span>, "
            f"vertical scale "
            f"<span class='{_delta_class_offset(v_dev, 1, 3)}'>{sc['vertical']:.1f}%</span>.</li>"
            f"<li>Keystone: {ks_h_text}; {ks_v_text}.</li>"
            "</ul>"
        )
    else:
        disp_html = "<p class='muted'>displacement not derivable</p>"

    # --- Clip detection ---
    clip = d.get("clip_detected") or {}
    clip_chips = []
    for tid in ("TL", "TR", "BL", "BR"):
        entry = clip.get(tid, {})
        visible = entry.get("apex_visible", False)
        interp = entry.get("interpretation", "?")
        cls = "delta-good" if visible else "delta-bad"
        label = "visible" if visible else "clipped"
        clip_chips.append(
            f"<span class='clip-chip {cls}' title='{_h.escape(interp)}'>"
            f"{tid}: {label}</span>"
        )
    clip_html = "<div class='clip-chips'>" + " ".join(clip_chips) + "</div>"

    # --- Registration cross (kept) ---
    cross_off = d.get("cross_offset_from_ideal") or [None, None]
    aperture = d.get("aperture_symmetry")
    if cross_off[0] is not None and aperture is not None:
        cross_html = (
            f"<div class='small'>"
            f"Center offset dx={cross_off[0]:+.2f} px, dy={cross_off[1]:+.2f} px"
            f" &nbsp;·&nbsp; aperture symmetry "
            f"<span class='{_delta_class_offset((1.0-aperture)*100, 5, 15)}'>"
            f"{aperture:.3f}</span> "
            f"<span class='muted'>(1.00 = horizontal and vertical apertures match)</span>"
            f"</div>"
        )
    else:
        cross_html = "<p class='muted'>registration cross missing</p>"

    # --- Circle (PAR-aware) ---
    circ = summary.get("circle") or {}
    hd = circ.get("horizontal_diameter_px")
    vd = circ.get("vertical_diameter_px")
    ahv = circ.get("actual_h_over_v")
    ehv = circ.get("expected_h_over_v_for_round")
    dc  = circ.get("displayed_circularity")
    rot = circ.get("rotation_deg")
    if hd is not None and vd is not None and ahv is not None and dc is not None:
        dc_dev = dc - 1.0
        if dc > 1.005:
            shape_word = "horizontally stretched"
        elif dc < 0.995:
            shape_word = "vertically stretched"
        else:
            shape_word = "round"
        rot_text = ""
        if rot is not None:
            # cv2.fitEllipse returns angle in [0, 180); near 0 or 180 means
            # the major axis is the vertical one, near 90 means major axis
            # is horizontal. Normalize to a small tilt-from-axis-aligned.
            tilt = min(rot % 90.0, 90.0 - (rot % 90.0))
            if tilt > 1.0:
                rot_text = f" &nbsp;·&nbsp; tilt {tilt:.1f}°"
        circle_html = (
            "<ul class='geo-list'>"
            f"<li>Horizontal diameter <b>{hd:.1f} px</b>, "
            f"vertical diameter <b>{vd:.1f} px</b>.</li>"
            f"<li>H/V ratio {ahv:.3f}; "
            f"expected {ehv:.3f} for a round shape (NTSC 10:11 PAR).</li>"
            f"<li>Displayed circularity "
            f"<span class='{_delta_class_offset(dc_dev*100, 2, 5)}'>{dc:.3f}</span> "
            f"— <b>{shape_word}</b> "
            f"<span class='muted'>(1.00 = round in display; "
            f"&gt;1 horizontally stretched, &lt;1 vertically stretched)</span>"
            f"{rot_text}.</li>"
            "</ul>"
        )
    else:
        circle_html = "<p class='muted'>circle not derivable (detector returned no fit)</p>"

    # --- Refit benefit (diagnostic, kept) ---
    refit = g.get("registration_refit", {})
    mean_res = refit.get("final_residuals_px", {}).get("mean", float("nan"))
    refit_html = (
        f"<div class='small muted'>"
        f"Registration refit: inliers {refit.get('inlier_count_initial', '?')} "
        f"&rarr; <b>{refit.get('inlier_count_final', '?')}</b>, "
        f"mean residual {mean_res:.2f} px"
        f"</div>"
    )

    fids_html = _render_fiducial_crops(c)

    return (
        f"<div class='geometry-panel'>"
        f"<h3>{_h.escape(cap_name)} "
        f"<span class='{flag_class}'>[{flag}]</span></h3>"
        f"<h4>Arrowhead spacing (vs chart spec)</h4>{spacing_html}"
        f"<h4>Picture displacement</h4>{disp_html}"
        f"<h4>Clip detection</h4>{clip_html}"
        f"<h4>Registration cross</h4>{cross_html}"
        f"<h4>Circle (PAR-aware, NTSC 10:11)</h4>{circle_html}"
        f"{refit_html}"
        f"{fids_html}"
        f"</div>"
    )


def render_geometry_section(captures: List[Dict[str, Any]]) -> str:
    panels = [_render_geometry_panel(c) for c in captures]
    return f"""
<section class="geometry">
  <h2>Geometry</h2>
  <p class="legend">
    Picture-in-raster geometry derived from the SW2 chart's four arrowhead
    fiducials, registration cross, and black ring. Spacings and offsets are
    compared to the canonical 720×486 chart so any horizontal/vertical
    displacement, scale error, or keystone shows up here. The ring check is
    PAR-aware (NTSC has 10:11 non-square pixels): a perfectly-round display
    reads displayed circularity = 1.00 even though the ring is elliptical
    in raster pixels.
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
    rows.append(
        '<table class="freq-table"><thead><tr>'
        + _th("Clip", "Capture file name.")
        + _th("−3 dB (MHz)",
              "Highest burst frequency where the luma modulation is still within -3 dB "
              "of the chart-spec full contrast. Higher = wider luma bandwidth.")
        + _th("−6 dB (MHz)",
              "Highest burst frequency where luma modulation is within -6 dB. "
              "Stricter cut-off than -3 dB; a useful proxy for the practical luma "
              "resolution limit.")
        + '</tr></thead><tbody>'
    )
    for c in captures:
        fr = c.get("frequency_response") or {}
        s = (fr.get("summary") or {}) if isinstance(fr, dict) else {}
        m3 = s.get("minus_3db_freq_MHz")
        m6 = s.get("minus_6db_freq_MHz")
        name = _basename(c["_meta"].get("capture", c["_meta"].get("capture_path", "?")))
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
        name = _basename(c["_meta"].get("capture", c["_meta"].get("capture_path", "?")))
        datasets.append({
            "label": name,
            "data": [{"x": float(f), "y": float(d)} for f, d in curve],
        })
    # Y/C timing burst table: per-clip chroma + dot-crawl per frequency.
    yc_freqs = ("YC_BURST_0p5MHZ", "YC_BURST_1p0MHZ", "YC_BURST_1p5MHZ")
    has_yc = any(
        ((c.get("frequency_response") or {}).get("regions") or {}).get(rid)
        for c in captures for rid in yc_freqs
    )
    if has_yc:
        rows.append("<h3>Y/C timing bursts (row 9 chroma)</h3>")
        rows.append(
            '<table class="freq-table"><thead><tr>'
            + _th("Clip", "Capture file name.")
            + _th("0.5 MHz chroma",
                  "Peak chroma modulation at 0.5 MHz in cells (9,6-7) "
                  "(blue/yellow stripes). Higher = wider chroma bandwidth.")
            + _th("0.5 MHz dot crawl",
                  "Luma modulation at 0.5 MHz in the same region. "
                  "On the chart the colors also alternate in luma, so "
                  "this is high even on a clean decoder; compare to "
                  "the chroma value -- a clean decoder reproduces both "
                  "in lockstep. Excess luma vs chroma = cross-luma.")
            + _th("1.0 MHz chroma", "Peak chroma modulation at 1.0 MHz in cell (9,5).")
            + _th("1.0 MHz dot crawl", "Luma modulation at 1.0 MHz in cell (9,5).")
            + _th("1.5 MHz chroma", "Peak chroma modulation at 1.5 MHz in cell (9,8).")
            + _th("1.5 MHz dot crawl", "Luma modulation at 1.5 MHz in cell (9,8).")
            + "</tr></thead><tbody>"
        )
        for c in captures:
            regs = ((c.get("frequency_response") or {}).get("regions") or {})
            name = _basename(c["_meta"].get("capture", "?"))
            cells = [f"<td>{_h.escape(name)}</td>"]
            for rid in yc_freqs:
                r = regs.get(rid, {})
                ch = r.get("chroma_modulation_pct", 0.0)
                lu = r.get("luma_dot_crawl_pct", 0.0)
                cells.append(f"<td>{ch:.1f}%</td>")
                cells.append(f"<td>{lu:.1f}%</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        rows.append("</tbody></table>")

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
    rows.append(
        '<table class="artifact-table"><thead><tr>'
        + _th("Clip", "Capture file name.")
        + _th("Hanging dots (Y pp)",
              "Peak-to-peak luma modulation in the grey strip just above a chroma "
              "block (row-8 cells adjacent to red/magenta). High values indicate "
              "cross-luma at the subcarrier — a line-comb decoder signature.")
        + _th("Dot crawl (chroma RMS)",
              "Chroma RMS in the grey region just below the tartan bars. "
              "Spurious chroma here means the decoder spreads cross-color away "
              "from sharp chroma edges.")
        + _th("Cross-color (chroma RMS)",
              "Chroma RMS inside luma-only burst regions (B&W stripes that "
              "should have zero chroma). Notch/simple decoders mistake high-freq "
              "luma for chroma here.")
        + _th("Cross-luma (Y pp)",
              "Luma modulation at subcarrier (3-4 MHz bandpass) inside flat "
              "chroma blocks. Indicates the decoder is mistaking subcarrier "
              "for luma detail.")
        + _th("Zone-plate chroma RMS",
              "Chroma RMS over the moving zone-plate container (cells 3,4-6,9). "
              "The zone plate is luma-only on the chart; any chroma here is "
              "decoder-introduced cross-color.")
        + _th("ZP chroma present?",
              "Binary flag: zone-plate chroma RMS exceeds T_zp_threshold (=15 codes). "
              "YES is a strong indicator of a notch/simple decoder.")
        + '</tr></thead><tbody>'
    )
    for c in captures:
        a = c.get("artifacts") or {}
        s = (a.get("summary") or {}) if isinstance(a, dict) else {}
        zp_present = s.get("zone_plate_chroma_present", False)
        name = _basename(c["_meta"].get("capture", c["_meta"].get("capture_path", "?")))
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


def render_radial_wedge(captures: List[Dict[str, Any]]) -> str:
    """Compact table for the radial wedge: cross-color (chroma RMS, since
    the wedge is luma-only on the chart) and H/V symmetry of luma
    modulation through the wedge centre."""
    def _ok(c):
        a = c.get("artifacts") or {}
        regs = a.get("regions") if isinstance(a, dict) else None
        if not regs:
            return False
        return ("WEDGE_HV_SYMMETRY" in regs) or ("XC_RADIAL_WEDGE" in regs)
    if not any(_ok(c) for c in captures):
        return ""
    rows = ['<section class="radial-wedge">'
            '<h2>Radial wedge (Stage 3)</h2>'
            '<p class="legend">Cell (8,11). The wedge is black/white only, '
            'so chroma here is decoder cross-color. H/V std compares luma '
            'modulation along horizontal vs vertical cross-sections '
            'through the wedge centre -- the ratio reveals decoder '
            'aperture-correction asymmetry (1.0 = balanced).</p>']
    rows.append(
        '<table class="data wedge-table"><thead><tr>'
        + _th("Clip", "Capture file name.")
        + _th("Cross-color chroma RMS",
              "Chroma RMS over the wedge box. The wedge has no native "
              "chroma; any value here is decoder-injected cross-color.")
        + _th("H modulation (std)",
              "Standard deviation of the luma cross-section through the "
              "wedge centre along the horizontal axis. Higher = more "
              "high-spatial-frequency luma along H.")
        + _th("V modulation (std)",
              "Same metric along the vertical axis through wedge centre.")
        + _th("H/V ratio",
              "h_std / v_std. 1.0 = symmetric aperture. >1 = decoder "
              "sharpens horizontally more than vertically; <1 = vice versa.")
        + "</tr></thead><tbody>"
    )
    def _cell(v, fmt="{:.2f}"):
        return f"<td>{fmt.format(v)}</td>" if v is not None else "<td>—</td>"
    for c in captures:
        a = c.get("artifacts") or {}
        regs = (a.get("regions") or {}) if isinstance(a, dict) else {}
        xc = regs.get("XC_RADIAL_WEDGE", {}).get("chroma_rms")
        wsym = regs.get("WEDGE_HV_SYMMETRY", {})
        h_std = wsym.get("h_modulation_std")
        v_std = wsym.get("v_modulation_std")
        hv = wsym.get("hv_ratio")
        name = _basename(c["_meta"].get("capture", "?"))
        rows.append(
            "<tr>"
            f"<td>{_h.escape(name)}</td>"
            f"{_cell(xc)}{_cell(h_std)}{_cell(v_std)}{_cell(hv)}"
            "</tr>"
        )
    rows.append("</tbody></table></section>")
    return "\n".join(rows)


def render_decoder_class(captures: List[Dict[str, Any]]) -> str:
    if not any(c.get("decoder_class") for c in captures):
        return ""
    rows = ['<section class="decoder-class"><h2>Decoder class (Stage 3)</h2>']
    rows.append(
        '<table class="decoder-table"><thead><tr>'
        + _th("Clip", "Capture file name.")
        + _th("Class",
              "Inferred decoder class: notch (simple low-pass), line_comb, "
              "temporal_comb_or_adaptive (cleanest — cannot distinguish "
              "temporal vs adaptive without zone-plate motion analysis), "
              "or undetermined (no rule above 0.5 confidence).")
        + _th("Confidence",
              "Geometric mean of per-metric confidences for the winning rule. "
              "Range 0..1. Below 0.5 yields 'undetermined'.")
        + _th("Candidate confidences",
              "Confidence for each of the three rules. Compare to see how "
              "close the runner-up was.")
        + '</tr></thead><tbody>'
    )
    for c in captures:
        dc = c.get("decoder_class") or {}
        cls = dc.get("decoder_class", "—")
        conf = float(dc.get("confidence", 0.0))
        ev = (dc.get("evidence") or {})
        cands = ev.get("candidate_confidences", {}) or {}
        cands_str = ", ".join(f"{k}: {v:.2f}" for k, v in cands.items())
        name = _basename(c["_meta"].get("capture", c["_meta"].get("capture_path", "?")))
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
    overviews = (
        render_overall_summary(captures)
        + render_geometry_overview(captures)
        + render_tartan_overview(captures)
        + render_grayscale_overview(captures)
        + render_frequency_response_overview(captures)
    )
    details = (
        render_geometry_section(captures)
        + render_tartan_panels(captures)
        + render_gray_panels(captures)
        + render_frequency_response_section(captures)
        + render_luma_scale_analysis(captures)
        + render_artifacts(captures)
        + render_radial_wedge(captures)
        + render_decoder_class(captures)
        + render_sample_diagnostics(captures)
    )
    appendix = (
        '<section class="appendix">'
        '<h2>Technical Appendix</h2>'
        + render_registration_summary(captures)
        + render_tartan_deltas(captures)
        + render_gray_deltas(captures)
        + render_frequency_response(captures)
        + '</section>'
    )
    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{_h.escape(title)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>{_CSS}</style>
</head><body>
<h1>{_h.escape(title)}</h1>
{overviews}
{details}
{appendix}
{_OVERVIEW_SORT_JS}
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
