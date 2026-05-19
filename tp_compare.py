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
    dimension. Column order matches the order of the detail sections
    below."""
    head = (
        "<tr>"
        + _sortable_th("Capture",
                       "Capture file name.", kind="text")
        + _sortable_th("Geometry",
                       "Picture-in-raster correctness: arrowhead "
                       "spacing, scale, keystone, PAR-aware circularity. "
                       "Higher = the raster matches the chart.")
        + _sortable_th("Color",
                       "Tartan color match: mean YUV10 distance from "
                       "chart spec across the 8 SMPTE 75% patches. "
                       "Higher = more faithful color reproduction.")
        + _sortable_th("Chroma Lin",
                       "Chroma non-linearity: magenta saturation "
                       "staircase R² and differential phase. Higher = "
                       "saturated colors stay on-hue.")
        + _sortable_th("Grayscale",
                       "Gray-step accuracy: mean |ΔY| at 20/40/60/80% "
                       "IRE plus a chroma-cast penalty. Higher = neutral "
                       "tonal scale.")
        + _sortable_th("Pulse",
                       "2T pulse transient response: FWHM ≈ 200 ns, "
                       "ringing / echo, black-clipper presence. Higher "
                       "= clean sharp transitions.")
        + _sortable_th("Frequency",
                       "Row-2 burst luma + Y/C-separation behaviour at "
                       "3.58/4.43 MHz and 300/400 TVL diagonals. "
                       "Higher = wider luma bandwidth and cleaner Y/C.")
        + _sortable_th("Wedge",
                       "Radial wedge resolution and decoder cross-"
                       "effects (≤ 450 TVL). Higher = sharper image "
                       "with no cross-color or H/V aperture asymmetry.")
        + _sortable_th("VertRes",
                       "Vertical-axis modulation at 100/200/300 TVL. "
                       "Higher = the decoder preserves vertical detail.")
        + _sortable_th("Y/C",
                       "Row-9 chroma bursts measured across 3 frames "
                       "for dot-crawl wiggle + chroma bandwidth. "
                       "Higher = stable, full-bandwidth chroma.")
        + _sortable_th("ZonePlate",
                       "Cross-color injected into the moving zone-plate "
                       "(luma-only) area, averaged over 3 frames. "
                       "Higher = no false color on detailed moving "
                       "content.")
        + _sortable_th("Overall",
                       "Mean of the available category scores.")
        + "</tr>"
    )
    rows = []
    for c in captures:
        name = _basename(c["_meta"]["capture"])
        g  = _score_geometry(c)
        co = _score_color(c)
        cl = _score_chroma_staircase(c)
        gs = _score_grayscale(c)
        pl = _score_pulse(c)
        fq = _score_frequency(c)
        rw = _score_radial_wedge(c)
        vr = _score_vertical_response(c)
        yc = _score_yc_timing(c)
        zp = _score_zone_plate(c)
        vals = [v for v in (g, co, cl, gs, pl, fq, rw, vr, yc, zp)
                if v is not None and not (isinstance(v, float) and v != v)]
        overall = sum(vals) / len(vals) if vals else float("nan")
        cells = [
            _td_name(name),
            _td_num(g,       "{:.1f}", cls=_score_class(g)),
            _td_num(co,      "{:.1f}", cls=_score_class(co)),
            _td_num(cl,      "{:.1f}", cls=_score_class(cl)),
            _td_num(gs,      "{:.1f}", cls=_score_class(gs)),
            _td_num(pl,      "{:.1f}", cls=_score_class(pl)),
            _td_num(fq,      "{:.1f}", cls=_score_class(fq)),
            _td_num(rw,      "{:.1f}", cls=_score_class(rw)),
            _td_num(vr,      "{:.1f}", cls=_score_class(vr)),
            _td_num(yc,      "{:.1f}", cls=_score_class(yc)),
            _td_num(zp,      "{:.1f}", cls=_score_class(zp)),
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
  <details class="legend-key">
    <summary>What each column measures</summary>
    <dl class="metric-key">
      <dt>Geometry</dt>
      <dd>How faithfully the picture is positioned, scaled, and shaped
        inside the raster (arrowhead spacings, scale, keystone,
        PAR-aware circularity). Drives whether straight edges look
        straight and circles look round — the foundation of "the
        picture looks right".</dd>
      <dt>Color</dt>
      <dd>Tartan-patch reproduction. Tells you whether saturated SMPTE
        colors come back at the right hue and saturation. Off-color
        readings look unnatural on skin tones and brand colors.</dd>
      <dt>Chroma Lin</dt>
      <dd>Whether color reproduces linearly with saturation (33 %, 66 %,
        100 % magenta stay on the same hue and scale 1:2:3 in chroma
        magnitude). Failures look like mid-saturation colors
        drifting in hue, or pastels and saturated tones being
        compressed against each other.</dd>
      <dt>Grayscale</dt>
      <dd>Tonal accuracy at 20/40/60/80 % IRE plus chroma neutrality.
        Failures appear as wrong overall brightness, crushed shadows
        / blown highlights, or a tint on what should be neutral
        grays.</dd>
      <dt>Pulse</dt>
      <dd>2T pulse-and-bar transient response: how sharply the
        decoder reproduces a 200 ns edge, how much it rings, and
        whether a black clipper hides below-pedestal undershoots.
        Drives perceived <i>crispness</i> on text and hard edges.</dd>
      <dt>Frequency</dt>
      <dd>High-frequency luma response + Y/C separation at the row-2
        bursts (3.58 MHz NTSC subcarrier, 4.43 MHz PAL, 300 / 400
        TVL diagonals). Drives both fine-detail clarity AND the
        absence of false color on busy luma.</dd>
      <dt>Wedge</dt>
      <dd>Radial-wedge resolution limit (up to ~450 TVL) plus decoder
        cross-effects: cross-color injection and H/V aperture
        balance. Drives image sharpness and whether the decoder
        tints high-frequency detail.</dd>
      <dt>VertRes</dt>
      <dd>Vertical-axis modulation at 100 / 200 / 300 TVL. Drives
        whether horizontal lines and fine vertical detail stay sharp
        through scan converters, deinterlacers, and vertical
        sharpeners.</dd>
      <dt>Y/C</dt>
      <dd>Multi-frame dot-crawl analysis on row-9 chroma bursts plus
        chroma bandwidth. Dot crawl looks like shimmering dots that
        crawl along vertical color edges; narrow chroma bandwidth
        looks like washed-out fine color detail.</dd>
      <dt>ZonePlate</dt>
      <dd>Cross-color injected into the moving zone-plate area in the
        center of the chart (luma-only by design). Drives whether
        moving high-detail content acquires false color.</dd>
      <dt>Overall</dt>
      <dd>Mean of the categories above. A rough single-number
        ranking; the column scores tell you where each processor
        spends its strengths and weaknesses.</dd>
    </dl>
  </details>
</section>
"""


def _yuv10_to_css_rgb(y10: float, u10: float, v10: float) -> str:
    """CSS rgb() wrapper around tp_chart.yuv10_to_rgb8 so visual-overview
    swatches stay bit-for-bit consistent with the rest of the report."""
    r, g, b = tp_chart.yuv10_to_rgb8(float(y10), float(u10), float(v10))
    return f"rgb({r},{g},{b})"


_TARTAN_LABELS = {
    "YEL":  "Yel",  "CYN":  "Cyn",  "BLU":  "Blu",  "RED":  "Red",
    "MAG":  "Mag",  "GRN":  "Grn",  "RED2": "Red²", "CYN2": "Cyn²",
}
_GRAY_LABELS = {
    "G1": "20 %",  "G2": "40 %",  "G3": "60 %",  "G4": "80 %",
}


def render_visual_color_summary(captures: List[Dict[str, Any]]) -> str:
    """Compact side-by-side color swatch table: each cell shows the
    capture's measured color next to the chart reference. Lets a reader
    eyeball color shift without parsing ΔE numbers."""
    if not captures:
        return ""
    tartan_ids = [r["id"] for r in tp_chart.TARTAN_REGIONS]
    gray_ids   = [r["id"] for r in tp_chart.GRAY_REGIONS]
    if not any(c.get("tartan") for c in captures) and \
       not any(c.get("grays") for c in captures):
        return ""

    def _tartan_index(c):
        return {r["id"]: r for r in (c.get("tartan") or [])}

    def _gray_index(c):
        return {r["id"]: r for r in (c.get("grays") or [])}

    # Reference swatches (from any capture — ideal values are chart-spec
    # constants, identical across all captures).
    ref_tartan = {}
    ref_gray = {}
    for c in captures:
        for r in (c.get("tartan") or []):
            ref_tartan.setdefault(r["id"], r.get("ideal_yuv10"))
        for r in (c.get("grays") or []):
            ref_gray.setdefault(r["id"], r.get("ideal_y10"))

    def _ref_swatch_tartan(rid):
        v = ref_tartan.get(rid)
        if v is None:
            return "rgb(128,128,128)"
        return _yuv10_to_css_rgb(v[0], v[1], v[2])

    def _ref_swatch_gray(rid):
        v = ref_gray.get(rid)
        if v is None:
            return "rgb(128,128,128)"
        return _yuv10_to_css_rgb(v, tp_chart.CHROMA_CENTER,
                                 tp_chart.CHROMA_CENTER)

    head_cells = ["<th class='visrow-cap'>Capture</th>"]
    for rid in tartan_ids:
        head_cells.append(
            f"<th class='visrow-color'>{_h.escape(_TARTAN_LABELS.get(rid, rid))}"
            f"<div class='visrow-refswatch' style='background:{_ref_swatch_tartan(rid)}'></div>"
            f"</th>"
        )
    for rid in gray_ids:
        head_cells.append(
            f"<th class='visrow-color'>{_h.escape(_GRAY_LABELS.get(rid, rid))}"
            f"<div class='visrow-refswatch' style='background:{_ref_swatch_gray(rid)}'></div>"
            f"</th>"
        )
    head = "<tr>" + "".join(head_cells) + "</tr>"

    rows = []
    for c in captures:
        ti = _tartan_index(c)
        gi = _gray_index(c)
        cells = [f"<td class='visrow-cap'>{_h.escape(_basename(c.get('_source_json_path','')))}</td>"]
        for rid in tartan_ids:
            entry = ti.get(rid)
            ref_css = _ref_swatch_tartan(rid)
            if entry and entry.get("measured_yuv10"):
                m = entry["measured_yuv10"]
                meas_css = _yuv10_to_css_rgb(m[0], m[1], m[2])
            else:
                meas_css = ref_css
            cells.append(
                f"<td class='visrow-pair'>"
                f"<div class='swatch' style='background:{meas_css}'></div>"
                f"<div class='swatch' style='background:{ref_css}'></div>"
                f"</td>"
            )
        for rid in gray_ids:
            entry = gi.get(rid)
            ref_css = _ref_swatch_gray(rid)
            if entry and entry.get("measured_y10") is not None:
                m_y = entry["measured_y10"]
                m_u = entry.get("u10", tp_chart.CHROMA_CENTER)
                m_v = entry.get("v10", tp_chart.CHROMA_CENTER)
                meas_css = _yuv10_to_css_rgb(m_y, m_u, m_v)
            else:
                meas_css = ref_css
            cells.append(
                f"<td class='visrow-pair'>"
                f"<div class='swatch' style='background:{meas_css}'></div>"
                f"<div class='swatch' style='background:{ref_css}'></div>"
                f"</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
<section class="visual-overview">
  <h2>Color &amp; grayscale — at a glance</h2>
  <p class="legend">
    Each cell shows the capture's measured patch on the left and the
    chart reference on the right, side-by-side with no border so the
    eye can pick up subtle hue or brightness shifts. Tartan columns
    are the 8 saturated 75 % color patches; the four grays are the
    20/40/60/80 % luminance steps. The small swatch under each column
    header is the reference color for that column.
  </p>
  <table class="visual-color-summary">
    <thead>{head}</thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def _capture_overview_data_url(c: Dict[str, Any], suffix: str) -> str:
    """Embed a per-capture overview sidecar PNG as a base64 data URL.
    `suffix` is one of tp_overview_crops.output_suffixes() values."""
    import base64
    src = c.get("_source_json_path")
    if not src:
        return ""
    stem = os.path.splitext(src)[0]
    path = stem + suffix
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return ("data:image/png;base64,"
                + base64.b64encode(f.read()).decode("ascii"))


# Columns shown in the visual frequency-response summary, in order.
# Suffix is resolved from tp_overview_crops.output_suffixes() at module
# load so it stays in lockstep with the sidecar generator — a rename
# there will fail the assertion below at import rather than silently
# rendering "—" cells in the report.
_VISFREQ_LABELS = [
    ("BURST_3p58",        "3.58 MHz",
     "NTSC color-subcarrier vertical bars. Clean luma bars should look identical."),
    ("BURST_4p43",        "4.43 MHz",
     "PAL color-subcarrier vertical bars. On a clean NTSC decoder these should pass through as pure luma."),
    ("BURST_300TVL_DIAG", "300 TVL diag",
     "Diagonal stripes at ~3.95 MHz. Tests off-axis resolution and cross-color."),
    ("BURST_400TVL_DIAG", "400 TVL diag",
     "Diagonal stripes at ~5.27 MHz — past most analog decoders' design bandwidth."),
    ("RADIAL_WEDGE",      "Radial wedge",
     "Siemens-star resolution probe. Wedges should fade smoothly to grey near the center; coloured fringing = cross-color."),
]


def _build_visfreq_cols():
    import tp_overview_crops
    suffixes = tp_overview_crops.output_suffixes()
    cols = []
    for region_id, label, tip in _VISFREQ_LABELS:
        if region_id not in suffixes:
            raise RuntimeError(
                f"visual-frequency column {region_id} has no matching "
                f"sidecar suffix in tp_overview_crops.output_suffixes()")
        cols.append((region_id, suffixes[region_id], label, tip))
    return cols


_VISFREQ_COLS = _build_visfreq_cols()


def render_visual_frequency_summary(captures: List[Dict[str, Any]]) -> str:
    """At-a-glance grid: reference row on top, then one row per capture,
    each showing the 4× upscaled crops of the key frequency-response
    cells. Lets a reader eyeball how each processor handles each
    pattern relative to the synth reference."""
    if not captures:
        return ""
    # Reference row uses synth crops generated inline.
    try:
        bgr = _synth_frame_bgr()
    except Exception:
        bgr = None

    def _synth_cell_url(region_id: str) -> str:
        if bgr is None:
            return ""
        if region_id == "RADIAL_WEDGE":
            cell = tp_chart.RADIAL_WEDGE["cell_box"]
            return _bgr_crop_to_png_data_url(bgr, cell, upscale=4)
        box = _burst_box_for(region_id)
        if box is None:
            return ""
        return _bgr_crop_to_png_data_url(bgr, box, upscale=4)

    head_cells = ["<th class='visfreq-cap'>Source</th>"]
    for _, _, label, tip in _VISFREQ_COLS:
        head_cells.append(_th(label, tip))
    head = "<tr>" + "".join(head_cells) + "</tr>"

    # Reference row.
    ref_cells = ["<td class='visfreq-cap visfreq-ref'>Reference (synth)</td>"]
    for region_id, _suffix, label, _tip in _VISFREQ_COLS:
        url = _synth_cell_url(region_id)
        if url:
            ref_cells.append(
                f"<td class='visfreq-img'><img src='{url}' alt='{_h.escape(label)} reference'/></td>"
            )
        else:
            ref_cells.append("<td class='visfreq-img'>—</td>")
    rows = ["<tr class='visfreq-refrow'>" + "".join(ref_cells) + "</tr>"]

    # Per-capture rows.
    for c in captures:
        cells = [f"<td class='visfreq-cap'>{_h.escape(_basename(c.get('_source_json_path','')))}</td>"]
        for _region_id, suffix, label, _tip in _VISFREQ_COLS:
            url = _capture_overview_data_url(c, suffix)
            if url:
                cells.append(
                    f"<td class='visfreq-img'><img src='{url}' alt='{_h.escape(label)}'/></td>"
                )
            else:
                cells.append("<td class='visfreq-img'>—</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
<section class="visual-overview">
  <h2>Frequency response — at a glance</h2>
  <p class="legend">
    Each row is a capture; each column is a critical test cell from
    the chart. The top row is the synthesized reference — what a
    clean signal path produces. Compare each capture row against the
    reference to see at a glance which processors keep the pattern
    intact and which lose modulation, blur, or introduce cross-color
    fringes. The numerical breakdown for each cell is in the detail
    sections below.
  </p>
  <table class="visual-frequency-summary">
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
# Vertical-axis frequency response (col 1, rows 4..6).
# ---------------------------------------------------------------------

def _vertical_response_data(c: Dict[str, Any]):
    return c.get("vertical_response") or {}


def _vmod_class(pct):
    if pct is None:
        return ""
    if pct >= 70:
        return "delta-good"
    if pct >= 40:
        return "delta-warn"
    return "delta-bad"


def _score_vertical_response(c: Dict[str, Any]) -> float:
    """0-100. Higher mean modulation across the 100/200/300 TVL
    vertical bursts = better vertical-axis resolution. Penalise the
    shortfall from 100 %."""
    s = (_vertical_response_data(c).get("summary") or {})
    mean = s.get("mean_modulation_pct")
    if mean is None:
        return float("nan")
    # 100 % = no penalty; 50 % = full penalty.
    pen = _clamp((100.0 - mean) * 1.5, 0, 100)
    return max(0.0, 100.0 - pen)


def render_vertical_response_overview(captures: List[Dict[str, Any]]) -> str:
    head = (
        "<tr>"
        + _sortable_th("Capture",
                       "Capture file name.", kind="text")
        + _sortable_th("Mean modulation %",
                       "Mean luma modulation across the three "
                       "vertical bursts. Higher = the decoder "
                       "preserves vertical detail.")
        + _sortable_th("100 TVL %",
                       "Modulation at the 100 TVL vertical burst "
                       "(easy — well within vertical resolution).")
        + _sortable_th("200 TVL %",
                       "Modulation at the 200 TVL vertical burst.")
        + _sortable_th("300 TVL %",
                       "Modulation at the 300 TVL vertical burst — "
                       "the stiffest vertical-response test.")
        + "</tr>"
    )
    rows = []
    for c in captures:
        name = _basename(c["_meta"]["capture"])
        s = (_vertical_response_data(c).get("summary") or {})
        mean = s.get("mean_modulation_pct")
        m100 = s.get("modulation_at_100tvl")
        m200 = s.get("modulation_at_200tvl")
        m300 = s.get("modulation_at_300tvl")
        cells = [
            _td_name(name),
            _td_num(mean, "{:.1f}", cls=_vmod_class(mean)),
            _td_num(m100, "{:.1f}", cls=_vmod_class(m100)),
            _td_num(m200, "{:.1f}", cls=_vmod_class(m200)),
            _td_num(m300, "{:.1f}", cls=_vmod_class(m300)),
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<section class="overview">
  <h2>Vertical Response Overview</h2>
  <p class="legend">
    Headline numbers from the three slightly-oblique horizontal-stripe
    bursts on the left edge of the chart at 100, 200, and 300 TVL.
    Modulation is the peak-to-peak luma swing in the burst, as a
    percent of full chart contrast — higher = the decoder preserves
    vertical detail.
  </p>
  <table class="overview-table">
    <thead>{head}</thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def _vmod_verdict(pct):
    if pct is None:
        return "—"
    if pct >= 80:
        return f"clean ({pct:.0f}%)"
    if pct >= 50:
        return f"moderate rolloff ({pct:.0f}%)"
    if pct >= 20:
        return f"heavy rolloff ({pct:.0f}%)"
    return f"vertical resolution lost ({pct:.0f}%)"


def _render_vertical_response_panel(c: Dict[str, Any]) -> str:
    cap_name = _basename(c["_meta"]["capture"])
    vr = _vertical_response_data(c)
    if not vr:
        return (f"<div class='color-panel'><h3>{_h.escape(cap_name)}</h3>"
                f"<p class='muted'>no vertical response data — re-run "
                f"tp_measure.</p></div>")
    regions = vr.get("regions") or []
    summary = vr.get("summary") or {}
    rows_html = []
    for r in regions:
        tvl = r.get("target_tvl")
        mod = r.get("modulation_pct")
        snr = r.get("snr_db")
        det = r.get("detected_freq_cycles_per_row")
        target = r.get("target_freq_cycles_per_row")
        if mod is None:
            rows_html.append(
                f"<tr><td class='name'>{tvl} TVL</td>"
                f"<td colspan='4' class='muted small'>"
                f"{r.get('error', 'not measured')}</td></tr>"
            )
            continue
        rows_html.append(
            f"<tr>"
            f"<td class='name'>{tvl} TVL</td>"
            f"<td class='delta'>"
            f"{(target * 486 * 2 if target else 0):.0f} "
            f"<span class='muted small'>(target)</span></td>"
            f"<td class='delta'>"
            f"{(det * 486 * 2 if det else 0):.0f} "
            f"<span class='muted small'>(detected)</span></td>"
            f"<td class='delta {_vmod_class(mod)}'>{mod:.1f}%</td>"
            f"<td class='delta'>"
            f"{('—' if snr is None else f'{snr:.1f} dB')}</td>"
            f"<td class='verdict'>{_h.escape(_vmod_verdict(mod))}</td>"
            f"</tr>"
        )
    mean = summary.get("mean_modulation_pct")
    summary_html = (
        "<ul class='geo-list'>"
        f"<li>Mean modulation across the 3 bursts: <b>"
        f"<span class='{_vmod_class(mean)}'>"
        f"{('—' if mean is None else f'{mean:.1f}%')}</span></b></li>"
        f"<li>Rolloff: <b>"
        f"{(summary.get('modulation_at_100tvl') or 0):.0f}% → "
        f"{(summary.get('modulation_at_200tvl') or 0):.0f}% → "
        f"{(summary.get('modulation_at_300tvl') or 0):.0f}%</b> "
        f"<span class='muted'>(100 → 200 → 300 TVL).</span></li>"
        "</ul>"
    )
    return (
        f"<div class='color-panel'>"
        f"<h3>{_h.escape(cap_name)}</h3>"
        f"<table class='color-table'>"
        f"<tr>{_th('Burst', 'Vertical-frequency burst.')}"
        f"{_th('Target', 'Chart-spec TVL of this burst, computed from the labelled frequency.')}"
        f"{_th('Detected', 'TVL inferred from the FFT peak — small deviations from target are sampling noise.')}"
        f"{_th('Modulation', 'Peak-to-peak luma swing at the burst frequency, as a percent of full chart contrast (black→white = 100 %).')}"
        f"{_th('SNR', 'Signal-to-noise ratio of the peak vs the median FFT magnitude. Higher = burst clearly resolved.')}"
        f"{_th('Verdict', 'Plain-language summary.')}"
        f"</tr>"
        + "".join(rows_html) +
        f"</table>"
        f"<h4>Summary</h4>{summary_html}"
        f"</div>"
    )


def render_vertical_response_panels(captures: List[Dict[str, Any]]) -> str:
    if not any(c.get("vertical_response") for c in captures):
        return ""
    intro = """
<p class="legend">
  The slightly-oblique near-horizontal stripe bursts in col 1, rows
  4 / 5 / 6 probe vertical-axis resolution at 100, 200, and 300 TVL.
  The stripes are deliberately tilted a few degrees so any straight-
  horizontal interlace-comb or vertical-aperture filter inside the
  decoder shows up as additional asymmetric modulation.
</p>
<p class="legend">
  <b>Why this matters:</b> the vertical-frequency response is what
  scan converters, deinterlacers, and vertical-aperture sharpeners
  modify. A decoder that rolls off above 200 TVL gives a soft
  picture; one with too much vertical peaking produces ringing /
  edge enhancement that looks crisp on test patterns but artificial
  on real content. Modulation should stay high at 100 and 200 TVL
  with a graceful taper toward 300 TVL.
</p>
"""
    panels = [_render_vertical_response_panel(c) for c in captures]
    return f"""
<section class="vertical-response-panels">
  <h2>Vertical Response — 100 / 200 / 300 TVL bursts</h2>
  {intro}
  {''.join(panels)}
</section>
"""


# ---------------------------------------------------------------------
# Zone plate — chroma leak into the moving luma-only zone pattern.
# ---------------------------------------------------------------------

def _zone_plate_data(c: Dict[str, Any]):
    return c.get("zone_plate") or {}


def _zp_chroma_class(rms):
    if rms is None:
        return ""
    if rms < 30:
        return "delta-good"
    if rms < 80:
        return "delta-warn"
    return "delta-bad"


def _score_zone_plate(c: Dict[str, Any]) -> float:
    """0-100. Lower chroma RMS in the zone plate region = cleaner Y/C
    separation on high-frequency moving luma content."""
    zp = _zone_plate_data(c)
    if not zp:
        return float("nan")
    mean = zp.get("mean_chroma_rms")
    if mean is None:
        return float("nan")
    # 20 rms = noise floor (no penalty); 200 rms = full penalty.
    pen = _clamp((mean - 20.0) / 1.8, 0, 100)
    return max(0.0, 100.0 - pen)


def render_zone_plate_overview(captures: List[Dict[str, Any]]) -> str:
    head = (
        "<tr>"
        + _sortable_th("Capture",
                       "Capture file name.", kind="text")
        + _sortable_th("Mean chroma RMS",
                       "Mean chroma RMS over the moving zone-plate "
                       "region (cells 3..6, 4..9), averaged across "
                       "3 successive frames. The zone plate is "
                       "luma-only, so any chroma here is decoder "
                       "cross-color. Lower is better.")
        + _sortable_th("Min",
                       "Single-frame minimum chroma RMS over the "
                       "3-frame stack.")
        + _sortable_th("Max",
                       "Single-frame maximum chroma RMS over the "
                       "3-frame stack.")
        + _sortable_th("Frame-to-frame std",
                       "Std-dev of the per-frame chroma RMS values. "
                       "Non-zero = the cross-color amount varies as "
                       "the zone plate moves; near-zero = the "
                       "cross-color is locked to a static feature.")
        + _sortable_th("Chroma present?",
                       "Binary flag (mean RMS above the 15-code "
                       "noise-floor threshold).", kind="text")
        + "</tr>"
    )
    rows = []
    for c in captures:
        name = _basename(c["_meta"]["capture"])
        zp = _zone_plate_data(c)
        mean = zp.get("mean_chroma_rms")
        mn = zp.get("min_chroma_rms")
        mx = zp.get("max_chroma_rms")
        std = zp.get("frame_std_chroma_rms")
        present = zp.get("chroma_present")
        present_str = ("—" if present is None
                       else ("yes" if present else "no"))
        present_class = ("" if present is None
                         else ("delta-bad" if present else "delta-good"))
        cells = [
            _td_name(name),
            _td_num(mean, "{:.1f}", cls=_zp_chroma_class(mean)),
            _td_num(mn,   "{:.1f}"),
            _td_num(mx,   "{:.1f}"),
            _td_num(std,  "{:.2f}"),
            f'<td class="name {present_class}" '
            f'data-v="{0 if present else 1}">{present_str}</td>',
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<section class="overview">
  <h2>Zone Plate Overview</h2>
  <p class="legend">
    Per-capture summary of decoder cross-color injected into the
    moving zone-plate area in the center of the chart. The chart
    renders the zone plate as luma-only; any chroma we measure here
    is the decoder mistaking high-frequency luma for chroma. Mean
    chroma RMS is averaged across 3 successive frames to smooth out
    the moving-pattern variation.
  </p>
  <table class="overview-table">
    <thead>{head}</thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def _zp_verdict(mean_rms):
    if mean_rms is None:
        return "—"
    if mean_rms < 30:
        return "clean — no significant cross-color"
    if mean_rms < 80:
        return f"moderate cross-color ({mean_rms:.0f} rms)"
    return f"heavy cross-color ({mean_rms:.0f} rms)"


def _render_zone_plate_panel(c: Dict[str, Any]) -> str:
    cap_name = _basename(c["_meta"]["capture"])
    zp = _zone_plate_data(c)
    if not zp:
        return (f"<div class='color-panel'><h3>{_h.escape(cap_name)}</h3>"
                f"<p class='muted'>no zone-plate data — re-run "
                f"tp_measure.</p></div>")
    per_frame = zp.get("per_frame") or []
    if not per_frame:
        return (f"<div class='color-panel'><h3>{_h.escape(cap_name)}</h3>"
                f"<p class='muted'>zone-plate region out of frame.</p>"
                f"</div>")
    rows_html = []
    for i, pf in enumerate(per_frame):
        rms = pf.get("chroma_rms")
        rows_html.append(
            f"<tr>"
            f"<td class='name'>Frame {i}</td>"
            f"<td class='delta {_zp_chroma_class(rms)}'>"
            f"{('—' if rms is None else f'{rms:.2f}')}</td>"
            f"</tr>"
        )
    mean = zp.get("mean_chroma_rms")
    min_rms = zp.get("min_chroma_rms")
    max_rms = zp.get("max_chroma_rms")
    std = zp.get("frame_std_chroma_rms")
    threshold = zp.get("threshold", 15.0)
    present = zp.get("chroma_present")
    n_frames = zp.get("n_frames")
    summary_html = (
        "<ul class='geo-list'>"
        f"<li>Frames analysed: <b>{n_frames}</b>.</li>"
        f"<li>Mean chroma RMS: <b>"
        f"<span class='{_zp_chroma_class(mean)}'>"
        f"{('—' if mean is None else f'{mean:.2f}')}</span></b> "
        f"— {_h.escape(_zp_verdict(mean))}.</li>"
        f"<li>Per-frame range: "
        f"min <b>{('—' if min_rms is None else f'{min_rms:.2f}')}</b>, "
        f"max <b>{('—' if max_rms is None else f'{max_rms:.2f}')}</b>, "
        f"std <b>{('—' if std is None else f'{std:.2f}')}</b> "
        f"<span class='muted'>(non-zero std = chroma varies as the "
        f"zone plate moves).</span></li>"
        f"<li>Above threshold ({threshold:.0f} rms): "
        f"<b>{('yes' if present else 'no')}</b> "
        f"<span class='muted'>(binary flag for the older "
        f"decoder-class rule).</span></li>"
        "</ul>"
    )
    return (
        f"<div class='color-panel'>"
        f"<h3>{_h.escape(cap_name)}</h3>"
        f"<table class='color-table'>"
        f"<tr>{_th('Frame', 'Index in the 3-frame stack (0 = the requested frame_index).')}"
        f"{_th('Chroma RMS', 'Chroma deviation from neutral, RMS-averaged over the entire zone-plate box. The chart renders the zone plate as luma-only so any chroma is decoder-injected.')}"
        f"</tr>"
        + "".join(rows_html) +
        f"</table>"
        f"<h4>Summary</h4>{summary_html}"
        f"</div>"
    )


def render_zone_plate_panels(captures: List[Dict[str, Any]]) -> str:
    if not any(c.get("zone_plate") for c in captures):
        return ""
    intro = """
<p class="legend">
  The center of the chart (cells 3..6, 4..9) carries a <b>moving
  zone-plate</b> pattern — concentric ring structure with no chroma.
  In an analog composite chain the high-frequency luma in this region
  spans and overlaps the color-subcarrier band, so a decoder with
  imperfect Y/C separation injects chroma where none was authored.
  We average chroma RMS over the same 3 frames used by the Y/C
  timing test so the moving pattern's frame-to-frame variation is
  smoothed out.
</p>
<p class="legend">
  <b>How to read the numbers:</b> chroma RMS &lt; 30 codes is clean
  (sits near the noise floor); 30–80 is moderate cross-color; &gt; 80
  is heavy cross-color (decoder is producing significant false
  color where the chart had none).
</p>
"""
    panels = [_render_zone_plate_panel(c) for c in captures]
    return f"""
<section class="zone-plate-panels">
  <h2>Zone Plate — cross-color on moving luma content</h2>
  {intro}
  {''.join(panels)}
</section>
"""


# ---------------------------------------------------------------------
# Y/C timing — chroma bursts in row 9 sampled across 3 frames.
# ---------------------------------------------------------------------

_YC_BURST_LABELS = {
    "YC_BURST_0p5MHZ": "0.5 MHz (blue / yellow)",
    "YC_BURST_1p0MHZ": "1.0 MHz (red / cyan)",
    "YC_BURST_1p5MHZ": "1.5 MHz (red / cyan)",
}


def _yc_timing_data(c: Dict[str, Any]):
    return c.get("yc_timing") or {}


def _wiggle_class(pct):
    if pct is None:
        return ""
    p = abs(pct)
    if p < 0.5:
        return "delta-good"
    if p < 2.0:
        return "delta-warn"
    return "delta-bad"


def _chroma_bw_class(mhz):
    if mhz is None:
        return ""
    if mhz >= 1.2:
        return "delta-good"
    if mhz >= 0.8:
        return "delta-warn"
    return "delta-bad"


def _score_yc_timing(c: Dict[str, Any]) -> float:
    """0-100. Penalises high dot-crawl wiggle (frame-to-frame motion
    of luma artefacts at chroma transitions) and narrow chroma
    bandwidth (chroma -3 dB frequency well below 1.0 MHz)."""
    s = (_yc_timing_data(c).get("summary") or {})
    pen = 0.0
    wiggle = s.get("mean_dot_crawl_wiggle_pct")
    if wiggle is not None:
        pen += _clamp(wiggle * 30.0, 0, 55)
    max_wiggle = s.get("max_dot_crawl_wiggle_pct")
    if max_wiggle is not None and max_wiggle > 2.0:
        pen += _clamp((max_wiggle - 2.0) * 5.0, 0, 15)
    f3db = s.get("chroma_minus_3db_MHz")
    if f3db is None:
        pen += 15
    elif f3db < 1.0:
        pen += _clamp((1.0 - f3db) * 30.0, 0, 20)
    return max(0.0, 100.0 - pen)


def render_yc_timing_overview(captures: List[Dict[str, Any]]) -> str:
    head = (
        "<tr>"
        + _sortable_th("Capture",
                       "Capture file name.", kind="text")
        + _sortable_th("Mean dot-crawl wiggle %",
                       "Mean pixel-wise luma std-dev across 3 frames "
                       "inside the YC chroma bursts, expressed as a "
                       "percent of the full luma swing. 0 = no "
                       "frame-to-frame motion; >1 = visible dot crawl "
                       "shimmering between fields.")
        + _sortable_th("Max wiggle %",
                       "Worst-pixel std-dev across 3 frames in any of "
                       "the bursts — surfaces a hotspot of frame-to-"
                       "frame motion even when the average is small.")
        + _sortable_th("Chroma -3 dB (MHz)",
                       "Highest frequency where the chroma magnitude "
                       "is still ≥ 70.8 % of its peak across the "
                       "0.5/1.0/1.5 MHz curve. Higher = wider chroma "
                       "bandwidth.")
        + _sortable_th("0.5 MHz chroma %",
                       "Single-frame chroma modulation at the 0.5 MHz "
                       "burst (blue/yellow stripes).")
        + _sortable_th("1.5 MHz chroma %",
                       "Single-frame chroma modulation at the 1.5 MHz "
                       "burst (red/cyan stripes).")
        + "</tr>"
    )
    rows = []
    for c in captures:
        name = _basename(c["_meta"]["capture"])
        yc = _yc_timing_data(c)
        s = (yc.get("summary") or {})
        by_id = {r["id"]: r for r in (yc.get("regions") or [])}
        mean_w = s.get("mean_dot_crawl_wiggle_pct")
        max_w  = s.get("max_dot_crawl_wiggle_pct")
        f3db   = s.get("chroma_minus_3db_MHz")
        ch05 = (by_id.get("YC_BURST_0p5MHZ") or {}).get("chroma_modulation_pct")
        ch15 = (by_id.get("YC_BURST_1p5MHZ") or {}).get("chroma_modulation_pct")
        cells = [
            _td_name(name),
            _td_num(mean_w, "{:.3f}", cls=_wiggle_class(mean_w)),
            _td_num(max_w,  "{:.2f}", cls=_wiggle_class(max_w)),
            _td_num(f3db,   "{:.2f}", cls=_chroma_bw_class(f3db)),
            _td_num(ch05,   "{:.1f}"),
            _td_num(ch15,   "{:.1f}"),
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<section class="overview">
  <h2>Y/C Timing Overview</h2>
  <p class="legend">
    Headline numbers from the row-9 chroma bursts (blue/yellow at
    0.5 MHz, red/cyan at 1.0 and 1.5 MHz). The wiggle column is the
    multi-frame metric — it measures how much luma at chroma
    transitions oscillates between successive frames, which is
    exactly the dot-crawl shimmer the test is designed to expose.
  </p>
  <table class="overview-table">
    <thead>{head}</thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def _yc_verdict(region) -> str:
    wiggle = region.get("dot_crawl_wiggle_pct")
    chroma = region.get("chroma_modulation_pct")
    parts = []
    if wiggle is None:
        parts.append("wiggle not measured")
    elif wiggle < 0.5:
        parts.append("no dot-crawl wiggle")
    elif wiggle < 2.0:
        parts.append(f"mild wiggle ({wiggle:.2f}%)")
    else:
        parts.append(f"heavy wiggle ({wiggle:.2f}%)")
    if chroma is not None:
        if chroma >= 30:
            parts.append("strong chroma")
        elif chroma >= 5:
            parts.append("normal chroma")
        else:
            parts.append("weak chroma")
    return "; ".join(parts)


def _render_yc_timing_panel(c: Dict[str, Any]) -> str:
    cap_name = _basename(c["_meta"]["capture"])
    yc = _yc_timing_data(c)
    if not yc:
        return (f"<div class='color-panel'><h3>{_h.escape(cap_name)}</h3>"
                f"<p class='muted'>no Y/C timing data — re-run "
                f"tp_measure.</p></div>")
    summary = yc.get("summary") or {}
    regions = yc.get("regions") or []
    n_frames = summary.get("frames_used") or len(regions[0].get("per_frame", [])) if regions else 0
    rows_html = []
    for r in regions:
        label = _YC_BURST_LABELS.get(r["id"], r["id"])
        chroma = r.get("chroma_modulation_pct")
        # Per-frame chroma values from the per_frame array.
        per_frame = r.get("per_frame") or []
        pf_chroma = ", ".join(f"{pf['chroma_modulation_pct']:.1f}%"
                              for pf in per_frame)
        wiggle = r.get("dot_crawl_wiggle_pct")
        wiggle_max = r.get("dot_crawl_wiggle_max_pct")
        rows_html.append(
            f"<tr>"
            f"<td class='name'>{_h.escape(label)}</td>"
            f"<td class='delta'>"
            f"{('—' if chroma is None else f'{chroma:.1f}%')}"
            f" <span class='muted small'>(per frame: {_h.escape(pf_chroma) or '—'})</span></td>"
            f"<td class='delta {_wiggle_class(wiggle)}'>"
            f"{('—' if wiggle is None else f'{wiggle:.3f}%')}</td>"
            f"<td class='delta {_wiggle_class(wiggle_max)}'>"
            f"{('—' if wiggle_max is None else f'{wiggle_max:.2f}%')}</td>"
            f"<td class='verdict'>{_h.escape(_yc_verdict(r))}</td>"
            f"</tr>"
        )

    mean_wiggle = summary.get("mean_dot_crawl_wiggle_pct")
    max_wiggle  = summary.get("max_dot_crawl_wiggle_pct")
    f3db        = summary.get("chroma_minus_3db_MHz")
    f6db        = summary.get("chroma_minus_6db_MHz")
    def _verdict_dot_crawl(w):
        if w is None: return "—"
        if w < 0.5:   return "clean — no detectable dot-crawl wiggle"
        if w < 2.0:   return "mild dot-crawl wiggle"
        return "heavy dot-crawl wiggle"
    def _verdict_chroma(f):
        if f is None: return "—"
        if f >= 1.2:  return "wide chroma bandwidth"
        if f >= 0.8:  return "moderate chroma bandwidth"
        return "narrow chroma bandwidth"

    summary_html = (
        "<ul class='geo-list'>"
        f"<li>Frames analysed: <b>{n_frames}</b> "
        f"<span class='muted'>(consecutive frames starting at the "
        f"requested frame_index).</span></li>"
        f"<li>Mean dot-crawl wiggle: "
        f"<span class='{_wiggle_class(mean_wiggle)}'>"
        f"{('—' if mean_wiggle is None else f'{mean_wiggle:.3f}%')}</span> "
        f"— {_h.escape(_verdict_dot_crawl(mean_wiggle))}. "
        f"<span class='muted'>(peak pixel "
        f"{('—' if max_wiggle is None else f'{max_wiggle:.2f}%')})</span></li>"
        f"<li>Chroma <b>-3 dB</b> cutoff: <b>"
        f"{('not reached in [0.5, 1.5 MHz]' if f3db is None else f'{f3db:.2f} MHz')}</b> "
        f"— {_h.escape(_verdict_chroma(f3db))}.</li>"
        f"<li>Chroma <b>-6 dB</b> cutoff: <b>"
        f"{('not reached in [0.5, 1.5 MHz]' if f6db is None else f'{f6db:.2f} MHz')}</b>.</li>"
        "</ul>"
    )

    return (
        f"<div class='color-panel'>"
        f"<h3>{_h.escape(cap_name)}</h3>"
        f"<table class='color-table'>"
        f"<tr>{_th('Burst', 'Which Y/C timing burst.')}"
        f"{_th('Chroma modulation', 'Single-frame chroma modulation at the burst frequency. Per-frame values shown alongside for reference; small variations between frames are sampling noise, not a defect.')}"
        f"{_th('Dot-crawl wiggle (mean)', 'Mean pixel-wise std-dev of luma across the captured frames inside the burst, as a percent of full luma swing. A perfectly-stable image gives ~0 here; non-zero is decoder-injected dot-crawl moving between fields.')}"
        f"{_th('Dot-crawl wiggle (peak)', 'Single-pixel maximum of the same std-dev — flags hotspots of frame-to-frame motion that may not dominate the average.')}"
        f"{_th('Verdict', 'Plain-language summary.')}"
        f"</tr>"
        + "".join(rows_html) +
        f"</table>"
        f"<h4>Summary</h4>{summary_html}"
        f"</div>"
    )


def render_yc_timing_panels(captures: List[Dict[str, Any]]) -> str:
    if not any(c.get("yc_timing") for c in captures):
        return ""
    intro = """
<p class="legend">
  Row 9 of the chart carries three alternating-chroma stripe patterns:
  blue/yellow at 0.5 MHz (cells 9,6 + 9,7) and red/cyan at 1.0 MHz
  (cell 9,5) and 1.5 MHz (cell 9,8). The stripes are designed so that
  Y and C transition at the same x-position — a Y/C-timing-aligned
  decoder reproduces the stripes cleanly while a misaligned one
  shifts the colors away from the luma edges.
</p>
<p class="legend">
  <b>Why multi-frame analysis?</b> Many composite-NTSC decoders inject
  a small luma pattern at the color-subcarrier frequency at every
  chroma transition (the dot-crawl artifact). Because the subcarrier
  phase advances by ≈ 162° from one frame to the next on NTSC, that
  injected pattern <i>moves</i> between successive frames — the
  classic shimmering dots that crawl up vertical color edges. A
  single-frame measurement misses this motion entirely. We sample
  the bursts on 3 consecutive frames and compute the pixel-wise
  std-dev of luma across the stack; a static, well-decoded image
  gives ~0, while a decoder with cross-luma produces a clearly non-
  zero number.
</p>
<p class="legend">
  We also derive a <b>chroma bandwidth</b> proxy: interpolated -3 dB
  and -6 dB crossings on the 0.5/1.0/1.5 MHz chroma-magnitude curve.
  Wider bandwidth → the decoder preserves higher-frequency chroma
  detail (smoother color transitions and more saturated narrow
  features).
</p>
"""
    panels = [_render_yc_timing_panel(c) for c in captures]
    return f"""
<section class="yc-timing-panels">
  <h2>Y/C Timing — dot-crawl wiggle &amp; chroma bandwidth</h2>
  {intro}
  {''.join(panels)}
</section>
"""


# ---------------------------------------------------------------------
# Radial wedge — resolution limit + decoder cross-effects (cell 8,11).
# ---------------------------------------------------------------------

def _radial_wedge_data(c: Dict[str, Any]):
    return c.get("radial_wedge") or {}


def _tvl_class(tvl):
    if tvl is None:
        return ""
    if tvl >= 400:
        return "delta-good"
    if tvl >= 300:
        return "delta-warn"
    return "delta-bad"


def _hv_ratio_class(ratio):
    if ratio is None:
        return ""
    d = abs(ratio - 1.0)
    if d < 0.05:
        return "delta-good"
    if d < 0.15:
        return "delta-warn"
    return "delta-bad"


def _score_radial_wedge(c: Dict[str, Any]) -> float:
    """0-100. Penalises low resolution (TVL well below the chart's max),
    cross-color leak into the luma-only wedge, and H/V aperture
    asymmetry (decoder sharpens one axis more than the other)."""
    rw = _radial_wedge_data(c)
    if not rw:
        return float("nan")
    pen = 0.0
    rl = rw.get("resolution_limit") or {}
    tvl = rl.get("tvl")
    # Target: chart-max ≈ 450 TVL. Penalise shortfall below 400 TVL.
    if tvl is not None:
        if tvl < 400:
            pen += _clamp((400 - tvl) / 5.0, 0, 40)
    else:
        pen += 25
    cc = rw.get("cross_color_chroma_rms")
    if cc is not None:
        pen += _clamp((cc - 5.0) / 4.0, 0, 30)
    ratio = rw.get("hv_ratio")
    if ratio is not None:
        pen += _clamp(abs(ratio - 1.0) * 80.0, 0, 20)
    return max(0.0, 100.0 - pen)


_SYNTH_RADIAL_WEDGE_URL = None


def _synth_radial_wedge_data_url(upscale: int = 8) -> str:
    """Embed an 8×-upscaled crop of the synth radial wedge as a
    reference baseline."""
    global _SYNTH_RADIAL_WEDGE_URL
    if _SYNTH_RADIAL_WEDGE_URL is not None:
        return _SYNTH_RADIAL_WEDGE_URL
    try:
        import base64
        import cv2
        bgr = _synth_frame_bgr()
        rw = tp_chart.RADIAL_WEDGE
        x, y, w, h = rw["cell_box"]
        crop = bgr[y:y + h, x:x + w]
        if upscale > 1:
            crop = cv2.resize(
                crop, (crop.shape[1] * upscale, crop.shape[0] * upscale),
                interpolation=cv2.INTER_NEAREST,
            )
        ok, png = cv2.imencode(".png", crop)
        if not ok:
            return ""
        _SYNTH_RADIAL_WEDGE_URL = (
            "data:image/png;base64,"
            + base64.b64encode(png.tobytes()).decode("ascii")
        )
    except Exception:
        _SYNTH_RADIAL_WEDGE_URL = ""
    return _SYNTH_RADIAL_WEDGE_URL


def _capture_radial_wedge_data_url(c: Dict[str, Any]) -> str:
    import base64
    src = c.get("_source_json_path")
    if not src:
        return ""
    stem = os.path.splitext(src)[0]
    path = stem + "_radial_wedge.png"
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return ("data:image/png;base64,"
                + base64.b64encode(f.read()).decode("ascii"))


def render_radial_wedge_overview(captures: List[Dict[str, Any]]) -> str:
    head = (
        "<tr>"
        + _sortable_th("Capture",
                       "Capture file name.", kind="text")
        + _sortable_th("Resolution limit (TVL)",
                       "Highest TV-line count the decoder can still "
                       "resolve in the radial wedge (the smallest "
                       "radius where wedge modulation is still ≥ 50 % "
                       "of its peak). Higher = sharper.")
        + _sortable_th("Cross-color (rms)",
                       "Chroma RMS over the wedge box. The wedge is "
                       "black-and-white only, so any chroma here is "
                       "decoder cross-color.")
        + _sortable_th("H/V ratio",
                       "Std-dev ratio of luma along a horizontal vs "
                       "vertical cross-section through the wedge "
                       "center. 1.00 = balanced; >1 = decoder "
                       "sharpens horizontally more than vertically; "
                       "<1 = vice versa.")
        + _sortable_th("N pairs",
                       "Number of black/white wedge pairs detected by "
                       "FFT of the outer radii. The chart's design "
                       "value is 16; a different number on a real "
                       "capture usually means the wedge fell partly "
                       "outside the registration window.")
        + "</tr>"
    )
    rows = []
    for c in captures:
        name = _basename(c["_meta"]["capture"])
        rw = _radial_wedge_data(c)
        rl = rw.get("resolution_limit") or {}
        tvl   = rl.get("tvl")
        cc    = rw.get("cross_color_chroma_rms")
        ratio = rw.get("hv_ratio")
        n     = rw.get("n_wedge_pairs_detected")
        cells = [
            _td_name(name),
            _td_num(tvl,   "{:.0f}", cls=_tvl_class(tvl)),
            _td_num(cc,    "{:.1f}", cls=_chroma_class(cc)),
            _td_num(ratio, "{:.2f}", cls=_hv_ratio_class(ratio)),
            _td_num(n,     "{:.0f}",
                    cls=("delta-good" if n is not None and 14 <= n <= 22
                         else "delta-warn")),
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<section class="overview">
  <h2>Radial Wedge Overview</h2>
  <p class="legend">
    Per-capture summary of the radial wedge in cell (8,11). The wedge
    is luma-only, so chroma here is cross-color (Y/C separation
    failure on a high-frequency luma pattern). The resolution limit
    is the highest TV-line equivalent the decoder still resolves
    cleanly; chart-design max is ≈ 450 TVL.
  </p>
  <table class="overview-table">
    <thead>{head}</thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def _render_radial_wedge_panel(c: Dict[str, Any]) -> str:
    cap_name = _basename(c["_meta"]["capture"])
    rw = _radial_wedge_data(c)
    if not rw:
        return (f"<div class='color-panel'><h3>{_h.escape(cap_name)}</h3>"
                f"<p class='muted'>no radial-wedge data — re-run "
                f"tp_measure.</p></div>")
    rl = rw.get("resolution_limit") or {}
    curve = rw.get("radial_modulation_curve") or []
    cc    = rw.get("cross_color_chroma_rms")
    ratio = rw.get("hv_ratio")
    n     = rw.get("n_wedge_pairs_detected")
    h_std = rw.get("h_modulation_std")
    v_std = rw.get("v_modulation_std")

    cap_url = _capture_radial_wedge_data_url(c)
    if cap_url:
        cap_img = (f"<figure class='radial-wedge-fig'>"
                   f"<img src='{cap_url}' alt='radial wedge crop'/>"
                   f"<figcaption>Decoded wedge "
                   f"(decoder output, 8× upscaled).</figcaption>"
                   f"</figure>")
    else:
        cap_img = ("<p class='muted small'>no radial-wedge crop "
                   "sidecar — re-run tp_measure.</p>")

    # Modulation profile mini-table — radius, modulation, FFT peak bin.
    mod_rows = "".join(
        f"<tr>"
        f"<td class='delta'>{entry['radius_px']:.0f}</td>"
        f"<td class='delta'>{entry['modulation_std']:.1f}</td>"
        f"<td class='delta'>{entry['fft_peak_bin']}</td>"
        f"</tr>"
        for entry in curve
    )

    tvl_text = (f"<b>{rl['tvl']:.0f} TVL</b>"
                if rl.get("tvl") is not None else "<b>—</b>")
    r_text = (f"r = {rl['radius_px']:.0f} px"
              if rl.get("radius_px") is not None else "")

    cc_verdict = ("clean — no cross-color"
                  if cc is not None and cc < 10
                  else f"chroma leak ({cc:.0f} rms)"
                  if cc is not None and cc < 100
                  else f"heavy cross-color ({cc:.0f} rms)"
                  if cc is not None else "—")
    if ratio is None:
        hv_verdict = "—"
    elif abs(ratio - 1.0) < 0.05:
        hv_verdict = "balanced aperture"
    elif ratio > 1.0:
        hv_verdict = (f"sharpens H more than V "
                      f"({(ratio - 1.0) * 100:+.0f} %)")
    else:
        hv_verdict = (f"sharpens V more than H "
                      f"({(1.0 - ratio) * 100:.0f} %)")

    summary_html = (
        "<ul class='geo-list'>"
        f"<li>Resolution limit: {tvl_text} "
        f"<span class='muted'>({r_text}, threshold 50 % of peak "
        f"modulation)</span>.</li>"
        f"<li>Cross-color: "
        f"<span class='{_chroma_class(cc)}'>"
        f"{('—' if cc is None else f'{cc:.1f} rms')}</span> "
        f"— {_h.escape(cc_verdict)}.</li>"
        f"<li>H/V aperture: "
        f"<span class='{_hv_ratio_class(ratio)}'>"
        f"{('—' if ratio is None else f'{ratio:.2f}')}</span> "
        f"— {_h.escape(hv_verdict)}. "
        f"<span class='muted'>(H std "
        f"{('—' if h_std is None else f'{h_std:.1f}')}, V std "
        f"{('—' if v_std is None else f'{v_std:.1f}')})</span></li>"
        f"<li>Detected wedge pairs: <b>"
        f"{('—' if n is None else n)}</b> "
        f"<span class='muted'>(chart design 16; "
        f"large deviations point at registration drift).</span></li>"
        "</ul>"
    )

    return (
        f"<div class='color-panel'>"
        f"<h3>{_h.escape(cap_name)}</h3>"
        f"<div class='wedge-row'>"
        f"<div class='wedge-img-col'>{cap_img}</div>"
        f"<div class='wedge-table-col'>"
        f"<table class='color-table'>"
        f"<tr>{_th('Radius (px)', 'Distance from wedge center.')}"
        f"{_th('Modulation', 'Std-dev of luma samples along the circle of this radius — high = wedges resolvable, low = blurred to grey.')}"
        f"{_th('FFT peak', 'Dominant angular frequency at this radius (cycles per circumference) from FFT.')}"
        f"</tr>"
        + mod_rows +
        f"</table>"
        f"<h4>Summary</h4>{summary_html}"
        f"</div></div></div>"
    )


def render_radial_wedge_panels(captures: List[Dict[str, Any]]) -> str:
    if not any(c.get("radial_wedge") for c in captures):
        return ""
    synth_url = _synth_radial_wedge_data_url()
    synth_block = (
        f"<figure class='radial-wedge-fig'>"
        f"<img src='{synth_url}' alt='synth radial wedge reference'/>"
        f"<figcaption>Synthetic reference (clean Siemens-star-style "
        f"radial wedge, 8× upscaled).</figcaption>"
        f"</figure>" if synth_url else ""
    )
    intro = """
<p class="legend">
  A small Siemens-star-style radial wedge sits in cell (8,11) — black
  and white pie slices radiating from a center, narrowing as they
  approach it. The local spatial frequency at each radius is set by
  the wedge geometry; the chart spec puts the highest frequency at
  ~450 TVL near the center, dropping to ~150 TVL at the outer edge.
</p>
<p class="legend">
  <b>What we look for:</b>
  <ul class="legend">
    <li><b>Resolution limit</b> — at what radius (and therefore what
    TVL equivalent) does the decoder stop resolving the wedges? Below
    the limit, the wedges blur into uniform mid-grey. We report the
    smallest radius where angular-modulation std-dev is still ≥ 50 %
    of its peak value.</li>
    <li><b>Cross-color</b> — the wedge is luma-only, so any chroma
    energy across the cell is decoder-injected cross-color. High
    values here mean the decoder is mistaking the wedge's
    high-frequency luma for chroma (especially near 3.58 MHz).</li>
    <li><b>H/V aperture asymmetry</b> — std-dev of luma along a
    horizontal cross-section divided by the same along a vertical
    cross-section through the wedge center. A decoder with more
    horizontal sharpening than vertical (or vice versa) pushes this
    ratio away from 1.0. SDI captures are usually 1.05-1.15 because
    NTSC's analog horizontal pipeline carries more bandwidth than
    its vertical (line-rate) one.</li>
  </ul>
</p>
"""
    panels = [_render_radial_wedge_panel(c) for c in captures]
    return f"""
<section class="radial-wedge-panels">
  <h2>Radial Wedge — resolution &amp; decoder cross-effects</h2>
  {intro}
  {synth_block}
  {''.join(panels)}
</section>
"""


# ---------------------------------------------------------------------
# Pulse-and-bar (2T pulse) analysis.
# ---------------------------------------------------------------------

PULSE_LABELS = {
    "PULSE_WOB": "White pulse on black",
    "PULSE_BOW": "Black pulse on white",
    "PULSE_WOG": "White pulse on 20% grey",
}

_PULSE_ORDER = ["PULSE_WOB", "PULSE_BOW", "PULSE_WOG"]


def _pulse_data(c: Dict[str, Any]):
    return c.get("pulse_response") or {}


def _pulse_region_by_id(c: Dict[str, Any], rid: str):
    data = _pulse_data(c)
    for r in data.get("regions") or []:
        if r.get("id") == rid:
            return r
    return None


def _ideal_2t_fwhm_ns():
    return tp_chart.NTSC_PULSE_2T_FWHM_PX / tp_chart.NTSC_SAMPLE_RATE_MHZ * 1000.0


def _fwhm_class(ns):
    if ns is None:
        return ""
    ideal = _ideal_2t_fwhm_ns()
    # Sharper than 2T (under-shoot of ideal) is also unexpected — flag
    # both directions. Tolerance ±50 ns ≈ ±0.7 px.
    if abs(ns - ideal) < 50:
        return "delta-good"
    if abs(ns - ideal) < 100:
        return "delta-warn"
    return "delta-bad"


def _ringing_class(pct):
    if pct is None:
        return ""
    p = abs(pct)
    if p < 5:
        return "delta-good"
    if p < 15:
        return "delta-warn"
    return "delta-bad"


def _echo_class(pct):
    if pct is None:
        return ""
    p = abs(pct)
    if p < 3:
        return "delta-good"
    if p < 8:
        return "delta-warn"
    return "delta-bad"


def _score_pulse(c: Dict[str, Any]) -> float:
    """0-100. Penalises FWHM broadening past ideal, excessive ringing on
    the white-on-grey pulse, and the presence of a black clipper that
    pins footroom values at pedestal.

    The FWHM penalty only kicks in past ~250 ns. Real decoders almost
    always broaden a 2T (200 ns) pulse to 220-260 ns just from BT.601
    filtering — that's not a defect.
    """
    s = (_pulse_data(c).get("summary") or {})
    pen = 0.0
    fwhm_tolerance_ns = 250.0
    for fwhm in (s.get("wob_fwhm_ns"), s.get("bow_fwhm_ns"),
                 s.get("wog_fwhm_ns")):
        if fwhm is None:
            continue
        if fwhm > fwhm_tolerance_ns:
            pen += _clamp((fwhm - fwhm_tolerance_ns) / 10.0, 0, 10)
    ringing = s.get("wog_ringing_pct")
    if ringing is not None:
        pen += _clamp(abs(ringing) * 1.2, 0, 25)
    echo = s.get("wog_echo_pct")
    if echo is not None:
        pen += _clamp(abs(echo) * 2.0, 0, 15)
    if s.get("black_clipper_present") is True:
        pen += 10
    return max(0.0, 100.0 - pen)


def _profile_to_png_data_url(profile, baseline_y10, peak_y10, polarity,
                             width_px: int = 240, height_px: int = 80):
    """Render the 1D pulse profile as a small spark-line PNG suitable
    for embedding in the report. The horizontal axis is sample number;
    the vertical axis goes from 0 to 1023 (full 10-bit Y range)."""
    import base64
    import cv2
    import numpy as np
    img = np.full((height_px, width_px, 3), 22, dtype=np.uint8)
    # Reference baseline line.
    by = int(round((1.0 - baseline_y10 / 1023.0) * (height_px - 1)))
    cv2.line(img, (0, by), (width_px - 1, by), (60, 65, 72), 1, cv2.LINE_AA)
    # Plot.
    n = len(profile)
    if n < 2:
        ok, png = cv2.imencode(".png", img)
        if not ok:
            return ""
        return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")
    pts = []
    for i, v in enumerate(profile):
        px = int(round(i * (width_px - 1) / (n - 1)))
        py = int(round((1.0 - max(0.0, min(1023.0, v)) / 1023.0)
                       * (height_px - 1)))
        pts.append((px, py))
    color = (140, 200, 240) if polarity > 0 else (240, 160, 140)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, a, b, color, 1, cv2.LINE_AA)
    # Mark BLACK_Y10 as a thin dashed line.
    bky = int(round((1.0 - tp_chart.BLACK_Y10 / 1023.0) * (height_px - 1)))
    for x in range(0, width_px, 4):
        img[bky, x:x + 2] = (40, 70, 100)
    ok, png = cv2.imencode(".png", img)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")


def render_pulse_overview(captures: List[Dict[str, Any]]) -> str:
    head = (
        "<tr>"
        + _sortable_th("Capture",
                       "Capture file name.", kind="text")
        + _sortable_th("WoB FWHM (ns)",
                       "Half-amplitude width of the white-on-black 2T "
                       "pulse, in nanoseconds. The chart spec is "
                       "200 ns (2T at 13.5 MHz NTSC SDI). Higher = the "
                       "decoder has broadened the pulse (luma-bandwidth "
                       "shortfall).")
        + _sortable_th("BoW FWHM (ns)",
                       "Same metric on the black-on-white pulse. "
                       "Asymmetry vs WoB is a clue to non-linear "
                       "transient response.")
        + _sortable_th("WoG ringing %",
                       "Max signed ringing on the white-on-grey pulse, "
                       "as a percent of the pulse amplitude. Positive = "
                       "overshoot in the same direction as the pulse; "
                       "negative = undershoot. Lower magnitude is better.")
        + _sortable_th("WoG echo %",
                       "Largest delayed copy of the pulse 15–25 px "
                       "downstream (a sign of multipath / filter "
                       "ringing). 0 = no echo.")
        + _sortable_th("Black clipper",
                       "Black-clipper detected on the WoB cell's "
                       "background: True = decoder pins values at Y=64 "
                       "and rejects footroom; False = footroom is "
                       "passed through.")
        + "</tr>"
    )
    rows = []
    for c in captures:
        name = _basename(c["_meta"]["capture"])
        s = (_pulse_data(c).get("summary") or {})
        wob_fwhm = s.get("wob_fwhm_ns")
        bow_fwhm = s.get("bow_fwhm_ns")
        wog_ring = s.get("wog_ringing_pct")
        wog_echo = s.get("wog_echo_pct")
        clip = s.get("black_clipper_present")
        clip_str = (
            "—" if clip is None
            else ("yes" if clip else "no")
        )
        clip_class = ("" if clip is None
                      else ("delta-bad" if clip else "delta-good"))
        cells = [
            _td_name(name),
            _td_num(wob_fwhm, "{:.0f}", cls=_fwhm_class(wob_fwhm)),
            _td_num(bow_fwhm, "{:.0f}", cls=_fwhm_class(bow_fwhm)),
            _td_num(wog_ring, "{:+.2f}", cls=_ringing_class(wog_ring)),
            _td_num(wog_echo, "{:.2f}",  cls=_echo_class(wog_echo)),
            f'<td class="numeric {clip_class}" data-v="{0 if clip else 1}">'
            f'{clip_str}</td>',
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<section class="overview">
  <h2>Pulse-and-Bar Overview</h2>
  <p class="legend">
    Per-capture summary of the three 2T pulse-and-bar cells on the
    right edge of the chart. Chart spec is 200 ns FWHM (2T at NTSC
    13.5 MHz SDI). The WoG cell is the primary ringing/echo probe;
    the WoB cell's background tells us whether the decoder allows
    SDI footroom (Y &lt; 64) or clips at the pedestal.
  </p>
  <table class="overview-table">
    <thead>{head}</thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def _pulse_verdict(r):
    fwhm = r.get("fwhm_ns")
    ring = r.get("max_ringing_pct")
    echo = r.get("echo_pct")
    parts = []
    if fwhm is None:
        parts.append("FWHM not measured")
    else:
        ideal = _ideal_2t_fwhm_ns()
        if abs(fwhm - ideal) < 30:
            parts.append(f"sharp ({fwhm:.0f} ns ≈ 2T)")
        elif fwhm > ideal:
            parts.append(f"broadened to {fwhm:.0f} ns "
                         f"({fwhm - ideal:+.0f} ns vs ideal)")
        else:
            parts.append(f"sharper than 2T ({fwhm:.0f} ns)")
    if ring is not None and abs(ring) >= 2:
        if ring > 0:
            parts.append(f"overshoot {ring:.1f}%")
        else:
            parts.append(f"undershoot {ring:.1f}%")
    if echo is not None and echo >= 2:
        parts.append(f"echo {echo:.1f}%")
    return "; ".join(parts) if parts else "—"


def _render_pulse_panel(c: Dict[str, Any]) -> str:
    cap_name = _basename(c["_meta"]["capture"])
    rows_html = []
    clip_text = ""
    for rid in _PULSE_ORDER:
        r = _pulse_region_by_id(c, rid)
        label = PULSE_LABELS[rid]
        if r is None or "error" in r:
            rows_html.append(
                f"<tr><td class='name'>{_h.escape(label)}</td>"
                f"<td colspan='6' class='muted small'>"
                f"{_h.escape(r['error'] if r else 'not measured')}</td>"
                f"</tr>"
            )
            continue
        profile = r.get("profile") or []
        polarity = +1 if "amplitude_y10" in r and r["amplitude_y10"] > 0 else -1
        spark = _profile_to_png_data_url(
            profile,
            baseline_y10=r.get("background_y10", 64.0),
            peak_y10=r.get("peak_y10", 940.0),
            polarity=polarity,
        )
        fwhm_ns = r.get("fwhm_ns")
        ring = r.get("max_ringing_pct")
        echo = r.get("echo_pct")
        amp = r.get("amplitude_abs_y10")
        amp_pct = r.get("amplitude_pct_full_scale")
        rows_html.append(
            f"<tr>"
            f"<td class='name'>{_h.escape(label)}</td>"
            f"<td class='swatch-cell'>"
            f"<img class='pulse-spark' src='{spark}' alt='{rid} profile'/>"
            f"</td>"
            f"<td class='delta'>{r.get('background_y10', 0.0):.0f} → "
            f"{r.get('peak_y10', 0.0):.0f} "
            f"<span class='muted small'>({amp_pct:.0f}% of full)</span></td>"
            f"<td class='delta {_fwhm_class(fwhm_ns)}'>"
            f"{('—' if fwhm_ns is None else f'{fwhm_ns:.0f} ns')}</td>"
            f"<td class='delta {_ringing_class(ring)}'>"
            f"{('—' if ring is None else f'{ring:+.1f}%')}</td>"
            f"<td class='delta {_echo_class(echo)}'>"
            f"{('—' if echo is None else f'{echo:.1f}%')}</td>"
            f"<td class='verdict'>{_h.escape(_pulse_verdict(r))}</td>"
            f"</tr>"
        )
        if rid == "PULSE_WOB" and r.get("clip"):
            clip = r["clip"]
            min_bg = clip.get("min_background_y10")
            n_below = clip.get("below_black_pixels")
            n_total = clip.get("background_pixels")
            clip_present = clip.get("clip_present")
            if clip_present:
                clip_text = (
                    "<p class='legend'>"
                    "<b>Black clipper detected.</b> Background pixels on "
                    "the WoB cell are pinned at Y10 ≥ "
                    f"{tp_chart.BLACK_Y10} with no footroom samples — "
                    "ringing that should dip below pedestal is being "
                    "clamped at black. Min Y10 observed: "
                    f"<b>{(min_bg if min_bg is not None else float('nan')):.0f}</b>."
                    "</p>"
                )
            else:
                clip_text = (
                    "<p class='legend'>"
                    f"<b>No black clipper.</b> {n_below}/{n_total} "
                    "background pixels sit in SDI footroom "
                    f"(Y10 &lt; {tp_chart.BLACK_Y10}); min Y10 = "
                    f"<b>{(min_bg if min_bg is not None else float('nan')):.0f}</b>. "
                    "Undershoots from pulse ringing are allowed to fall "
                    "below pedestal."
                    "</p>"
                )

    return (
        f"<div class='color-panel'>"
        f"<h3>{_h.escape(cap_name)}</h3>"
        f"<table class='color-table pulse-table'>"
        f"<tr>{_th('Pulse', 'Which pulse-and-bar cell. White-on-black tests the rising edge response and black-clipper behavior; black-on-white tests the falling edge; white-on-grey is the primary ringing/echo probe (background sits between black and white so symmetric ringing on either side is detectable).')}"
        f"{_th('Profile', '1D luma profile across the pulse, with the cell background drawn as the horizontal line and the BLACK_Y10 pedestal as a dashed reference. X axis = sample number.')}"
        f"{_th('Levels (Y10)', 'Measured background and peak pulse values in 10-bit luma codes, plus pulse amplitude as a percent of the chart full-contrast swing.')}"
        f"{_th('FWHM', 'Full width at half-maximum of the pulse, in nanoseconds. Chart spec = 200 ns (2T).')}"
        f"{_th('Ringing', 'Largest signed overshoot/undershoot adjacent to the pulse (3–12 px from center), as a percent of pulse amplitude.')}"
        f"{_th('Echo', 'Largest deviation 15–25 px from the pulse center — a check for delayed copies of the pulse (filter / multipath echo).')}"
        f"{_th('Verdict', 'Plain-language summary.')}"
        f"</tr>"
        + "".join(rows_html) +
        f"</table>"
        f"{clip_text}"
        f"</div>"
    )


def render_pulse_panels(captures: List[Dict[str, Any]]) -> str:
    if not any(c.get("pulse_response") for c in captures):
        return ""
    panels = [_render_pulse_panel(c) for c in captures]
    intro = """
<p class="legend">
  Three pulse-and-bar test cells on the right edge of the chart at
  column 12, rows 4 / 5 / 6 (just to the right of the frequency
  wedge):
  <ul class="legend">
    <li><b>White pulse on black</b> — exercises a sharp positive
    transient against a quiet background. The black-background area
    around the pulse also lets us detect whether the equipment has a
    <b>black clipper</b>.</li>
    <li><b>Black pulse on white</b> — the mirror test. Asymmetry vs
    the WoB pulse signals non-linear transient response.</li>
    <li><b>White pulse on 20%-grey</b> — the primary ringing/echo
    probe. The mid-grey background allows symmetric overshoot or
    undershoot on either side of the pulse to be measured cleanly,
    without saturation against the rail.</li>
  </ul>
</p>
<p class="legend">
  <b>About SDI "below black":</b> Each pulse is a sin²-shaped 2T
  pulse (≈ 200 ns FWHM at 13.5 MHz NTSC SDI). 10-bit limited-range
  SDI legally carries Y values from 1 to 1019; codes 1..63 are
  footroom (below the pedestal at Y=64), and ringing or undershoot
  on a sharp transition can legitimately fall into this range. A
  decoder with a black clipper clamps everything at Y=64, masking
  whatever distortion lives below pedestal. The WoB cell's
  background — far from the pulse — should sit at or near Y=64; if
  the captured chart's black pixels show <i>no</i> footroom samples
  AND sit exactly at the pedestal, a black clipper is in play.
</p>
"""
    return f"""
<section class="pulse-panels">
  <h2>Pulse-and-Bar — per capture</h2>
  {intro}
  {''.join(panels)}
</section>
"""


# ---------------------------------------------------------------------
# Chroma non-linearity staircase (3 magenta boxes, 33/66/100%).
# ---------------------------------------------------------------------

def _chroma_staircase_data(c: Dict[str, Any]):
    return c.get("chroma_staircase") or {}


def _score_chroma_staircase(c: Dict[str, Any]) -> float:
    """0-100. Penalises non-linear chroma scaling + phase wander."""
    cs = _chroma_staircase_data(c)
    s = cs.get("summary") or {}
    r2  = s.get("chroma_linearity_r2")
    dpd = s.get("differential_phase_deg")
    if r2 is None or (isinstance(r2, float) and r2 != r2):
        return float("nan")
    if dpd is None or (isinstance(dpd, float) and dpd != dpd):
        dpd = 0.0
    pen = _clamp((1.0 - r2) * 200, 0, 60) + _clamp(abs(dpd) * 4, 0, 40)
    return max(0.0, 100.0 - pen)


def _r2_class(r2) -> str:
    if r2 is None:
        return ""
    if r2 >= 0.99:
        return "delta-good"
    if r2 >= 0.95:
        return "delta-warn"
    return "delta-bad"


def _phase_class(deg) -> str:
    if deg is None:
        return ""
    d = abs(deg)
    if d < 3:
        return "delta-good"
    if d < 10:
        return "delta-warn"
    return "delta-bad"


def _step_dev_class(pct) -> str:
    if pct is None:
        return ""
    d = abs(pct)
    if d < 3:
        return "delta-good"
    if d < 8:
        return "delta-warn"
    return "delta-bad"


def render_chroma_staircase_overview(captures: List[Dict[str, Any]]) -> str:
    head = (
        "<tr>"
        + _sortable_th("Capture",
                       "Capture file name.", kind="text")
        + _sortable_th("Chroma R²",
                       "Linear-fit R² for chroma magnitude vs intended "
                       "level (33/66/100 %). 1.000 = perfectly linear; "
                       "&lt;0.99 = visible non-linearity in chroma gain.")
        + _sortable_th("Max step Δ %",
                       "Largest deviation of any of the 3 boxes from the "
                       "best-fit linear ramp, as a percent of the 100% "
                       "chroma magnitude. Lower = better.")
        + _sortable_th("Diff phase (°)",
                       "Max − min chroma phase across the 3 boxes. The "
                       "magenta hue should stay constant as saturation "
                       "rises; nonzero = differential phase error.")
        + _sortable_th("Luma R²",
                       "Linear-fit R² on the staircase luma. The boxes "
                       "are full magenta (R=B=N) so luma also scales "
                       "linearly with N; non-linearity here points at a "
                       "compressed gamma curve in the dark region.")
        + "</tr>"
    )
    rows = []
    for c in captures:
        name = _basename(c["_meta"]["capture"])
        s = (_chroma_staircase_data(c).get("summary") or {})
        r2 = s.get("chroma_linearity_r2")
        max_dev = s.get("max_step_deviation_pct")
        diff_phase = s.get("differential_phase_deg")
        y_r2 = s.get("luma_linearity_r2")
        cells = [
            _td_name(name),
            _td_num(r2,         "{:.4f}", cls=_r2_class(r2)),
            _td_num(max_dev,    "{:.2f}", cls=_step_dev_class(max_dev)),
            _td_num(diff_phase, "{:.2f}", cls=_phase_class(diff_phase)),
            _td_num(y_r2,       "{:.4f}", cls=_r2_class(y_r2)),
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<section class="overview">
  <h2>Chroma Non-Linearity Overview</h2>
  <p class="legend">
    Per-capture summary from the 3 magenta saturation boxes at the
    bottom-left of the chart (33 % → 66 % → 100 %). Chroma R² tells you
    how linearly the decoder reproduces increasing saturation;
    differential phase tells you whether the magenta hue stays put as
    saturation rises (a classic NTSC failure mode).
  </p>
  <table class="overview-table">
    <thead>{head}</thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def _render_chroma_staircase_panel(c: Dict[str, Any]) -> str:
    cap_name = _basename(c["_meta"]["capture"])
    cs = _chroma_staircase_data(c)
    regions = cs.get("regions") or []
    summary = cs.get("summary") or {}
    if not regions:
        return (f"<div class='color-panel'><h3>{_h.escape(cap_name)}</h3>"
                f"<p class='muted'>no chroma staircase data — re-run "
                f"tp_measure.</p></div>")
    rows_html = []
    step_devs = summary.get("step_deviations_pct") or []
    phase_steps = summary.get("phase_steps_deg") or []
    for i, r in enumerate(regions):
        rid = r["id"]
        lvl_pct = r["level"] * 100.0
        i_y10, i_u10, i_v10 = r["ideal_yuv10"]
        m_y10, m_u10, m_v10 = r["measured_yuv10"]
        ideal_rgb = tp_chart.yuv10_to_rgb8(i_y10, i_u10, i_v10)
        meas_rgb  = tp_chart.yuv10_to_rgb8(m_y10, m_u10, m_v10)
        chroma_mag = r["chroma_magnitude"]
        ideal_mag  = r["ideal_chroma_magnitude"]
        phase_deg  = r["chroma_phase_deg"]
        step_dev = step_devs[i] if i < len(step_devs) else None
        phase_step = phase_steps[i] if i < len(phase_steps) else None
        phase_cell = (
            f"{phase_deg:.2f}°"
            + (f" <span class='muted small'>(Δ {phase_step:+.2f}°)</span>"
               if phase_step is not None else "")
        )
        rows_html.append(
            f"<tr>"
            f"<td class='name'>{rid}</td>"
            f"<td class='muted small'>{lvl_pct:.1f}%</td>"
            f"<td class='swatch-cell'>{_swatch_inline(ideal_rgb, 'ideal')}</td>"
            f"<td class='swatch-cell'>{_swatch_inline(meas_rgb, 'measured')}</td>"
            f"<td class='delta'>{chroma_mag:.0f} "
            f"<span class='muted small'>(ideal {ideal_mag:.0f})</span></td>"
            f"<td class='delta {_phase_class(phase_step)}'>{phase_cell}</td>"
            f"<td class='delta {_step_dev_class(step_dev)}'>"
            f"{('—' if step_dev is None else f'{step_dev:+.2f}%')}</td>"
            f"</tr>"
        )
    body = "".join(rows_html)

    r2  = summary.get("chroma_linearity_r2")
    dpd = summary.get("differential_phase_deg")
    slope = summary.get("chroma_fit_slope")
    intercept = summary.get("chroma_fit_intercept")
    y_r2 = summary.get("luma_linearity_r2")

    def _verdict_lin(r2):
        if r2 is None: return "—"
        if r2 >= 0.99:  return "linear"
        if r2 >= 0.95:  return "slight non-linearity"
        return "non-linear"
    def _verdict_phase(d):
        if d is None: return "—"
        if abs(d) < 3:   return "phase stable"
        if abs(d) < 10:  return "mild phase shift"
        return "differential phase error"

    summary_html = (
        "<ul class='geo-list'>"
        f"<li>Chroma linearity: <b>R² = "
        f"{('—' if r2 is None else f'{r2:.4f}')}</b> — "
        f"<span class='{_r2_class(r2)}'>{_h.escape(_verdict_lin(r2))}</span>. "
        f"<span class='muted'>slope {('—' if slope is None else f'{slope:.1f}')} "
        f"chroma codes/level, intercept {('—' if intercept is None else f'{intercept:+.1f}')}</span>.</li>"
        f"<li>Differential phase: <b>"
        f"{('—' if dpd is None else f'{dpd:.2f}°')}</b> — "
        f"<span class='{_phase_class(dpd)}'>{_h.escape(_verdict_phase(dpd))}</span>.</li>"
        f"<li>Luma linearity: <b>R² = "
        f"{('—' if y_r2 is None else f'{y_r2:.4f}')}</b> "
        f"<span class='muted'>(staircase luma should also scale linearly "
        f"with level).</span></li>"
        "</ul>"
    )

    return (
        f"<div class='color-panel'>"
        f"<h3>{_h.escape(cap_name)}</h3>"
        f"<table class='color-table'>"
        f"<tr>{_th('ID', 'Region code.')}"
        f"{_th('Level', 'Intended chroma saturation (33 / 66 / 100 %).')}"
        f"{_th('Ref', 'Ideal swatch for full magenta at this saturation.')}"
        f"{_th('Cap', 'Measured swatch from the capture.')}"
        f"{_th('Chroma mag', 'Measured chroma magnitude in 10-bit codes (√(ΔU² + ΔV²)).')}"
        f"{_th('Phase', 'Measured chroma phase angle. Δ shows the shift relative to the 33% box.')}"
        f"{_th('Step Δ', 'Deviation from best-fit linear ramp, as a percent of the 100% chroma magnitude.')}"
        f"</tr>"
        + body +
        f"</table>"
        f"<h4>Summary</h4>{summary_html}"
        f"</div>"
    )


def render_chroma_staircase_panels(captures: List[Dict[str, Any]]) -> str:
    if not any(c.get("chroma_staircase") for c in captures):
        return ""
    panels = [_render_chroma_staircase_panel(c) for c in captures]
    intro = """
<p class="legend">
  Three boxes of full-saturation <b>magenta</b> rendered at 33.3 %,
  66.7 %, and 100 % chroma at the bottom-left of the chart. The
  staircase is designed to test two related artifacts:
  <ul class="legend">
    <li><b>Chroma non-linearity (differential gain)</b> — the decoder
    should reproduce chroma magnitude linearly: doubling the intended
    saturation should double the measured chroma. A non-linear curve
    means saturated colors are compressed or expanded relative to
    pastel colors.</li>
    <li><b>Differential phase</b> — the hue (chroma phase angle)
    should stay constant across all three boxes. A drift in phase as
    saturation rises is the classic NTSC differential-phase error and
    shows up as a hue shift between dark and bright versions of the
    same color.</li>
  </ul>
</p>
"""
    return f"""
<section class="chroma-staircase-panels">
  <h2>Chroma Non-Linearity — per capture</h2>
  {intro}
  {''.join(panels)}
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
    intro = """
<p class="legend">
  <b>What it shows:</b> The chart carries 8 small color patches in a
  tartan arrangement — the canonical SMPTE 75 % bar colors (yellow,
  cyan, blue, red on the top row; magenta, green, plus duplicate red
  and cyan on the bottom row). The bottom row deliberately repeats
  red and cyan in different positions to create vertical chroma
  transitions that stress comb-decoders.
</p>
<p class="legend">
  <b>What it measures:</b> How faithfully the decoder reproduces each
  patch's chrominance and luminance. We compare every measured patch
  to its Rec.601 chart-spec value and report ΔY / ΔU / ΔV (the raw
  10-bit code differences), ΔE (a plain Euclidean YUV10 distance you
  can rank against), and saturation as a percentage of the ideal
  chroma magnitude.
</p>
<p class="legend">
  <b>Why TV engineers and viewers care:</b> Color shifts here propagate
  to every other color the decoder reproduces. Skin tones, brand
  colors, and broadcast-graphics colors all live close to these
  SMPTE primaries; a few per-cent error in saturation or a small hue
  rotation produces unmistakably "wrong" color on real content.
  Engineers also use the bottom-row duplicates to gauge comb-decoder
  performance — the magenta/red vertical transition is where
  line-comb decoders typically leak chroma into luma.
</p>
<p class="legend">
  <b>How we characterize each capture:</b> One row per color with
  ideal-vs-measured swatches side-by-side, raw YUV10 deltas, an
  overall ΔE score, the measured saturation %, and a plain-language
  verdict ("match" / "close" / "off" plus saturation flags).
</p>
"""
    return f"""
<section class="tartan-panels">
  <h2>Color (Tartan) — per capture</h2>
  {intro}
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


# Wedge sample points in the order they appear top-to-bottom on the chart.
WEDGE_SAMPLE_FREQS = [
    ("WEDGE_2p0MHz",  "XC_WEDGE_2p0MHz",  2.0),
    ("WEDGE_2p5MHz",  "XC_WEDGE_2p5MHz",  2.5),
    ("WEDGE_3MHz",    "XC_WEDGE_3MHz",    3.0),
    ("WEDGE_3p5MHz",  "XC_WEDGE_3p5MHz",  3.5),
    ("WEDGE_4MHz",    "XC_WEDGE_4MHz",    4.0),
    ("WEDGE_4p5MHz",  "XC_WEDGE_4p5MHz",  4.5),
    ("WEDGE_5MHz",    "XC_WEDGE_5MHz",    5.0),
]


def _wedge_metrics(c: Dict[str, Any]):
    """Pull luma modulation % + chroma_rms for each wedge sample point.

    Returns list[(freq_MHz, modulation_pct, modulation_db, chroma_rms)],
    one entry per WEDGE_SAMPLE_FREQS in order. None values where the
    measurement is missing."""
    fr_regions  = ((c.get("frequency_response") or {}).get("regions") or {})
    art_regions = ((c.get("artifacts") or {}).get("regions") or {})
    out = []
    for bid, xid, freq in WEDGE_SAMPLE_FREQS:
        lu = fr_regions.get(bid) or {}
        xc = art_regions.get(xid) or {}
        out.append((
            freq,
            lu.get("modulation_pct"),
            lu.get("modulation_db"),
            xc.get("chroma_rms"),
        ))
    return out


def _wedge_minus_n_db_freq(curve, threshold_db: float):
    """Highest frequency where luma modulation >= threshold (db). Linear
    interpolation between adjacent sample points where the threshold is
    crossed. Returns None if the response is below threshold from the
    start, or above threshold all the way through (in which case it
    returns the last sampled freq)."""
    last_above = None
    for freq, _pct, db, _rms in curve:
        if db is None or db == float("-inf"):
            continue
        if db >= threshold_db:
            last_above = (freq, db)
        elif last_above is not None:
            f0, db0 = last_above
            if db0 == db:
                return float(f0)
            ratio = (db0 - threshold_db) / (db0 - db)
            return float(f0 + ratio * (freq - f0))
    if last_above is not None:
        return float(last_above[0])
    return None


def _wedge_first_chroma_freq(curve, rms_threshold: float = 10.0):
    """Lowest frequency where chroma_rms first exceeds the threshold —
    indicates where the decoder starts injecting cross-color into the
    wedge. None if no sample exceeds the threshold."""
    for freq, _pct, _db, rms in curve:
        if rms is None:
            continue
        if rms >= rms_threshold:
            return float(freq)
    return None


def _wedge_max_chroma_rms(curve):
    vals = [rms for _f, _p, _d, rms in curve if rms is not None]
    return float(max(vals)) if vals else None


def _score_frequency(c: Dict[str, Any]) -> float:
    """0-100 composite frequency score. Penalties cover:
      - row-2 burst chroma leak (decoder mistakes luma for chroma)
      - row-2 burst luma loss (decoder kills high-frequency luma)
      - wedge -6 dB cutoff below 4.0 MHz (low luma bandwidth)
      - wedge chroma intrusion above 2.5 MHz (cross-color)
    """
    pen = 0.0
    # Row-2 luma + chroma penalties (max ~50 pts).
    for b in FREQ_BURST_REGIONS:
        m = _freq_burst_metrics(c, b["id"], b["xc_id"])
        lu = m["luma_modulation_pct"]
        xc = m["chroma_leak_rms"]
        if lu is not None and lu < 30:
            pen += _clamp((30 - lu) / 30 * 6, 0, 6)
        if xc is not None and xc > 10:
            # log-scale because real-world values run 10..500+
            pen += _clamp((xc - 10) ** 0.5, 0, 8)
    # Wedge penalties.
    curve = _wedge_metrics(c)
    target_minus6 = 4.0
    f6 = _wedge_minus_n_db_freq(curve, -6.0)
    if f6 is None:
        pen += 25
    elif f6 < target_minus6:
        pen += _clamp((target_minus6 - f6) * 20, 0, 25)
    first_chroma = _wedge_first_chroma_freq(curve, rms_threshold=10.0)
    if first_chroma is not None and first_chroma <= 3.5:
        # Chroma appearing early in the wedge = bad Y/C separation.
        pen += _clamp((3.5 - first_chroma + 0.5) * 10, 0, 15)
    return max(0.0, 100.0 - pen)


_SYNTH_WEDGE_DATA_URL = None


def _synth_wedge_data_url(upscale: int = 4) -> str:
    """Render the wedge column from a clean synth once and return it as a
    base64 PNG data URL. Used as the visual reference at the top of the
    frequency wedge section."""
    global _SYNTH_WEDGE_DATA_URL
    if _SYNTH_WEDGE_DATA_URL is not None:
        return _SYNTH_WEDGE_DATA_URL
    try:
        import base64
        import cv2
        bgr = _synth_frame_bgr()
        w = tp_chart.WEDGE_COLUMN
        x0, y0 = w["x"], w["y_top"]
        x1 = x0 + w["width"]; y1 = w["y_bottom"]
        crop = bgr[y0:y1, x0:x1]
        if upscale > 1:
            crop = cv2.resize(
                crop, (crop.shape[1] * upscale, crop.shape[0] * upscale),
                interpolation=cv2.INTER_NEAREST,
            )
        ok, png = cv2.imencode(".png", crop)
        if not ok:
            return ""
        _SYNTH_WEDGE_DATA_URL = (
            "data:image/png;base64,"
            + base64.b64encode(png.tobytes()).decode("ascii")
        )
    except Exception:
        _SYNTH_WEDGE_DATA_URL = ""
    return _SYNTH_WEDGE_DATA_URL


def _capture_wedge_data_url(c: Dict[str, Any]) -> str:
    """Embed the per-capture wedge crop sidecar PNG that tp_measure
    writes alongside the JSON. Returns "" if the sidecar isn't there
    (e.g. older JSON)."""
    import base64
    src = c.get("_source_json_path")
    if not src:
        return ""
    stem = os.path.splitext(src)[0]
    path = stem + "_wedge.png"
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return ("data:image/png;base64,"
                + base64.b64encode(f.read()).decode("ascii"))


def _modulation_class(pct) -> str:
    """Color class for wedge modulation %. Stricter than the row-2 burst
    threshold (more sample points, finer gradation)."""
    if pct is None:
        return ""
    if pct >= 50:
        return "delta-good"
    if pct >= 20:
        return "delta-warn"
    return "delta-bad"


def render_frequency_wedge_overview(captures: List[Dict[str, Any]]) -> str:
    head = (
        "<tr>"
        + _sortable_th("Capture",
                       "Capture file name.", kind="text")
        + _sortable_th("Luma -6 dB (MHz)",
                       "Highest frequency in the wedge where luma "
                       "modulation is still ≥ 50 % of full contrast "
                       "(−6 dB cut-off). Higher = wider luma bandwidth.")
        + _sortable_th("Luma -12 dB (MHz)",
                       "Highest frequency where luma is still ≥ 25 % of "
                       "full contrast. A useful proxy for the practical "
                       "resolution limit before bars become noise.")
        + _sortable_th("First chroma intrusion (MHz)",
                       "Lowest wedge frequency where chroma_rms ≥ 10. "
                       "On a clean Y/C-separating decoder this is "
                       "absent. Lower = decoder starts faking chroma "
                       "from luma earlier in the wedge.")
        + _sortable_th("Max chroma leak",
                       "Largest chroma_rms across the 7 wedge sample "
                       "points.")
        + "</tr>"
    )
    rows = []
    for c in captures:
        name = _basename(c["_meta"]["capture"])
        curve = _wedge_metrics(c)
        f6  = _wedge_minus_n_db_freq(curve, -6.0)
        f12 = _wedge_minus_n_db_freq(curve, -12.0)
        first_chroma = _wedge_first_chroma_freq(curve)
        max_chroma   = _wedge_max_chroma_rms(curve)
        cells = [
            _td_name(name),
            _td_num(f6,  "{:.2f}",
                    cls=("delta-good" if f6  and f6  >= 4.5
                         else "delta-warn" if f6  and f6  >= 3.5
                         else "delta-bad")),
            _td_num(f12, "{:.2f}",
                    cls=("delta-good" if f12 and f12 >= 5.0
                         else "delta-warn" if f12 and f12 >= 4.0
                         else "delta-bad")),
            _td_num(first_chroma, "{:.2f}",
                    cls=("delta-bad" if first_chroma and first_chroma <= 2.5
                         else "delta-warn" if first_chroma and first_chroma <= 3.5
                         else "delta-good")),
            _td_num(max_chroma, "{:.1f}", cls=_chroma_class(max_chroma)),
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<section class="overview">
  <h2>Frequency Wedge Overview</h2>
  <p class="legend">
    Headline numbers from the right-column narrowing wedge (1.5 MHz at
    the top through 5.5 MHz at the bottom). The -6 dB cutoff tells you
    where the decoder loses half the bar contrast; the first chroma
    intrusion frequency tells you where Y/C separation starts to fail.
  </p>
  <table class="overview-table">
    <thead>{head}</thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def _render_frequency_wedge_panel(c: Dict[str, Any]) -> str:
    cap_name = _basename(c["_meta"]["capture"])
    curve = _wedge_metrics(c)
    rows_html = []
    for freq, lu_pct, lu_db, xc_rms in curve:
        mod_cell = ("—" if lu_pct is None
                    else f"{lu_pct:.1f}%"
                         + (f" ({lu_db:+.1f} dB)"
                            if lu_db is not None and lu_db != float("-inf")
                            else ""))
        chroma_cell = ("—" if xc_rms is None else f"{xc_rms:.1f}")
        rows_html.append(
            f"<tr>"
            f"<td class='name'>{freq:.1f} MHz</td>"
            f"<td class='delta {_modulation_class(lu_pct)}'>{mod_cell}</td>"
            f"<td class='delta {_chroma_class(xc_rms)}'>{chroma_cell}</td>"
            f"</tr>"
        )
    f6  = _wedge_minus_n_db_freq(curve, -6.0)
    f12 = _wedge_minus_n_db_freq(curve, -12.0)
    first_chroma = _wedge_first_chroma_freq(curve)
    summary_html = (
        "<ul class='geo-list'>"
        f"<li>Luma <b>-6 dB</b> cutoff: "
        f"<b>{('not reached' if f6 is None else f'{f6:.2f} MHz')}</b> "
        f"<span class='muted'>(higher = wider luma bandwidth)</span>.</li>"
        f"<li>Luma <b>-12 dB</b> cutoff: "
        f"<b>{('not reached' if f12 is None else f'{f12:.2f} MHz')}</b> "
        f"<span class='muted'>(point where bars become noise)</span>.</li>"
        f"<li>First chroma intrusion: "
        f"<b>{('none' if first_chroma is None else f'{first_chroma:.2f} MHz')}</b> "
        f"<span class='muted'>(clean Y/C decoders show none)</span>.</li>"
        "</ul>"
    )
    wedge_url = _capture_wedge_data_url(c)
    if wedge_url:
        wedge_img = (
            f"<figure class='wedge-fig'><img src='{wedge_url}' "
            f"alt='wedge crop'/>"
            f"<figcaption>Decoded wedge (decoder output, 4× upscaled).</figcaption>"
            f"</figure>"
        )
    else:
        wedge_img = ("<p class='muted small'>no wedge crop sidecar — "
                     "re-run tp_measure to generate.</p>")
    return (
        f"<div class='color-panel'>"
        f"<h3>{_h.escape(cap_name)}</h3>"
        f"<div class='wedge-row'>"
        f"<div class='wedge-img-col'>{wedge_img}</div>"
        f"<div class='wedge-table-col'>"
        f"<table class='color-table'>"
        f"<tr>{_th('Frequency', 'Local frequency at the sample y-position in the wedge.')}"
        f"{_th('Luma modulation', 'Peak modulation as a fraction of full contrast.')}"
        f"{_th('Chroma RMS', 'Chroma deviation from neutral inside the luma-only stripe pattern.')}"
        f"</tr>"
        + "".join(rows_html) +
        f"</table>"
        f"<h4>Summary</h4>{summary_html}"
        f"</div></div></div>"
    )


def render_frequency_wedge_section(captures: List[Dict[str, Any]]) -> str:
    if not any(c.get("frequency_response") for c in captures):
        return ""
    intro = """
<p class="legend">
  The right side of the chart contains a <b>continuous frequency
  wedge</b> running top-to-bottom from <b>1.5 MHz</b> down to
  <b>5.5 MHz</b>. The stripes are pure luma (black/white) and they
  narrow as you read down — each row's local frequency rises with
  position. This is the canonical way to <b>see</b> where a decoder
  hits its bandwidth limit.
</p>
<p class="legend">
  <b>What should happen:</b> on a clean signal path, the stripes stay
  crisp and high-contrast all the way down. The modulation amplitude
  rolls off gently (every analog system rolls off eventually), but you
  can still resolve individual bars at 5 MHz.
</p>
<p class="legend">
  <b>Failure modes to look for:</b>
  <ul class="legend">
    <li><b>Mid-grey takeover</b> — the modulation amplitude drops, so
    bars no longer reach pure black/white. The wedge looks washed out
    in its lower half. Quantified as the <i>−6 dB cutoff frequency</i>
    (where the modulation has lost half its contrast).</li>
    <li><b>Stripes disappear into noise</b> — past the decoder's
    bandwidth limit the bars degrade to flat grey. Quantified as the
    <i>−12 dB cutoff</i>.</li>
    <li><b>Color creep / cross-color</b> — bars that should be pure
    black-and-white acquire a color tint (often blue/yellow on
    composite NTSC decoders). The decoder is misclassifying
    high-frequency luma as chroma. Quantified by chroma RMS at each
    sample point and surfaced as the
    <i>first chroma-intrusion frequency</i>.</li>
    <li><b>Ringing / overshoot</b> — bright/dark fringes at stripe
    edges. Not a single number on this section yet, but visible in
    the decoded-wedge crop.</li>
  </ul>
</p>
"""
    synth_url = _synth_wedge_data_url(upscale=4)
    synth_block = (
        f"<figure class='wedge-fig wedge-synth-fig'>"
        f"<img src='{synth_url}' alt='synth wedge reference'/>"
        f"<figcaption>Synthetic reference: continuous wedge from "
        f"1.5 MHz (top) to 5.5 MHz (bottom), 4× upscaled.</figcaption>"
        f"</figure>" if synth_url else ""
    )
    panels = [_render_frequency_wedge_panel(c) for c in captures]
    return f"""
<section class="freq-wedge">
  <h2>Frequency Wedge — narrowing-stripe analysis</h2>
  {intro}
  {synth_block}
  {''.join(panels)}
</section>
"""


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


def _failure_chips(lu_pct, xc_rms) -> str:
    """Two-chip failure-mode summary per burst:
      - LUMA: pass / weak / lost
      - CHROMA: clean / leak / heavy
    """
    if lu_pct is None:
        luma_chip = ("<span class='clip-chip muted'>luma —</span>")
    elif lu_pct < 10:
        luma_chip = ("<span class='clip-chip delta-bad'>luma lost</span>")
    elif lu_pct < 30:
        luma_chip = ("<span class='clip-chip delta-warn'>luma weak</span>")
    else:
        luma_chip = ("<span class='clip-chip delta-good'>luma pass</span>")
    if xc_rms is None:
        chroma_chip = ("<span class='clip-chip muted'>chroma —</span>")
    elif xc_rms < 10:
        chroma_chip = ("<span class='clip-chip delta-good'>Y/C clean</span>")
    elif xc_rms < 100:
        chroma_chip = ("<span class='clip-chip delta-warn'>chroma leak</span>")
    else:
        chroma_chip = ("<span class='clip-chip delta-bad'>heavy cross-color</span>")
    return f"<div class='clip-chips'>{luma_chip}{chroma_chip}</div>"


def _render_freq_burst_panel(c: Dict[str, Any]) -> str:
    cap_name = _basename(c["_meta"]["capture"])
    rows_html = []
    for b in FREQ_BURST_REGIONS:
        m = _freq_burst_metrics(c, b["id"], b["xc_id"])
        lu_pct = m["luma_modulation_pct"]
        lu_db  = m["luma_modulation_db"]
        xc_rms = m["chroma_leak_rms"]
        chips = _failure_chips(lu_pct, xc_rms)
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
            f"<td>{chips}</td>"
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
        f"{_th('Failure modes', 'Per-burst pass/fail tags for the two distinct failure modes on these patterns: luma loss (decoder cuts high-frequency luma) and chroma leak (decoder injects chroma where there should be none).')}"
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
    <b>What it shows:</b> A 4-step neutral grayscale ramp at 20 %, 40 %,
    60 %, and 80 % IRE in the upper-left composite region of the
    chart. Each step is a small uniform patch carrying only luminance
    — no chroma should be present.
  </p>
  <p class="legend">
    <b>What it measures:</b> Two things at once. First, the decoder's
    <i>tonal accuracy</i> — does the measured Y10 land on the
    chart-spec value at each step (linearity of the luma transfer
    function and correct black/white setup)? Second, the decoder's
    <i>chroma neutrality</i> on luma-only content — any non-zero U
    or V offset on a gray patch betrays a tint introduced by the
    decoder.
  </p>
  <p class="legend">
    <b>Why TV engineers and viewers care:</b> Grayscale errors look
    like the picture is too dark, too bright, washed out, or
    crushed. Lifted blacks reduce contrast; non-linear gray steps
    distort gamma and produce wrong skin tones. A chroma cast on
    grays is the smell test for sloppy comb-decoder design: it
    leaks chroma into what should be neutral.
  </p>
  <p class="legend">
    <b>How we characterize each capture:</b> One row per step with
    ideal-vs-measured swatches side-by-side, the ideal and
    measured Y10 codes, the signed ΔY, the U/V offset from neutral
    (chroma cast), and a plain-language verdict.
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
    border: 1px solid #2a2e36; background: #000; width: 720px; height: 540px;
    max-width: 100%; }
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
.wedge-row { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
.wedge-img-col { flex: 0 0 auto; }
.wedge-table-col { flex: 1 1 320px; min-width: 320px; }
.wedge-fig { margin: 6px 0; text-align: center; }
.wedge-fig img { display: block; border: 1px solid #2a2e36; image-rendering: pixelated;
    background: #14161a; max-height: 540px; }
.wedge-fig figcaption { font-size: 11px; color: #b8c0cc; margin-top: 4px; max-width: 280px; }
.wedge-synth-fig { display: inline-block; margin: 0 12px 12px 0; }
.pulse-table img.pulse-spark { display: block; border: 1px solid #2a2e36;
    image-rendering: pixelated; background: #14161a; height: 80px; width: 240px; }
.pulse-table td.swatch-cell { width: 244px; }
.radial-wedge-fig { margin: 6px 12px 12px 0; text-align: center; display: inline-block; }
.radial-wedge-fig img { display: block; border: 1px solid #2a2e36;
    image-rendering: pixelated; background: #14161a;
    max-width: 480px; max-height: 480px; }
.legend-key { margin: 8px 0; font-size: 12px; color: #b8c0cc;
    background: #1d2026; padding: 6px 12px; border: 1px solid #2a2e36;
    border-radius: 4px; }
.legend-key summary { cursor: pointer; color: #c5d1e0; font-weight: 600; }
dl.metric-key { margin: 8px 0 0 0; line-height: 1.5; }
dl.metric-key dt { color: #f0b450; font-weight: 600; margin-top: 8px; }
dl.metric-key dd { margin: 2px 0 0 16px; color: #c5d1e0; }
.radial-wedge-fig figcaption { font-size: 11px; color: #b8c0cc;
    margin-top: 4px; max-width: 480px; }
section.visual-overview { margin: 16px 0 8px; }
section.visual-overview h2 { margin-bottom: 4px; }
section.visual-overview p.legend { margin-top: 4px; }
table.visual-color-summary { border-collapse: collapse; margin-top: 8px;
    font-size: 12px; color: #c5d1e0; }
table.visual-color-summary th, table.visual-color-summary td {
    border: none; padding: 3px 4px; vertical-align: middle; }
table.visual-color-summary th.visrow-cap { text-align: left;
    color: #c5d1e0; font-weight: 600; min-width: 220px; }
table.visual-color-summary th.visrow-color { font-weight: 500;
    color: #9aa5b4; text-align: center; padding: 2px 6px; }
table.visual-color-summary td.visrow-cap { color: #c5d1e0; font-weight: 500;
    white-space: nowrap; padding-right: 8px; }
table.visual-color-summary td.visrow-pair { padding: 0; line-height: 0; }
table.visual-color-summary td.visrow-pair .swatch {
    display: inline-block; width: 18px; height: 22px; margin: 0;
    padding: 0; border: 0; vertical-align: middle; }
table.visual-color-summary .visrow-refswatch {
    display: block; width: 18px; height: 6px; margin: 2px auto 0;
    border-radius: 1px; }
table.visual-color-summary tbody tr:nth-child(odd) { background: #181b21; }
table.visual-frequency-summary { border-collapse: collapse; margin-top: 8px;
    font-size: 12px; color: #c5d1e0; }
table.visual-frequency-summary th, table.visual-frequency-summary td {
    border: 1px solid #2a2e36; padding: 4px; vertical-align: middle;
    text-align: center; background: #14161a; }
table.visual-frequency-summary th.visfreq-cap { text-align: left;
    min-width: 220px; }
table.visual-frequency-summary td.visfreq-cap { text-align: left;
    color: #c5d1e0; font-weight: 500; min-width: 220px;
    background: #1a1d24; white-space: nowrap; }
table.visual-frequency-summary td.visfreq-ref { color: #f0b450;
    font-weight: 600; }
table.visual-frequency-summary tr.visfreq-refrow td { background: #1d2026; }
table.visual-frequency-summary td.visfreq-img { padding: 2px; }
table.visual-frequency-summary td.visfreq-img img {
    display: block; image-rendering: pixelated; max-width: 160px;
    max-height: 160px; margin: 0 auto; background: #14161a; }
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
    intro = """
<p class="legend">
  <b>What it shows:</b> Four arrowhead fiducials mark the corners of
  the active picture (TL / TR / BL / BR); a small black-on-white
  registration cross sits in the upper-right; and a large black ring
  circumscribes the picture height. Every other measurement in the
  report relies on locating these features accurately.
</p>
<p class="legend">
  <b>What it measures:</b> The picture's position, size, and shape
  inside the 720 × 486 NTSC SDI raster. Specifically: arrowhead
  spacing along each edge (horizontal vs the chart spec of 357 px,
  vertical vs 483 px), picture-center displacement, horizontal /
  vertical scale percentages, keystone (top-vs-bottom width
  difference), and a PAR-aware circularity check on the black ring
  (NTSC 10:11 pixel aspect ratio means a circle in display space is
  an ellipse in the raster — a perfectly-round display reads
  circularity = 1.00).
</p>
<p class="legend">
  <b>Why TV engineers and viewers care:</b> Geometry errors are the
  most visually-disturbing kind. A picture shifted or scaled wrong
  means content runs off-screen or sits in the wrong spot;
  keystoning makes straight edges curve; an aspect ratio error
  makes circles look like ovals and faces look squished or
  stretched. Engineers also use the geometry block to validate
  registration — if the chart's chart-spec features are mis-
  located, the rest of the measurements are sampling the wrong
  pixels.
</p>
<p class="legend">
  <b>How we characterize each capture:</b> An arrowhead spacing
  table (top / bottom / left / right vs chart spec), a picture-
  displacement summary (center shift, H/V scale %, keystone),
  per-corner clip detection (apex visible vs clipped), the
  registration cross's center offset and aperture symmetry, and
  the PAR-aware circle's horizontal / vertical diameter plus the
  displayed-circularity ratio.
</p>
"""
    return f"""
<section class="geometry">
  <h2>Geometry</h2>
  {intro}
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
    # Canonical section order (used in three places: the Overall
    # Summary columns above, the per-section Overview tables here, and
    # the per-capture detail panels below). Logical grouping:
    #   foundation: Geometry
    #   color:      Tartan → Chroma Linearity → Grayscale
    #   sharpness:  Pulse → Frequency (row-2) → Frequency Wedge → VertRes → Radial Wedge
    #   chroma Y/C: Y/C Timing → Zone Plate
    overviews = (
        render_overall_summary(captures)
        + render_visual_color_summary(captures)
        + render_visual_frequency_summary(captures)
        + render_geometry_overview(captures)
        + render_tartan_overview(captures)
        + render_chroma_staircase_overview(captures)
        + render_grayscale_overview(captures)
        + render_pulse_overview(captures)
        + render_frequency_response_overview(captures)
        + render_frequency_wedge_overview(captures)
        + render_vertical_response_overview(captures)
        + render_radial_wedge_overview(captures)
        + render_yc_timing_overview(captures)
        + render_zone_plate_overview(captures)
    )
    details = (
        render_geometry_section(captures)
        + render_tartan_panels(captures)
        + render_chroma_staircase_panels(captures)
        + render_gray_panels(captures)
        + render_pulse_panels(captures)
        + render_frequency_response_section(captures)
        + render_frequency_wedge_section(captures)
        + render_vertical_response_panels(captures)
        + render_radial_wedge_panels(captures)
        + render_yc_timing_panels(captures)
        + render_zone_plate_panels(captures)
        + render_luma_scale_analysis(captures)
        + render_artifacts(captures)
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
        + render_radial_wedge(captures)
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
