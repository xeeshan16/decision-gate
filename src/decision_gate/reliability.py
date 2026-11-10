"""Reliability diagrams as standalone SVG, with no plotting dependency.

A reliability diagram puts mean confidence on the x axis and observed accuracy on the y
axis, one point per confidence bin. A perfectly calibrated model sits on the diagonal;
points below it are overconfident. Under the plot, a strip shows how many decisions fell in
each bin, because a far-off point backed by five decisions means little.

Several series (raw confidence and calibrated confidence, say) share one plot. The output
is deterministic for the same input, so reports can be committed and diffed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from xml.sax.saxutils import escape

from decision_gate.metrics import (
    BinStrategy,
    Confidences,
    Correctness,
    ReliabilityBin,
    calibration_metrics,
)

# Colour-blind safe pair (red-orange and blue-teal), then two more for extra series.
PALETTE = ("#d1495b", "#00798c", "#edae49", "#30638e")

WIDTH = 480
PLOT = 360  # the square plot's side
LEFT, TOP = 64, 64
STRIP = 72  # height of the count strip under the plot
GAP = 48
HEIGHT = TOP + PLOT + GAP + STRIP + 56


@dataclass(frozen=True, slots=True)
class Series:
    label: str
    bins: Sequence[ReliabilityBin]
    ece: float | None = None
    count: int | None = None


def series_from_pairs(
    label: str,
    confidences: Confidences,
    correct: Correctness,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
) -> Series:
    """Bin ``(confidence, correct)`` pairs into a series, with its ECE and count."""
    m = calibration_metrics(confidences, correct, n_bins=n_bins, strategy=strategy)
    return Series(label, m.bins, m.ece, m.count)


def _x(value: float) -> float:
    return LEFT + value * PLOT


def _y(value: float) -> float:
    return TOP + (1.0 - value) * PLOT


def _f(value: float) -> str:
    return f"{value:.1f}"


def render_reliability_svg(
    title: str, series: Sequence[Series], *, subtitle: str | None = None
) -> str:
    """One SVG document: the diagram, a legend with each series' ECE, and a count strip.

    The count strip shows the first series' bins (every series of one diagram is meant
    to cover the same decisions, so their counts per bin differ only by where the
    calibrated confidences moved them).
    """
    if not series:
        raise ValueError("a reliability diagram needs at least one series")
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="12" role="img" aria-label="{escape(title, {chr(34): "&quot;"})}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        f'<text x="{LEFT}" y="24" font-size="15" font-weight="bold" fill="#1f2328">'
        f"{escape(title)}</text>",
    ]
    if subtitle:
        out.append(f'<text x="{LEFT}" y="42" fill="#59636e">{escape(subtitle)}</text>')

    # Grid, axes and ticks.
    for i in range(11):
        v = i / 10
        stroke = "#d0d7de" if i % 5 else "#afb8c1"
        out.append(
            f'<line x1="{_f(_x(v))}" y1="{TOP}" x2="{_f(_x(v))}" y2="{TOP + PLOT}" '
            f'stroke="{stroke}" stroke-width="0.5"/>'
        )
        out.append(
            f'<line x1="{LEFT}" y1="{_f(_y(v))}" x2="{LEFT + PLOT}" y2="{_f(_y(v))}" '
            f'stroke="{stroke}" stroke-width="0.5"/>'
        )
        if i % 2 == 0:
            out.append(
                f'<text x="{_f(_x(v))}" y="{TOP + PLOT + 16}" text-anchor="middle" '
                f'fill="#59636e">{v:.1f}</text>'
            )
            out.append(
                f'<text x="{LEFT - 8}" y="{_f(_y(v) + 4)}" text-anchor="end" '
                f'fill="#59636e">{v:.1f}</text>'
            )
    out.append(
        f'<rect x="{LEFT}" y="{TOP}" width="{PLOT}" height="{PLOT}" fill="none" '
        f'stroke="#59636e" stroke-width="1"/>'
    )
    out.append(
        f'<line x1="{_x(0)}" y1="{_y(0)}" x2="{_x(1)}" y2="{_y(1)}" stroke="#8c959f" '
        f'stroke-width="1" stroke-dasharray="4 4"/>'
    )
    out.append(
        f'<text x="{LEFT + PLOT / 2}" y="{TOP + PLOT + 34}" text-anchor="middle" '
        f'fill="#1f2328">mean confidence in bin</text>'
    )
    out.append(
        f'<text transform="translate(18 {TOP + PLOT / 2}) rotate(-90)" text-anchor="middle" '
        f'fill="#1f2328">observed accuracy</text>'
    )

    # Series: a line through the non-empty bins, a dot per bin.
    for index, s in enumerate(series):
        colour = PALETTE[index % len(PALETTE)]
        points = [
            (b.mean_confidence, b.accuracy)
            for b in s.bins
            if b.count and b.mean_confidence is not None and b.accuracy is not None
        ]
        if len(points) > 1:
            path = " ".join(f"{_f(_x(c))},{_f(_y(a))}" for c, a in points)
            out.append(
                f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="2"/>'
            )
        for c, a in points:
            out.append(f'<circle cx="{_f(_x(c))}" cy="{_f(_y(a))}" r="3.5" fill="{colour}"/>')
        # Legend, inside the plot's upper-left corner (the diagonal's empty side).
        ly = TOP + 18 + index * 18
        label = s.label
        if s.ece is not None:
            label += f"  (ECE {s.ece:.3f}"
            label += f", n={s.count})" if s.count is not None else ")"
        out.append(f'<rect x="{LEFT + 10}" y="{ly - 8}" width="14" height="4" fill="{colour}"/>')
        out.append(f'<text x="{LEFT + 30}" y="{ly - 2}" fill="#1f2328">{escape(label)}</text>')

    # Count strip for the first series.
    strip_top = TOP + PLOT + GAP + 20
    bins = list(series[0].bins)
    peak = max((b.count for b in bins), default=0)
    out.append(
        f'<text x="{LEFT}" y="{strip_top - 6}" fill="#59636e">'
        f"decisions per bin ({escape(series[0].label)})</text>"
    )
    for b in bins:
        if not b.count or not peak:
            continue
        h = STRIP * b.count / peak
        x0, x1 = _x(b.lower), _x(b.upper)
        out.append(
            f'<rect x="{_f(x0 + 0.5)}" y="{_f(strip_top + STRIP - h)}" '
            f'width="{_f(max(x1 - x0 - 1, 0.5))}" height="{_f(h)}" fill="#8c959f">'
            f"<title>{b.count} decisions in ({b.lower:.2f}, {b.upper:.2f}]</title></rect>"
        )
    out.append(
        f'<line x1="{LEFT}" y1="{strip_top + STRIP}" x2="{LEFT + PLOT}" '
        f'y2="{strip_top + STRIP}" stroke="#59636e" stroke-width="1"/>'
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"
