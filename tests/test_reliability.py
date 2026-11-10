"""Reliability diagrams: the SVG renderer, `dgate calibration diagram` and simulator diagrams."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from decision_gate.cli import app as cli
from decision_gate.questions import QuestionCatalog
from decision_gate.reliability import (
    HEIGHT,
    LEFT,
    PLOT,
    TOP,
    Series,
    render_reliability_svg,
    series_from_pairs,
)
from decision_gate.simulator.engine import SimulationConfig, simulate
from decision_gate.simulator.report import phase_diagrams
from tests.test_refit import run, seed_fake

runner = CliRunner()
NS = "{http://www.w3.org/2000/svg}"


def _parse(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def test_series_from_pairs_bins_and_scores() -> None:
    s = series_from_pairs("raw", [0.95, 0.95, 0.15, 0.15], [True, False, False, False], n_bins=10)
    assert s.count == 4
    assert s.ece == pytest.approx((2 * abs(0.95 - 0.5) + 2 * 0.15) / 4)
    assert sum(b.count for b in s.bins) == 4
    assert len(s.bins) == 10


def test_svg_is_well_formed_and_plots_each_non_empty_bin() -> None:
    rng = np.random.default_rng(0)
    c = rng.uniform(0.5, 1.0, 400)
    y = rng.uniform(size=400) < c
    raw = series_from_pairs("raw confidence", c, y)
    calibrated = series_from_pairs("calibrated", np.clip(c - 0.05, 0, 1), y)

    svg = render_reliability_svg("q@v1", [raw, calibrated], subtitle="400 decisions")
    root = _parse(svg)

    assert root.tag == f"{NS}svg"
    assert root.get("height") == str(HEIGHT)
    lines = root.findall(f"{NS}polyline")
    assert len(lines) == 2
    non_empty = sum(1 for b in raw.bins if b.count)
    assert len(lines[0].get("points", "").split()) == non_empty
    # Every point sits inside the plot square.
    for circle in root.findall(f"{NS}circle"):
        assert LEFT <= float(circle.get("cx", "0")) <= LEFT + PLOT
        assert TOP <= float(circle.get("cy", "0")) <= TOP + PLOT
    texts = [t.text or "" for t in root.iter(f"{NS}text")]
    assert "400 decisions" in texts
    assert any(t.startswith("raw confidence  (ECE ") and t.endswith("n=400)") for t in texts)
    # The count strip has one bar per non-empty bin of the first series.
    bars = [r for r in root.findall(f"{NS}rect") if r.find(f"{NS}title") is not None]
    assert len(bars) == non_empty
    # Deterministic: the same input renders byte for byte the same.
    assert svg == render_reliability_svg("q@v1", [raw, calibrated], subtitle="400 decisions")


def test_svg_escapes_text_and_handles_sparse_series() -> None:
    lone = series_from_pairs("<only> & one", [0.7], [True], n_bins=5)
    bare = Series("no ece", lone.bins)

    root = _parse(render_reliability_svg('a "quoted" <title>', [lone, bare]))

    assert root.get("aria-label") == 'a "quoted" <title>'
    texts = [t.text or "" for t in root.iter(f"{NS}text")]
    assert 'a "quoted" <title>' in texts
    assert "no ece" in texts
    assert root.findall(f"{NS}polyline") == []  # one point: a dot, no line
    assert len(root.findall(f"{NS}circle")) == 2


def test_empty_series_has_no_bars() -> None:
    empty = Series("empty", series_from_pairs("x", [0.5], [True], n_bins=4).bins[:0], 0.0, 0)
    root = _parse(render_reliability_svg("empty", [empty]))
    assert [r for r in root.findall(f"{NS}rect") if r.find(f"{NS}title") is not None] == []


def test_a_diagram_needs_a_series() -> None:
    with pytest.raises(ValueError, match="at least one series"):
        render_reliability_svg("nothing", [])


def test_calibration_diagram_draws_raw_and_calibrated(
    sqlite_url: str, example_catalog: QuestionCatalog, tmp_path: Path
) -> None:
    seed_fake(sqlite_url, example_catalog, "refund.eligible", 400)
    seed_fake(sqlite_url, example_catalog, "ticket.queue", 30)
    refit = runner.invoke(
        cli, ["calibrators", "refit", "refund.eligible", "--version", "1", "--url", sqlite_url]
    )
    assert refit.exit_code == 0, refit.output
    out = tmp_path / "diagrams"

    result = runner.invoke(cli, ["calibration", "diagram", str(out), "--url", sqlite_url])

    assert result.exit_code == 0, result.output
    # ticket.queue has 30 labels, below the default --min-count of 50.
    assert sorted(p.name for p in out.iterdir()) == ["refund.eligible@v1.svg"]
    root = _parse((out / "refund.eligible@v1.svg").read_text())
    assert len(root.findall(f"{NS}polyline")) == 2
    labels = [t.text or "" for t in root.iter(f"{NS}text")]
    assert any(t.startswith("calibrated (v1 ") for t in labels)
    assert "raw confidence ECE" in result.output
    assert "calibrated (v1" in result.output


def test_calibration_diagram_without_a_calibrator_or_labels(
    sqlite_url: str, example_catalog: QuestionCatalog, tmp_path: Path
) -> None:
    out = tmp_path / "d"
    empty = runner.invoke(cli, ["calibration", "diagram", str(out), "--url", sqlite_url])
    assert empty.exit_code == 0
    assert "no question version has enough labeled decisions" in empty.output
    assert not out.exists()

    seed_fake(sqlite_url, example_catalog, "ticket.queue", 60)
    result = runner.invoke(
        cli,
        [
            "calibration",
            "diagram",
            str(out),
            "--question",
            "ticket.queue",
            "--bins",
            "5",
            "--url",
            sqlite_url,
        ],
    )
    assert result.exit_code == 0, result.output
    root = _parse((out / "ticket.queue@v1.svg").read_text())
    assert len(root.findall(f"{NS}polyline")) == 1  # raw only


def test_simulation_diagrams_cover_each_phase(tmp_path: Path) -> None:
    config = SimulationConfig(
        datasets=("refund",),
        per_day=60,
        history_days=4,
        steady_days=2,
        drift_days=1,
        refit_every_days=0,
    )
    result = run(simulate(config))

    diagrams = phase_diagrams(result)

    # The one drift day has 60 decisions; every phase has at least 50.
    assert sorted(diagrams) == [
        "refund.eligible-drift.svg",
        "refund.eligible-history.svg",
        "refund.eligible-steady.svg",
    ]
    history = _parse(diagrams["refund.eligible-history.svg"])
    steady = _parse(diagrams["refund.eligible-steady.svg"])

    # No calibrator during history, so only the raw series.
    def labels(root: ET.Element) -> list[str]:
        return [t.text or "" for t in root.iter(f"{NS}text") if "ECE" in (t.text or "")]

    assert len(labels(history)) == 1
    assert [t.split("  ")[0] for t in labels(steady)] == [
        "raw confidence",
        "calibrated (active calibrator)",
    ]

    sparse = phase_diagrams(
        run(
            simulate(
                SimulationConfig(
                    datasets=("refund",), per_day=20, history_days=2, steady_days=1, drift_days=0
                )
            )
        )
    )
    assert sparse == {}  # fewer than 50 decisions in every phase
