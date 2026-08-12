# tests/test_plots.py
from pathlib import Path
from miarag.plots import plot_roc

def test_plot_roc_creates_file(tmp_path):
    out = tmp_path / "roc.png"
    p = plot_roc({"S2MIA": ([0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1])}, out)
    assert p.exists() and p.stat().st_size > 0
