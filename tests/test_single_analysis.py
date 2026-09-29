import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dxa_qc.analysis import analyze_single_run


def make_inputs(root, misalign=False):
    path = root / "A3"
    path.mkdir()
    y = np.tile([0., 1.], (10, 1)).T
    for seed in (17, 29, 43):
        groups = np.array(["g0", "g1"])
        if misalign and seed == 43:
            groups = groups[::-1]
        np.savez(path / f"predictions_seed_{seed}.npz",
                 y=y, p=.1 + .8*y, thresholds=np.full_like(y, .5), groups=groups)


def test_single_ablation_does_not_require_other_runs(tmp_path):
    make_inputs(tmp_path)
    analyze_single_run(tmp_path, "A3")
    report = json.loads((tmp_path / "seed_mean_std.json").read_text())
    assert report["A3"]["competition_violation_macro_f1"]["mean"] == 1.
    assert not (tmp_path / "A0").exists()
    assert not (tmp_path / "completion.json").exists()


def test_single_ablation_rejects_unaligned_seeds(tmp_path):
    make_inputs(tmp_path, misalign=True)
    with pytest.raises(ValueError, match="alignment"):
        analyze_single_run(tmp_path, "A3")
