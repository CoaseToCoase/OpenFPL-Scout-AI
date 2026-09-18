from pathlib import Path

from experiments.run_arms import ARTIFACT_ROOT, build_preprocessor


def test_preprocessor_uses_the_arms_own_feature_lists():
    pre = build_preprocessor(["element_type"], ["gameweek"])
    names = [name for name, _, _ in pre.transformers]
    assert names == ["categorical", "numerical"]
    assert pre.transformers[0][2] == ["element_type"]
    assert pre.transformers[1][2] == ["gameweek"]


def test_artifacts_never_land_in_the_production_models_dir():
    """models/*.pkl is loaded by the nightly production path."""
    assert ARTIFACT_ROOT == Path("models") / "experiments"
    assert ARTIFACT_ROOT != Path("models")
