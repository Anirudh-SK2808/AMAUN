"""M6.8 -- normalisation provenance (criterion E4, idea.md S4.7 / S16.5)."""

import pytest

from amaun.errors import NormalizerLeakageError
from amaun.features.base import FeatureVector
from amaun.features.normalizer import FeatureNormalizer


def _vector(values, names=None):
    names = names or [f"f{i}" for i in range(len(values))]
    return FeatureVector(names=names, values=list(values))


def test_fitting_on_test_is_refused():
    normalizer = FeatureNormalizer()
    with pytest.raises(NormalizerLeakageError, match="test"):
        normalizer.fit([_vector([1.0, 2.0])], split="test", run_mode="dummy")


def test_unknown_split_is_refused():
    with pytest.raises(NormalizerLeakageError):
        FeatureNormalizer().fit([_vector([1.0])], split="mystery", run_mode="dummy")


def test_transform_across_run_modes_is_refused(tmp_path):
    """idea.md S4.7: statistics must never be shared between backbone tiers."""
    normalizer = FeatureNormalizer()
    normalizer.fit([_vector([1.0, 2.0]), _vector([3.0, 4.0])], split="utility_train", run_mode="dummy")
    normalizer.transform(_vector([1.0, 2.0]), run_mode="dummy")
    with pytest.raises(NormalizerLeakageError, match="run_mode"):
        normalizer.transform(_vector([1.0, 2.0]), run_mode="actual")


def test_loading_across_run_modes_is_refused(tmp_path):
    normalizer = FeatureNormalizer()
    normalizer.fit([_vector([1.0]), _vector([2.0])], split="utility_train", run_mode="local_smoke")
    path = normalizer.save(tmp_path / "n.json")
    FeatureNormalizer.load(path, expected_run_mode="local_smoke")
    with pytest.raises(NormalizerLeakageError):
        FeatureNormalizer.load(path, expected_run_mode="actual")


def test_feature_order_mismatch_is_refused():
    """A silently permuted feature vector would corrupt every prediction invisibly."""
    normalizer = FeatureNormalizer()
    normalizer.fit(
        [_vector([1.0, 2.0], ["a", "b"]), _vector([3.0, 4.0], ["a", "b"])],
        split="utility_train", run_mode="dummy",
    )
    with pytest.raises(NormalizerLeakageError, match="mismatch"):
        normalizer.transform(_vector([1.0, 2.0], ["b", "a"]), run_mode="dummy")
    with pytest.raises(NormalizerLeakageError):
        normalizer.transform(_vector([1.0, 2.0, 3.0], ["a", "b", "c"]), run_mode="dummy")


def test_inconsistent_fit_vectors_are_refused():
    with pytest.raises(NormalizerLeakageError):
        FeatureNormalizer().fit(
            [_vector([1.0], ["a"]), _vector([2.0], ["b"])], split="utility_train", run_mode="dummy"
        )


def test_standardisation_is_correct():
    normalizer = FeatureNormalizer(epsilon=0.0)
    normalizer.fit(
        [_vector([0.0]), _vector([2.0]), _vector([4.0])], split="utility_train", run_mode="dummy"
    )
    assert normalizer.mean == pytest.approx([2.0])
    assert normalizer.std == pytest.approx([1.632993], abs=1e-5)   # population std
    assert normalizer.transform(_vector([2.0]), run_mode="dummy")[0] == pytest.approx(0.0)


def test_provenance_is_persisted(tmp_path):
    normalizer = FeatureNormalizer()
    normalizer.fit([_vector([1.0]), _vector([2.0])], split="utility_train", run_mode="dummy")
    reloaded = FeatureNormalizer.load(normalizer.save(tmp_path / "n.json"), expected_run_mode="dummy")
    assert reloaded.fitted_on_split == "utility_train"
    assert reloaded.fitted_run_mode == "dummy"
    assert reloaded.fitted_n_samples == 2
    assert reloaded.signature == normalizer.signature


def test_unfitted_normalizer_refuses_to_transform():
    with pytest.raises(NormalizerLeakageError):
        FeatureNormalizer().transform(_vector([1.0]), run_mode="dummy")
