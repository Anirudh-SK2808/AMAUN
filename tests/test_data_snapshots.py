"""M2 -- the content-keyed offline dataset cache (FR-5, AC-15).  Offline by construction.

The two claims that matter: a snapshot can never be reused under *different* coordinates (the
key hashes all nine content-determining inputs), and a matching snapshot is served with **no
network call at all** -- proved by monkeypatching ``datasets.load_dataset`` to raise, not by
inspecting a log line.
"""

from __future__ import annotations

import json

import pytest

from amaun.data import snapshots
from amaun.data.providers import huggingface_pool
from amaun.data.registry import task_spec_from_config
from amaun.data.row_adapters import resolve
from amaun.errors import DataError

BASE_SPEC_KWARGS = dict(
    hf_id="fixture/dataset",
    hf_revision="a" * 40,
    hf_config="default",
    split="train",
    size=10,
    skip=0,
    sampling="shuffled_buffer",
    shuffle_buffer_size=64,
    row_adapter="passthrough",
    row_adapter_options={"list_target_pick": "first"},
    required_fields=("article", "highlights"),
)


def _snapshot_spec(**overrides):
    return snapshots.SnapshotSpec(**{**BASE_SPEC_KWARGS, **overrides})


@pytest.mark.parametrize(
    "field,value",
    [
        ("hf_id", "other/dataset"),
        ("hf_revision", "b" * 40),
        ("hf_config", "plain_text"),
        ("split", "validation"),
        ("size", 11),
        ("skip", 5),
        ("sampling", "head_of_stream"),
        ("row_adapter", "finqa_text_only"),
        ("row_adapter_options", {"list_target_pick": "highest_score"}),
    ],
)
def test_key_changes_when_any_content_determining_input_changes(field, value):
    """FR-5.2: a stale snapshot may never be silently reused under new coordinates."""
    assert _snapshot_spec().key() != _snapshot_spec(**{field: value}).key()


def test_key_is_stable_across_calls_and_orderings():
    a = _snapshot_spec(row_adapter_options={"list_target_pick": "first", "z": 1})
    b = _snapshot_spec(row_adapter_options={"z": 1, "list_target_pick": "first"})
    assert a.key() == b.key() == _snapshot_spec(
        row_adapter_options={"list_target_pick": "first", "z": 1}
    ).key()


def test_buffer_size_is_part_of_the_key():
    """D-016 condition 1: the buffer size determines how near-random the sample is."""
    assert _snapshot_spec().key() != _snapshot_spec(shuffle_buffer_size=1024).key()


def test_round_trip_preserves_rows_exactly(tmp_path):
    spec = _snapshot_spec()
    rows = [
        {"article": "first article", "highlights": "first summary"},
        {"article": "second article éà", "highlights": "second summary"},
    ]
    path = snapshots.path_for(tmp_path, "sum", spec)
    snapshots.write(path, rows, spec=spec, task_id="sum", license="Apache-2.0", exhausted=False)
    loaded, manifest = snapshots.read(path, spec)
    assert loaded == rows
    assert manifest.rows_written == 2
    assert manifest.license == "Apache-2.0"
    assert manifest.exhausted is False


def test_nested_non_string_option_keys_still_hit_the_cache(tmp_path):
    """IMDb's ``labels: {0: negative, 1: positive}`` became {"0": ...} in the stored manifest,
    which made every such snapshot a permanent miss (and an offline run impossible)."""
    spec = _snapshot_spec(row_adapter="label_to_text",
                          row_adapter_options={"labels": {0: "negative", 1: "positive"}})
    rows = [{"text": "fine", "label": 1}]
    path = snapshots.path_for(tmp_path, "sentiment", spec)
    snapshots.write(path, rows, spec=spec, task_id="sentiment", license="other", exhausted=False)
    hit = snapshots.read(path, spec)
    assert hit is not None and hit[0] == rows


def test_missing_snapshot_is_a_miss(tmp_path):
    assert snapshots.read(snapshots.path_for(tmp_path, "sum", _snapshot_spec()), _snapshot_spec()) is None


def test_mismatched_manifest_is_a_miss_not_a_silent_reuse(tmp_path):
    """EC-3: a changed spec yields a new key; the old file is left untouched."""
    spec = _snapshot_spec()
    path = snapshots.path_for(tmp_path, "sum", spec)
    snapshots.write(path, [{"article": "a", "highlights": "b"}], spec=spec, task_id="sum",
                    license="Apache-2.0", exhausted=False)
    # Corrupt the manifest's recorded spec: same file, different declared coordinates.
    manifest_file = snapshots.manifest_path_for(path)
    payload = json.loads(manifest_file.read_text())
    payload["spec"]["skip"] = 999
    manifest_file.write_text(json.dumps(payload))
    assert snapshots.read(path, spec) is None
    assert path.exists()                       # append-only cache: nothing is deleted


def test_checksum_mismatch_raises_rather_than_serving_corrupt_rows(tmp_path):
    spec = _snapshot_spec()
    path = snapshots.path_for(tmp_path, "sum", spec)
    snapshots.write(path, [{"article": "a", "highlights": "b"}], spec=spec, task_id="sum",
                    license="Apache-2.0", exhausted=False)
    path.write_text(json.dumps({"article": "tampered", "highlights": "b"}) + "\n")
    with pytest.raises(DataError, match="checksum"):
        snapshots.read(path, spec)


def test_rows_are_restricted_to_the_adapters_required_fields():
    """FR-5.3: raw rows, but only the fields the adapter declared it reads."""
    row = {"article": "a", "highlights": "b", "id": "42", "extra": "ignored"}
    assert snapshots.restrict(row, ("article", "highlights")) == {"article": "a", "highlights": "b"}


# --------------------------------------------------------------------------------------
# FR-5.4 -- a matching snapshot is served with no network call
# --------------------------------------------------------------------------------------


SUM_SPEC = task_spec_from_config(
    "sum",
    {
        "task_name": "Summarization", "task_type": "summarization", "primary_score": "rougeL",
        "metrics": ["rougeL"], "input_fields": ["article"], "target_field": "highlights",
        "prompt_template": "Summarize:\n{article}\n\nSummary:", "max_input_tokens": 64,
        "max_target_tokens": 16, "hf_id": "fixture/dataset", "hf_revision": "a" * 40,
        "hf_config": "default", "row_adapter": "passthrough",
        "row_adapter_options": {"list_target_pick": "first"},
    },
)


def _fake_stream(rows):
    class _Iterable:
        def __init__(self, items):
            self._items = list(items)

        def shuffle(self, seed, buffer_size):        # order-preserving fake: determinism is
            return _Iterable(self._items)            # asserted separately in test_single_split_carving

        def skip(self, n):
            return _Iterable(self._items[n:])

        def __iter__(self):
            return iter(self._items)

    return _Iterable(rows)


ROWS = [{"article": f"article number {i}", "highlights": f"summary {i}"} for i in range(20)]


def _load(tmp_path, monkeypatch, *, rows=ROWS, fail_loader=False):
    def loader(*args, **kwargs):
        if fail_loader:
            raise AssertionError("network loader called, but a local snapshot should have served")
        return _fake_stream(rows)

    monkeypatch.setattr("datasets.load_dataset", loader)
    return huggingface_pool(
        SUM_SPEC, 5, split="train", row_adapter=resolve(SUM_SPEC), data_seed=1234,
        sampling="shuffled_buffer", shuffle_buffer_size=64, offline_snapshots=True,
        cache_dir=str(tmp_path), skip=0,
    )


def test_first_load_downloads_and_writes_a_snapshot(tmp_path, monkeypatch):
    examples, report = _load(tmp_path, monkeypatch)
    assert report.source == "download"
    assert report.n_examples == 5
    assert report.rows_consumed == 5
    written = list(tmp_path.rglob("*.jsonl"))
    assert len(written) == 1
    assert snapshots.manifest_path_for(written[0]).exists()


def test_second_load_is_served_locally_with_the_loader_monkeypatched_to_raise(tmp_path, monkeypatch):
    first, first_report = _load(tmp_path, monkeypatch)
    second, second_report = _load(tmp_path, monkeypatch, fail_loader=True)
    assert first_report.source == "download"
    assert second_report.source == "local"
    assert [e.example_id for e in first] == [e.example_id for e in second]
    assert first_report.snapshot_key == second_report.snapshot_key


def test_manifest_records_the_license_and_the_over_length_proxy(tmp_path, monkeypatch):
    _load(tmp_path, monkeypatch)
    manifest_file = next(tmp_path.rglob("*.manifest.json"))
    payload = json.loads(manifest_file.read_text())
    for key in ("hf_id", "hf_revision", "hf_config", "size", "skip", "sampling", "row_adapter"):
        assert key in payload["spec"], key
    assert payload["license"] == SUM_SPEC.license
    assert "over_length_fraction" in payload
    assert "proxy" in payload["over_length_metric"]


def test_disabling_snapshots_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: _fake_stream(ROWS))
    examples, report = huggingface_pool(
        SUM_SPEC, 5, split="train", row_adapter=resolve(SUM_SPEC), data_seed=1234,
        sampling="head_of_stream", shuffle_buffer_size=None, offline_snapshots=False,
        cache_dir=str(tmp_path), skip=0,
    )
    assert report.source == "download"
    assert list(tmp_path.rglob("*.jsonl")) == []
