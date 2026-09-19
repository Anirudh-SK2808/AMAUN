"""M3 -- delta math, IO, switching, repository lifecycle (criteria C5, C6, D1-D6)."""

import pytest
import torch

from amaun.adapters.delta import AdapterDelta, LayerDelta, combine_deltas
from amaun.adapters.lora import ensure_lora, extract_delta, load_delta_into, zero_delta_
from amaun.adapters.metadata import AdapterMetadata, AdapterOrigin
from amaun.adapters.repository import AdapterEntry, AdapterRepository
from amaun.errors import AdapterError, IncompatibleAdapterError, UnauthorisedRegistrationError
from tests.conftest import make_delta, make_metadata


class _Decision:
    outcome = "FUSE"
    decision_id = "test-decision"


def test_layer_delta_apply_equals_dense_application():
    """The factored path must be exact, not approximate -- it is used everywhere."""
    generator = torch.Generator().manual_seed(0)
    layer = LayerDelta(
        A=torch.randn(4, 12, generator=generator),
        B=torch.randn(9, 4, generator=generator),
        scaling=2.5,
    )
    x = torch.randn(7, 12, generator=generator)
    assert torch.allclose(layer.apply(x), x @ layer.dense().T, atol=1e-5)


def test_combine_deltas_is_exact_at_the_delta_level():
    """idea.md S16.3: fusion happens on Delta_W, and concatenation makes it exact."""
    a, b = make_delta(1, "a"), make_delta(2, "b")
    fused = combine_deltas([a, b], [0.3, 0.7])
    for key in a.keys():
        expected = 0.3 * a[key].dense() + 0.7 * b[key].dense()
        assert torch.allclose(fused[key].dense(), expected, atol=1e-5)


def test_factor_averaging_differs_from_delta_averaging():
    """The error idea.md S16.3 forbids is real: prove the two operations differ."""
    a, b = make_delta(3, "a"), make_delta(4, "b")
    key = a.keys()[0]
    delta_level = 0.5 * a[key].dense() + 0.5 * b[key].dense()
    mean_a = 0.5 * (a[key].A + b[key].A)
    mean_b = 0.5 * (a[key].B + b[key].B)
    factor_level = a[key].scaling * (mean_b @ mean_a)
    assert not torch.allclose(delta_level, factor_level, atol=1e-3)


def test_delta_save_load_roundtrip_is_exact(tmp_path):
    delta = make_delta(5, "round")
    reloaded = AdapterDelta.load(delta.save(tmp_path / "d.pt"))
    assert reloaded.content_hash() == delta.content_hash()


def test_shape_mismatch_is_rejected():
    a = make_delta(6, "a", d_in=8)
    b = make_delta(7, "b", d_in=16)
    with pytest.raises(AdapterError):
        a.assert_alignable(b)


def test_adapter_switching_leaves_no_residue(backbone, lora_spec):
    """Invariant I3.5."""
    model = ensure_lora(backbone.model, lora_spec)
    encoded = backbone.tokenizer("probe input", return_tensors="pt")
    zero_delta_(model)
    with torch.no_grad():
        base = model(**encoded).logits.clone()

    live = extract_delta(model, adapter_id="live")
    for _key, layer in live.items():
        layer.B.normal_(0.0, 0.5)
    load_delta_into(model, live)
    with torch.no_grad():
        adapted = model(**encoded).logits.clone()
    assert not torch.allclose(base, adapted, atol=1e-6)

    zero_delta_(model)
    with torch.no_grad():
        restored = model(**encoded).logits.clone()
    assert torch.allclose(base, restored, atol=1e-7)


def test_registration_requires_a_decision(tmp_path):
    """Master-prompt constraint 10 / invariant I3.2."""
    repository = AdapterRepository(tmp_path / "repo", backbone_fingerprint="fp")
    entry = AdapterEntry(delta=make_delta(8, "A1"), metadata=make_metadata("A1"))
    with pytest.raises(UnauthorisedRegistrationError):
        repository.register(entry)
    repository.register(entry, authorised_by=_Decision())
    assert "A1" in repository


def test_bootstrap_registration_is_journaled_as_such(tmp_path):
    repository = AdapterRepository(tmp_path / "repo", backbone_fingerprint="fp")
    repository.register(
        AdapterEntry(delta=make_delta(9, "A1"), metadata=make_metadata("A1")),
        bootstrap=True, task_index=0,
    )
    events = repository.events()
    assert events[0]["event"] == "register"
    assert events[0]["bootstrap"] is True


def test_incompatible_backbone_is_refused(tmp_path):
    """Invariant I3.3 / idea.md S16.12."""
    repository = AdapterRepository(tmp_path / "repo", backbone_fingerprint="fp-A")
    entry = AdapterEntry(
        delta=make_delta(10, "A1"), metadata=make_metadata("A1", backbone_fingerprint="fp-B")
    )
    with pytest.raises(IncompatibleAdapterError):
        repository.register(entry, bootstrap=True)


def test_metadata_has_every_required_field():
    """Criterion D3: brief S6 field list."""
    required = {
        "adapter_id", "task_name", "task_type", "dataset", "training_configuration",
        "creation_time", "model_version", "lora_rank", "target_modules",
        "performance_metrics", "task_embedding", "resource_cost", "parameter_count",
        "storage_bytes",
    }
    assert required <= set(AdapterMetadata.__dataclass_fields__)


def test_tasks_served_accumulates_through_fuse_and_reuse(tmp_path):
    """D-014: adapter count alone must not be able to hide what an adapter serves."""
    repository = AdapterRepository(tmp_path / "repo", backbone_fingerprint="fp")
    for index in (1, 2):
        repository.register(
            AdapterEntry(
                delta=make_delta(index, f"A{index}"),
                metadata=make_metadata(f"A{index}", tasks_served=[f"t{index}"]),
            ),
            bootstrap=True, task_index=0,
        )
    fused = AdapterEntry(
        delta=make_delta(99, "AF"),
        metadata=make_metadata("AF", tasks_served=["t3"], origin=AdapterOrigin.FUSED.value),
    )
    repository.replace_with_fused(
        replaced_id="A1", fused_entry=fused, authorised_by=_Decision(), task_index=1
    )
    assert set(repository.retrieve("AF").metadata.tasks_served) == {"t1", "t3"}
    assert repository.retrieve("AF").metadata.merge_count == 1
    repository.mark_reuse("A2", "t4", 2)
    assert set(repository.retrieve("A2").metadata.tasks_served) == {"t2", "t4"}
    assert "A1" in repository.archived_ids()


def test_archive_is_soft_and_auditable(tmp_path):
    repository = AdapterRepository(tmp_path / "repo", backbone_fingerprint="fp")
    repository.register(
        AdapterEntry(delta=make_delta(11, "A1"), metadata=make_metadata("A1")),
        bootstrap=True,
    )
    repository.archive("A1", reason="test")
    assert "A1" not in repository
    assert "A1" in repository.archived_ids()
    assert (repository.root / "archive" / "A1" / "delta.pt").exists()


def test_repository_reload_verifies_content_hash(tmp_path):
    """Checkpoint restoration must detect a corrupted adapter."""
    root = tmp_path / "repo"
    repository = AdapterRepository(root, backbone_fingerprint="fp")
    repository.register(
        AdapterEntry(delta=make_delta(12, "A1"), metadata=make_metadata("A1")), bootstrap=True
    )
    reloaded = AdapterRepository.load(root, backbone_fingerprint="fp")
    assert reloaded.ids() == ["A1"]

    corrupt = make_delta(13, "A1")
    corrupt.save(root / "A1" / "delta.pt")
    with pytest.raises(AdapterError):
        AdapterRepository.load(root, backbone_fingerprint="fp")


def test_candidate_store_is_separate_from_the_repository(tmp_path):
    repository = AdapterRepository(tmp_path / "repo", backbone_fingerprint="fp")
    entry = AdapterEntry(delta=make_delta(14, "temp"), metadata=make_metadata("temp"))
    repository.candidates.put(entry)
    assert "temp" in repository.candidates
    assert "temp" not in repository
    repository.candidates.discard("temp")
    assert "temp" not in repository.candidates
