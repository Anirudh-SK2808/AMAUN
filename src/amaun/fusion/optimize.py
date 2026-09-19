"""M8 -- Algorithm 2: layer-wise learnable fusion with KD-guided joint optimisation.

Implements idea.md S6.7 literally, in order:

    1. compute Delta_W_old_l and Delta_W_temp_l for every layer     -> build_fused_model
    2. initialise alpha_l = 0.5                                     -> FusedLoRALinear
    3. Delta_W_merge_l = alpha_l Delta_W_old_l + (1-alpha_l) Delta_W_temp_l
    4. construct the merged model
    5. run A_old on D_rep and store q_old                           <- BEFORE step 6
    6-11. repeat: L_total = L_new(D_t) + lambda L_KD(D_rep); update phi_merge AND every a_l
    12. return the optimised merged adapter and [alpha_1 .. alpha_L]

Step 5 happens before the fused model is built, because once fusion is in place the model no
longer *is* ``A_old``.  The step order is asserted, not just documented (invariant I8.6).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from amaun.adapters.delta import AdapterDelta
from amaun.adapters.lora import load_delta_into
from amaun.data.preprocess import TokenizedBatch, masked_cross_entropy
from amaun.errors import FusionError
from amaun.fusion.layerwise import (
    FusionConfig,
    FusionState,
    assert_sources_unmutated,
    build_fused_model,
    build_optimizer,
    collapse,
    teardown_fused_model,
)
from amaun.fusion.refactorize import RefactorizationReport
from amaun.models.backbone import assert_backbone_frozen
from amaun.training.kd import KDConfig, TeacherCache, capture_teacher, kd_loss, verify_teacher
from amaun.utils.logging import get_logger

log = get_logger("fusion.optimize")


@dataclass
class FusionRecord:
    """Everything observed during one fusion event, for the fusion log."""

    fused_adapter_id: str
    component_ids: list[str]
    steps: int = 0
    alpha_trajectory: list[dict[str, Any]] = field(default_factory=list)
    loss_trajectory: list[dict[str, float]] = field(default_factory=list)
    final_alpha_by_layer: dict[str, list[float]] = field(default_factory=dict)
    trainable_parameters: dict[str, int] = field(default_factory=dict)
    kd: dict[str, Any] = field(default_factory=dict)
    refactorization: dict[str, Any] = field(default_factory=dict)
    wall_seconds: float = 0.0
    sources_verified_unmutated: bool = False
    alpha_moved_from_init: bool = False

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    def alpha_summary(self) -> dict[str, float]:
        """Mean coefficient per component across layers -- the interpretability signal of S6.6."""
        if not self.final_alpha_by_layer:
            return {}
        n_components = len(next(iter(self.final_alpha_by_layer.values())))
        summary = {}
        for index in range(n_components):
            values = [alphas[index] for alphas in self.final_alpha_by_layer.values()]
            summary[f"alpha_mean_component_{index}"] = sum(values) / len(values)
        return summary


def run_fusion_kd(
    *,
    backbone: Any,
    peft_model: Any,
    components: Sequence[AdapterDelta],
    task_batches: Sequence[TokenizedBatch],
    retention_batches: Sequence[TokenizedBatch],
    fusion_cfg: FusionConfig,
    kd_cfg: KDConfig,
    fused_adapter_id: str,
    default_rank: int | None = None,
    budget: Any | None = None,
) -> tuple[AdapterDelta, FusionRecord]:
    """Run Algorithm 2 and return the optimised fused adapter.

    ``components[0]`` must be ``A_old`` (the selected repository adapter, which is also the
    KD teacher) and ``components[-1]`` is ``A_temp``.
    """
    if len(components) < 2:
        raise FusionError("fusion requires at least A_old and A_temp")
    if not task_batches:
        raise FusionError("fusion needs at least one batch of current-task data (D_t)")

    old_adapter = components[0]
    record = FusionRecord(
        fused_adapter_id=fused_adapter_id,
        component_ids=[component.adapter_id for component in components],
    )
    start = time.perf_counter()
    device = backbone.device

    # ---- Step 5 (done first): capture the teacher while the model is still A_old ---------
    teacher: TeacherCache | None = None
    if kd_cfg.enabled and retention_batches:
        load_delta_into(peft_model, old_adapter)
        teacher = capture_teacher(
            peft_model,
            retention_batches,
            teacher_adapter_id=old_adapter.adapter_id,
            temperature=kd_cfg.temperature,
            device=device,
            top_p_truncation=kd_cfg.top_p_truncation,
        )
        verify_teacher(teacher, old_adapter.adapter_id)
        record.kd = {**teacher.as_dict(), "lambda": kd_cfg.lam, "enabled": True}
    else:
        record.kd = {
            "enabled": False,
            "reason": (
                "kd.enabled=false (ablation A4)"
                if not kd_cfg.enabled
                else "no retention data available for D_rep"
            ),
            "lambda": kd_cfg.lam if kd_cfg.enabled else 0.0,
        }

    # ---- Steps 1-4: build the fused model -----------------------------------------------
    base_model = backbone.model
    source_hashes_before = {c.adapter_id: c.content_hash() for c in components}
    state: FusionState = build_fused_model(base_model, list(components), fusion_cfg)
    record.trainable_parameters = state.trainable_parameter_count()

    initial_alpha = state.alpha_by_layer()
    record.alpha_trajectory.append({"step": 0, "alpha": initial_alpha})

    try:
        optimizer = build_optimizer(state, fusion_cfg)
        # The fused model must not have unfrozen anything in the backbone.
        assert_backbone_frozen(base_model, allow=set(
            name for name, param in base_model.named_parameters() if param.requires_grad
            and ("fusion_logit" in name or "lora_A_" in name or "lora_B_" in name)
        ))

        # ---- Steps 6-11: joint optimisation ---------------------------------------------
        base_model.train()
        for step in range(1, fusion_cfg.steps + 1):
            batch = task_batches[(step - 1) % len(task_batches)].to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = base_model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
            loss_new = masked_cross_entropy(logits, batch.labels)

            loss_kd = torch.zeros((), device=loss_new.device)
            if teacher is not None and kd_cfg.lam != 0.0:
                index = (step - 1) % len(teacher.batches)
                retention = teacher.batches[index].to(device)
                student_logits = base_model(
                    input_ids=retention.input_ids, attention_mask=retention.attention_mask
                ).logits
                loss_kd = kd_loss(
                    student_logits,
                    teacher.logits[index],
                    retention.labels,
                    temperature=kd_cfg.temperature,
                )

            total = loss_new + kd_cfg.lam * loss_kd
            total.backward()
            optimizer.step()

            if budget is not None:
                budget.add_steps()
                budget.add_batch(tokens=int(batch.attention_mask.sum()), examples=len(batch))

            record.loss_trajectory.append(
                {
                    "step": step,
                    "L_new": float(loss_new.detach()),
                    "L_KD": float(loss_kd.detach()),
                    "L_total": float(total.detach()),
                }
            )
            record.alpha_trajectory.append({"step": step, "alpha": state.alpha_by_layer()})

        record.steps = fusion_cfg.steps
        record.final_alpha_by_layer = state.alpha_by_layer()

        # Invariant I8.5 in effect: if alpha never moved, the coefficients were not learning.
        record.alpha_moved_from_init = any(
            abs(final[i] - initial_alpha[key][i]) > 1e-9
            for key, final in record.final_alpha_by_layer.items()
            for i in range(len(final))
        )
        if fusion_cfg.steps > 0 and not record.alpha_moved_from_init:
            log.warning(
                "fusion coefficients did not move from initialisation after %d steps; alpha may "
                "be detached from the graph or the learning rate may be zero", fusion_cfg.steps,
            )

        # ---- Step 12: collapse to a storable adapter ------------------------------------
        fused_delta, report = collapse(
            state, list(components), fusion_cfg,
            adapter_id=fused_adapter_id, default_rank=default_rank,
        )
        record.refactorization = report.as_dict()
    finally:
        # Always restore the backbone, even if optimisation raised.
        teardown_fused_model(state)
        base_model.eval()

    # Invariant I8.4: the repository's adapters must be untouched.
    for component in components:
        if component.content_hash() != source_hashes_before[component.adapter_id]:
            from amaun.errors import FusionSourceMutatedError

            raise FusionSourceMutatedError(
                f"source adapter '{component.adapter_id}' was mutated during fusion"
            )
    record.sources_verified_unmutated = True
    record.wall_seconds = time.perf_counter() - start

    log.info(
        "fused %s from %s in %d steps (%.1fs); mean alpha %s",
        fused_adapter_id, record.component_ids, record.steps, record.wall_seconds,
        {k: round(v, 4) for k, v in record.alpha_summary().items()},
    )
    return fused_delta, record
