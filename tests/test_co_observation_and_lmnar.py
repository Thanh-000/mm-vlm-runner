"""Tests for G_obs co-observation graphs and exact zero-gradient guarantees of L_MNAR."""

from __future__ import annotations

import torch

from mnarm.methods.l_mnar import (
    analyze_co_observation_graph,
    compute_multimodal_lmnar_loss,
    compute_standard_ce_loss,
)


def _rec(uid: str, label: int, mask: list[str]) -> dict:
    return {"uid": uid, "label": label, "mask": mask}


def test_co_observation_graph_components() -> None:
    seen = list(range(8))
    # P1/P2 after historical paired reveal: all classes observed under (text, image) -> 1 component
    p1_records = [_rec(f"u{c}", c, ["text", "image"]) for c in seen]
    g1 = analyze_co_observation_graph(p1_records, seen)
    assert g1["num_components"] == 1
    assert g1["has_zero_support_mnar"] is False

    # P4 MNAR: early {0,1,4,5} image-only, late {2,3,6,7} paired -> 2 disconnected components
    p4_records = [
        _rec(f"e{c}", c, ["image"]) for c in (0, 1, 4, 5)
    ] + [
        _rec(f"l{c}", c, ["text", "image"]) for c in (2, 3, 6, 7)
    ]
    g4 = analyze_co_observation_graph(p4_records, seen)
    assert g4["num_components"] == 2
    assert g4["components"] == [[0, 1, 4, 5], [2, 3, 6, 7]]
    assert g4["tower_supports"]["text"] == [2, 3, 6, 7]
    assert g4["tower_supports"]["image"] == seen
    assert g4["has_zero_support_mnar"] is True

    # 3-expert disjoint MNAR: paired {0,1}, image {2,3}, text {4,5} -> 3 components
    r3 = (
        [_rec(f"a{c}", c, ["text", "image"]) for c in (0, 1)]
        + [_rec(f"b{c}", c, ["image"]) for c in (2, 3)]
        + [_rec(f"c{c}", c, ["text"]) for c in (4, 5)]
    )
    g3 = analyze_co_observation_graph(r3, [0, 1, 2, 3, 4, 5])
    assert g3["num_components"] == 3
    assert g3["components"] == [[0, 1], [2, 3], [4, 5]]


def test_lmnar_zero_gradient_on_unsupported_classes() -> None:
    torch.manual_seed(7)
    seen = [0, 1, 2, 3]
    # Early [0,1] image-only, Late [2,3] text+image (plus mixed propensity)
    records = (
        [_rec(f"e0_{i}", 0, ["image"]) for i in range(6)]
        + [_rec(f"e1_{i}", 1, ["image"]) for i in range(4)]
        + [_rec(f"l2t_{i}", 2, ["text", "image"]) for i in range(5)]
        + [_rec(f"l2i_{i}", 2, ["image"]) for i in range(3)]
        + [_rec(f"l3_{i}", 3, ["text", "image"]) for i in range(5)]
    )
    graph = analyze_co_observation_graph(records, seen)
    assert graph["tower_supports"]["text"] == [2, 3]
    assert abs(graph["tower_propensities"]["text"][2] - (5.0 / 8.0)) < 1e-6

    protos = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1).requires_grad_(True)
    ft = torch.randn(6, 16, requires_grad=True)
    fi = torch.randn(6, 16, requires_grad=True)
    y_late = torch.tensor([2, 3, 2, 3, 2, 3], dtype=torch.long)

    # Text-only batch with L_MNAR must put exact 0.0 gradient on unsupported prototypes [0, 1]
    loss_lmnar = compute_multimodal_lmnar_loss({"text": ft}, protos, 20.0, y_late, seen, graph)
    loss_lmnar.backward()
    assert protos.grad is not None
    assert float(protos.grad[[0, 1]].abs().max().item()) == 0.0
    assert float(protos.grad[[2, 3]].abs().max().item()) > 1e-4

    # Standard CE on same text-only batch contaminates unsupported classes [0, 1] with non-zero collider gradients
    protos.grad.zero_()
    loss_ce = compute_standard_ce_loss({"text": ft}, protos, 20.0, y_late, seen)
    loss_ce.backward()
    assert float(protos.grad[[0, 1]].abs().max().item()) > 1e-4

    # Image + Text batch updates all seen classes supported in image without error
    protos.grad.zero_()
    loss_both = compute_multimodal_lmnar_loss({"text": ft, "image": fi}, protos, 20.0, y_late, seen, graph)
    loss_both.backward()
    assert torch.isfinite(loss_both)
