"""Retriever delegates set_interpretation collapsing to TOM's dict post-solver.

These verify Retriever's wiring — the all-BATCH shortcut, delegation to the solver,
and MANY-as-BATCH — not TOM's grouping algorithm (covered by TOM's own tests).
"""

import uuid

from translator_tom.v2_0 import QueryGraph
from translator_tom.v2_0.model_dicts import KnowledgeGraphDict, NodeDict, ResultDict

from retriever.utils.logs import TRAPILogger
from retriever.utils.trapi import solve_set_interpretation

SET_ID = str(uuid.uuid4())


def _log() -> TRAPILogger:
    return TRAPILogger(job_id="set-interp-test")


def _result(n0: str, n1: str, edge: str) -> ResultDict:
    return {
        "node_bindings": {"n0": {"ids": [n0]}, "n1": {"ids": [n1]}},
        "analyses": [
            {
                "resource_id": "infores:retriever",
                "edge_bindings": {"e0": {"ids": [edge]}},
            }
        ],
    }


def _kgraph(node_ids: list[str]) -> KnowledgeGraphDict:
    return {"nodes": {nid: NodeDict(categories=[]) for nid in node_ids}, "edges": {}}


def _qgraph(interpretation: str, member_ids: list[str] | None = None) -> QueryGraph:
    n1: dict = {"categories": ["biolink:Disease"], "set_interpretation": interpretation}
    if interpretation == "BATCH":
        n1["ids"] = ["MONDO:1"]
    else:
        n1["ids"] = [SET_ID]
        n1["member_ids"] = member_ids or []

    return QueryGraph.model_validate(
        {
            "nodes": {
                "n0": {"ids": ["NCBIGene:1"], "categories": ["biolink:Gene"]},
                "n1": n1,
            },
            "edges": {"e0": {"subject": "n0", "object": "n1"}},
        }
    )


def test_all_batch_shortcuts_without_solving():
    """An all-BATCH graph skips the solver and returns the same results object."""
    results = [_result("NCBIGene:1", "MONDO:1", "e_a")]

    out = solve_set_interpretation(
        _qgraph("BATCH"), results, _kgraph(["NCBIGene:1", "MONDO:1"]), _log()
    )

    assert out is results


def test_all_collapses_to_set_binding():
    """ALL merges member results into one, binding the set node to its set id."""
    members = ["MONDO:1", "MONDO:2"]
    results = [
        _result("NCBIGene:1", "MONDO:1", "e_a"),
        _result("NCBIGene:1", "MONDO:2", "e_b"),
    ]

    out = solve_set_interpretation(
        _qgraph("ALL", members),
        results,
        _kgraph([SET_ID, "NCBIGene:1", *members]),
        _log(),
    )

    assert len(out) == 1
    assert out[0]["node_bindings"]["n1"]["ids"] == [SET_ID]


def test_many_treated_as_batch():
    """MANY is not collated (skip_many); distinct member results are retained."""
    results = [
        _result("NCBIGene:1", "MONDO:1", "e_a"),
        _result("NCBIGene:1", "MONDO:2", "e_b"),
    ]

    out = solve_set_interpretation(
        _qgraph("MANY", ["MONDO:1", "MONDO:2"]),
        results,
        _kgraph([SET_ID, "NCBIGene:1", "MONDO:1", "MONDO:2"]),
        _log(),
    )

    assert len(out) == 2
