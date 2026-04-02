"""
Live TRAPI endpoint tests.

Each test sends a POST request to the /query endpoint and validates the TRAPI
response structure and content. Tests are parametrised over tier 0 and tier 1
so the same semantic scenarios are exercised against both backends.

Set the RETRIEVER_URL environment variable to target a non-local instance.
"""

import os

import httpx
import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base URL of the running Retriever service.
# Override with the RETRIEVER_URL environment variable when targeting a
# non-local instance (e.g. the dev deployment).
RETRIEVER_URL: str = os.environ.get("RETRIEVER_URL", "http://localhost:8080")
QUERY_ENDPOINT: str = f"{RETRIEVER_URL}/query"

TIERS = [pytest.param(0, id="tier0"), pytest.param(1, id="tier1")]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request(tier: int, query_graph: dict) -> dict:
    return {
        "parameters": {"tiers": [tier]},
        "message": {"query_graph": query_graph},
    }


def _assert_ok(response: httpx.Response) -> dict:
    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}.\n{response.text}"
    )
    body = response.json()
    assert "message" in body, "Response body must contain a 'message' key"
    msg = body["message"]
    assert "knowledge_graph" in msg, "'message' must contain 'knowledge_graph'"
    assert "results" in msg, "'message' must contain 'results'"
    return msg


def _kg_node_ids(msg: dict) -> set[str]:
    return set(msg["knowledge_graph"].get("nodes", {}))


def _result_node_ids(msg: dict, binding: str) -> set[str]:
    ids: set[str] = set()
    for result in msg.get("results", []):
        for b in result.get("node_bindings", {}).get(binding, []):
            ids.add(b["id"])
    return ids


# ---------------------------------------------------------------------------
# Simple hop queries
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("tier", TIERS)
async def test_simple_one_query(tier: int) -> None:
    """
    NCBIGene:11276 (SYNRG) --located_in--> GO:0031410 (cytoplasmic vesicle).

    Both nodes and at least one result should be present.
    """
    query_graph = {
        "nodes": {
            "n0": {"ids": ["GO:0031410"], "constraints": []},
            "n1": {"ids": ["NCBIGene:11276"], "constraints": []},
        },
        "edges": {
            "e0": {
                "object": "n0",
                "subject": "n1",
                "predicates": ["biolink:located_in"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            }
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(QUERY_ENDPOINT, json=_request(tier, query_graph))

    msg = _assert_ok(response)
    assert msg["results"], "Expected at least one result for NCBIGene:11276 located_in GO:0031410"

    kg_nodes = _kg_node_ids(msg)
    assert "GO:0031410" in kg_nodes, "GO:0031410 must appear in the knowledge graph"
    assert "NCBIGene:11276" in kg_nodes, "NCBIGene:11276 must appear in the knowledge graph"

    n0_ids = _result_node_ids(msg, "n0")
    n1_ids = _result_node_ids(msg, "n1")
    assert "GO:0031410" in n0_ids, "GO:0031410 must appear in n0 result bindings"
    assert "NCBIGene:11276" in n1_ids, "NCBIGene:11276 must appear in n1 result bindings"


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("tier", TIERS)
async def test_reverse_query(tier: int) -> None:
    """
    DOID:0070271 --has_phenotype--> biolink:NamedThing (reversed traversal).

    DOID:0070271 is the subject; the object is any NamedThing.
    """
    query_graph = {
        "nodes": {
            "n0": {"categories": ["biolink:NamedThing"], "constraints": []},
            "n1": {"ids": ["DOID:0070271"], "constraints": []},
        },
        "edges": {
            "e0": {
                "object": "n0",
                "subject": "n1",
                "predicates": ["biolink:has_phenotype"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            }
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(QUERY_ENDPOINT, json=_request(tier, query_graph))

    msg = _assert_ok(response)
    assert msg["results"], "Expected has_phenotype results for DOID:0070271"

    kg_nodes = _kg_node_ids(msg)
    assert "DOID:0070271" in kg_nodes, "DOID:0070271 must appear in the knowledge graph"
    assert len(msg["results"]) >= 1


# ---------------------------------------------------------------------------
# Symmetric tests
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("tier", TIERS)
async def test_symmetric_predicate_query(tier: int) -> None:
    """
    NCBIGene:3778 --related_to--> biolink:NamedThing (symmetric predicate).

    The symmetric expansion should yield many results.
    """
    query_graph = {
        "nodes": {
            "n0": {"categories": ["biolink:NamedThing"], "constraints": []},
            "n1": {"ids": ["NCBIGene:3778"], "constraints": []},
        },
        "edges": {
            "e0": {
                "object": "n0",
                "subject": "n1",
                "predicates": ["biolink:related_to"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            }
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(QUERY_ENDPOINT, json=_request(tier, query_graph))

    msg = _assert_ok(response)
    assert msg["results"], "Expected related_to results for NCBIGene:3778"

    kg_nodes = _kg_node_ids(msg)
    assert "NCBIGene:3778" in kg_nodes, "NCBIGene:3778 must appear in the knowledge graph"
    assert len(msg["results"]) > 1, (
        "Expected multiple results from symmetric related_to expansion"
    )


# ---------------------------------------------------------------------------
# Subclassing tests
# ---------------------------------------------------------------------------
#
# At the TRAPI level, implicit subclassing is enabled by default in the service
#
# ── Case 1, Form B (ID→P→ID, source subclass) ────────────────────────────
#   A  = GO:0051055
#   A' = GO:0031393
#          A' subclass_of A;  A' → genetic_association → B
#   B  = EFO:0004528
#   No direct A → genetic_association → B edge exists.
#
# ── Case 1, Form C (ID→P→ID, target subclass) ────────────────────────────
#   A  = CHEBI:4042
#   B' = GO:0031393
#          A → affects → B' (direct);  B' subclass_of B
#   B  = GO:0051055
#   No direct A → affects → B edge exists.
#
# ── Case 2 (ID→P→CAT, source subclass) ───────────────────────────────────
#   A   = UMLS:C3273258
#   A'  = MONDO:0018551
#          A' → has_phenotype → HP:0034267 (PhenotypicFeature)
#   CAT = biolink:PhenotypicFeature
#
# ── Case 3 (CAT→P→ID, target subclass) ───────────────────────────────────
#   INTERMEDIATE = GO:0031393
#          → genetic_association → EFO:0004528
#          → subclass_of         → GO:0051055
#   n0 = {categories: ["biolink:BiologicalProcess"]}
#   n1 = {ids: ["EFO:0004528"]}
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("tier", TIERS)
async def test_subclass_case1_form_b(tier: int) -> None:
    """
    Case 1 Form B: GO:0051055 → genetic_association → EFO:0004528 via source subclass.
    """
    query_graph = {
        "nodes": {
            "n0": {"ids": ["GO:0051055"], "constraints": []},
            "n1": {"ids": ["EFO:0004528"], "constraints": []},
        },
        "edges": {
            "e0": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:genetic_association"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            }
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(QUERY_ENDPOINT, json=_request(tier, query_graph))

    msg = _assert_ok(response)
    assert msg["results"], "Expected results via subclass expansion (Case 1 Form B)"

    kg_nodes = _kg_node_ids(msg)
    assert "GO:0051055" in kg_nodes, (
        "A (GO:0051055, negative regulation of lipid biosynthetic process) "
        "must appear in the knowledge graph via Form B expansion"
    )
    assert "EFO:0004528" in kg_nodes, (
        "B (EFO:0004528, mean corpuscular hemoglobin concentration) "
        "must appear in the knowledge graph"
    )

    n0_ids = _result_node_ids(msg, "n0")
    assert "GO:0051055" in n0_ids, "GO:0051055 must appear in n0 result bindings"

    n1_ids = _result_node_ids(msg, "n1")
    assert "EFO:0004528" in n1_ids, "EFO:0004528 must appear in n1 result bindings"


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("tier", TIERS)
async def test_subclass_case1_form_c(tier: int) -> None:
    """
    Case 1 Form C: CHEBI:4042 → affects → GO:0051055 via target subclass.
    """
    query_graph = {
        "nodes": {
            "n0": {"ids": ["CHEBI:4042"], "constraints": []},
            "n1": {"ids": ["GO:0051055"], "constraints": []},
        },
        "edges": {
            "e0": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:affects"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            }
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(QUERY_ENDPOINT, json=_request(tier, query_graph))

    msg = _assert_ok(response)
    assert msg["results"], "Expected results via subclass expansion (Case 1 Form C)"

    kg_nodes = _kg_node_ids(msg)
    assert "CHEBI:4042" in kg_nodes, "A (CHEBI:4042, Cypermethrin) must appear in the knowledge graph"
    assert "GO:0051055" in kg_nodes, (
        "B (GO:0051055, negative regulation of lipid biosynthetic process) "
        "must appear in the knowledge graph via Form C expansion"
    )

    n0_ids = _result_node_ids(msg, "n0")
    assert "CHEBI:4042" in n0_ids, "CHEBI:4042 must appear in n0 result bindings"

    n1_ids = _result_node_ids(msg, "n1")
    assert "GO:0051055" in n1_ids, "GO:0051055 must appear in n1 result bindings"


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("tier", TIERS)
async def test_subclass_case2_id_to_cat(tier: int) -> None:
    """
    Case 2: UMLS:C3273258 → has_phenotype → biolink:PhenotypicFeature via source subclass.
    """
    query_graph = {
        "nodes": {
            "n0": {"ids": ["UMLS:C3273258"], "constraints": []},
            "n1": {"categories": ["biolink:PhenotypicFeature"], "constraints": []},
        },
        "edges": {
            "e0": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:has_phenotype"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            }
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(QUERY_ENDPOINT, json=_request(tier, query_graph))

    msg = _assert_ok(response)
    assert msg["results"], "Expected results via subclass expansion (Case 2)"

    kg_nodes = _kg_node_ids(msg)
    assert "UMLS:C3273258" in kg_nodes, (
        "A (UMLS:C3273258, Congenital Systemic Disorder) must appear in the knowledge graph"
    )

    known_phenotypes = {"HP:0034267", "HP:0000010"}
    assert known_phenotypes & kg_nodes, (
        f"At least one of the known phenotype nodes {known_phenotypes} must appear "
        "in the knowledge graph (reached via MONDO:0018551 subclass expansion)"
    )

    n0_ids = _result_node_ids(msg, "n0")
    assert "UMLS:C3273258" in n0_ids, "UMLS:C3273258 must appear in n0 result bindings"


# ---------------------------------------------------------------------------
# Multi-hop queries
#
# All four tests share a common interacts_with chain anchored at CHEBI:3125:
#   CHEBI:3125 → UMLS:C0282090  → CHEBI:22580 → UMLS:C0678941 → NCBIGene:3075 → NCBIGene:213
#
# Sub-paths exercised:
#   two-hop  : CHEBI:3125 → UMLS:C0282090 → CHEBI:22580
#   three-hop: CHEBI:3125 → UMLS:C0282090 → CHEBI:22580 → UMLS:C0678941
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("tier", TIERS)
async def test_two_hop_query(tier: int) -> None:
    """
    Two-hop chain: CHEBI:3125 → interacts_with → UMLS:C0282090 → interacts_with → CHEBI:22580.

    All three pinned nodes must appear in the knowledge graph and their
    respective result bindings.
    """
    query_graph = {
        "nodes": {
            "n0": {"ids": ["CHEBI:3125"], "constraints": []},
            "n1": {"ids": ["UMLS:C0282090"], "constraints": []},
            "n2": {"ids": ["CHEBI:22580"], "constraints": []},
        },
        "edges": {
            "e0": {
                "object": "n0",
                "subject": "n1",
                "predicates": ["biolink:interacts_with"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            },
            "e1": {
                "object": "n1",
                "subject": "n2",
                "predicates": ["biolink:interacts_with"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            },
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(QUERY_ENDPOINT, json=_request(tier, query_graph))

    msg = _assert_ok(response)
    assert msg["results"], "Expected results for two-hop interacts_with query"

    kg_nodes = _kg_node_ids(msg)
    for node_id in ("CHEBI:3125", "UMLS:C0282090", "CHEBI:22580"):
        assert node_id in kg_nodes, f"{node_id} must appear in the knowledge graph"

    assert "CHEBI:3125" in _result_node_ids(msg, "n0"), "CHEBI:3125 must appear in n0 bindings"
    assert "UMLS:C0282090" in _result_node_ids(msg, "n1"), "UMLS:C0282090 must appear in n1 bindings"
    assert "CHEBI:22580" in _result_node_ids(msg, "n2"), "CHEBI:22580 must appear in n2 bindings"


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("tier", TIERS)
async def test_three_hop_query(tier: int) -> None:
    """
    Three-hop chain: CHEBI:3125 → interacts_with → UMLS:C0282090 → interacts_with → CHEBI:22580 → interacts_with → UMLS:C0678941.

    All four pinned nodes must appear in the knowledge graph and their
    respective result bindings.
    """
    query_graph = {
        "nodes": {
            "n0": {"ids": ["CHEBI:3125"], "constraints": []},
            "n1": {"ids": ["UMLS:C0282090"], "constraints": []},
            "n2": {"ids": ["CHEBI:22580"], "constraints": []},
            "n3": {"ids": ["UMLS:C0678941"], "constraints": []},
        },
        "edges": {
            "e0": {
                "object": "n0",
                "subject": "n1",
                "predicates": ["biolink:interacts_with"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            },
            "e1": {
                "object": "n1",
                "subject": "n2",
                "predicates": ["biolink:interacts_with"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            },
            "e2": {
                "object": "n2",
                "subject": "n3",
                "predicates": ["biolink:interacts_with"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            },
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(QUERY_ENDPOINT, json=_request(tier, query_graph))

    msg = _assert_ok(response)
    assert msg["results"], "Expected results for three-hop interacts_with query"

    kg_nodes = _kg_node_ids(msg)
    for node_id in ("CHEBI:3125", "UMLS:C0282090", "CHEBI:22580", "UMLS:C0678941"):
        assert node_id in kg_nodes, f"{node_id} must appear in the knowledge graph"

    assert "CHEBI:3125" in _result_node_ids(msg, "n0"), "CHEBI:3125 must appear in n0 bindings"
    assert "UMLS:C0282090" in _result_node_ids(msg, "n1"), "UMLS:C0282090 must appear in n1 bindings"
    assert "CHEBI:22580" in _result_node_ids(msg, "n2"), "CHEBI:22580 must appear in n2 bindings"
    assert "UMLS:C0678941" in _result_node_ids(msg, "n3"), "UMLS:C0678941 must appear in n3 bindings"
