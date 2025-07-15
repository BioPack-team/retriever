from copy import deepcopy

import pytest
from reasoner_pydantic import QueryGraph
from utils.general import mock_inner_log  # pyright:ignore[reportImplicitRelativeImport]

from retriever.types.trapi import (
    AnalysisDict,
    AttributeDict,
    AuxGraphDict,
    EdgeBindingDict,
    EdgeDict,
    KnowledgeGraphDict,
    NodeBindingDict,
    NodeDict,
    ResultDict,
    RetrievalSourceDict,
)
from retriever.utils.logs import TRAPILogger
from retriever.utils.trapi import (
    edge_primary_knowledge_source,
    hash_attribute,
    hash_edge,
    hash_hex,
    hash_node_binding,
    hash_result,
    hash_retrieval_source,
    initialize_kgraph,
    merge_results,
    normalize_kgraph,
    prune_kg,
    update_edge,
    update_kgraph,
    update_node,
    update_retrieval_source,
)


@pytest.fixture
def kedge0() -> EdgeDict:
    return EdgeDict(
        {
            "predicate": "biolink:treats",
            "subject": "UNII:I031V2H011",
            "object": "MONDO:0015564",
            "attributes": [
                {
                    "attribute_type_id": "biolink:knowledge_level",
                    "value": "knowledge_assertion",
                },
                {"attribute_type_id": "biolink:agent_type", "value": "automated_agent"},
            ],
            "sources": [
                {
                    "resource_id": "infores:repodb",
                    "resource_role": "primary_knowledge_source",
                },
                {
                    "resource_id": "infores:biothings-repodb",
                    "resource_role": "aggregator_knowledge_source",
                    "upstream_resource_ids": ["infores:repodb"],
                },
                {
                    "resource_id": "infores:biothings-explorer",
                    "resource_role": "aggregator_knowledge_source",
                    "upstream_resource_ids": ["infores:biothings-repodb"],
                },
            ],
        }
    )


@pytest.fixture
def knode0() -> NodeDict:
    return NodeDict(
        {
            "categories": ["biolink:Disease"],
            "name": "Castleman disease",
            "attributes": [
                {
                    "attribute_type_id": "biolink:xref",
                    "value": [
                        "MONDO:0015564",
                        "DOID:0111157",
                        "orphanet:160",
                        "UMLS:C0017531",
                        "UMLS:C2931179",
                        "MESH:C536362",
                        "MESH:D005871",
                        "MEDDRA:10050251",
                        "NCIT:C3056",
                        "SNOMEDCT:207036003",
                        "SNOMEDCT:781094002",
                        "medgen:42211",
                        "ICD10:D47.Z2",
                    ],
                },
                {
                    "attribute_type_id": "biolink:synonym",
                    "value": [
                        "Castleman disease",
                        "Angiolymphoid hyperplasia",
                        "Angiofollicular ganglionic hyperplasia",
                        "Castleman Disease",
                    ],
                },
            ],
        }
    )


@pytest.fixture
def attr0() -> AttributeDict:
    return AttributeDict(
        attribute_type_id="some_attribute",
        value=["list", "of", "values", "should", "be", "hashable"],
    )


@pytest.fixture
def result0() -> ResultDict:
    return ResultDict(
        {
            "node_bindings": {
                "n01": [{"id": "CHEBI:4026", "attributes": []}],
                "n02": [{"id": "MONDO:0015564", "attributes": []}],
            },
            "analyses": [
                {
                    "resource_id": "infores:biothings-explorer",
                    "edge_bindings": {
                        "e01": [
                            {
                                "id": "inferred-CHEBI:4026-treats-MONDO:0015564",
                                "attributes": [],
                            }
                        ]
                    },
                    "score": 0.9619570542696206,
                }
            ],
        }
    )


@pytest.fixture
def kgraph0() -> KnowledgeGraphDict:
    return KnowledgeGraphDict(
        nodes={
            "n0": NodeDict(categories=["A"], attributes=[]),
            "n1": NodeDict(categories=["B"], attributes=[]),
        },
        edges={
            "e1": EdgeDict(
                subject="n0", predicate="biolink:related_to", object="n1", sources=[]
            )
        },
    )


@pytest.fixture
def kgraph1() -> KnowledgeGraphDict:
    return KnowledgeGraphDict(
        nodes={
            "n0": NodeDict(categories=["A"], attributes=[]),
            "n1": NodeDict(categories=["B"], attributes=[]),
            "n2": NodeDict(categories=["B"], attributes=[]),
        },
        edges={
            "e1": EdgeDict(
                subject="n0",
                predicate="biolink:related_to",
                object="n1",
                sources=[],
                attributes=[
                    AttributeDict(attribute_type_id="some_id", value="some_value")
                ],
            ),
            "e2": EdgeDict(
                subject="n0", predicate="biolink:has_subclass", object="n1", sources=[]
            ),
            "e3": EdgeDict(
                subject="n0", predicate="biolink:related_to", object="n2", sources=[]
            ),
        },
    )


@pytest.fixture
def results() -> list[ResultDict]:
    return [
        # Multiple analysis in one result shouldn't occur, but code should handle it.
        ResultDict(
            node_bindings={
                "qn0": [NodeBindingDict(id="n0", attributes=[])],
                "qn1": [NodeBindingDict(id="n1", attributes=[])],
            },
            analyses=[
                AnalysisDict(
                    resource_id="infores:retriever",
                    edge_bindings={"qe1": [EdgeBindingDict(id="e1", attributes=[])]},
                ),
                AnalysisDict(
                    resource_id="infores:retriever",
                    edge_bindings={"qe1": [EdgeBindingDict(id="e2", attributes=[])]},
                ),
            ],
        )
    ]


def test_initialize_kgraph() -> None:
    qgraph_dict = {
        "nodes": {
            "n0": {"categories": ["biolink:Gene"], "ids": ["NCBIGene:3778"]},
            "n1": {"categories": ["biolink:Disease"]},
        },
        "edges": {
            "e01": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:causes"],
            }
        },
    }
    qgraph = QueryGraph.model_validate(qgraph_dict)
    kgraph = initialize_kgraph(qgraph)

    assert len(kgraph["nodes"]) == 1
    assert "NCBIGene:3778" in kgraph["nodes"]
    assert "biolink:Gene" in kgraph["nodes"]["NCBIGene:3778"]["categories"]
    assert len(kgraph["edges"]) == 0


class TestEdgeDict:
    def test_attribute_hash_unchanged(self, kedge0: EdgeDict) -> None:
        hash0 = hash_edge(kedge0)
        kedge0["attributes"] = [
            *kedge0.get("attributes", []),
            AttributeDict(attribute_type_id="some_attribute", value="some_value"),
        ]
        hash1 = hash_edge(kedge0)

        assert hash0 == hash1

    def test_predicate_changes_hash(self, kedge0: EdgeDict) -> None:
        hash0 = hash_edge(kedge0)
        kedge0["predicate"] = "biolink:causes"
        hash1 = hash_edge(kedge0)

        assert hash0 != hash1

    def test_get_primary_knowledge_source(self, kedge0: EdgeDict) -> None:
        source = edge_primary_knowledge_source(kedge0)
        assert source is not None  # We know there is one on this edge
        assert source["resource_id"] == "infores:repodb"

        # Sources exist, but primary not present
        kedge0["sources"].pop(0)
        source = edge_primary_knowledge_source(kedge0)
        assert source is None

        # No sources
        kedge0["sources"].clear()
        source = edge_primary_knowledge_source(kedge0)
        assert source is None

    def test_update_edge(self, kedge0: EdgeDict) -> None:
        kedge1 = deepcopy(kedge0)
        new_attr = AttributeDict(attribute_type_id="some_id", value="some_value")
        kedge1["attributes"] = [*kedge1.get("attributes", []), new_attr]
        new_source = RetrievalSourceDict(
            resource_id="some_id", resource_role="some_role"
        )
        kedge1["sources"].pop(-1)
        kedge1["sources"].append(new_source)
        kedge1["sources"][0]["upstream_resource_ids"] = [
            *kedge1["sources"][0].get("upstream_resource_ids", []),
            "some_id",
        ]

        kedge2 = deepcopy(kedge0)
        del kedge2["attributes"]
        kedge2["sources"].clear()

        update_edge(kedge0, kedge1)

        assert "attributes" in kedge0
        assert new_attr in kedge0["attributes"]
        assert new_source in kedge0["sources"]
        assert "upstream_resource_ids" in kedge0["sources"][0]
        assert "some_id" in kedge0["sources"][0]["upstream_resource_ids"]

        update_edge(kedge0, kedge2)

        assert "attributes" in kedge0
        assert len(kedge0["attributes"]) > 0
        assert len(kedge0["sources"]) > 0


class TestAttributeDict:
    def test_hash_attribute(self, attr0: AttributeDict) -> None:
        hash0 = hash_attribute(attr0)

        assert type(hash0) is int

    def test_value_changes_hash(self, attr0: AttributeDict) -> None:
        hash0 = hash_attribute(attr0)
        attr0["value"].append("changes_hash")
        hash1 = hash_attribute(attr0)

        assert hash0 != hash1


def test_hash_hex(kedge0: EdgeDict) -> None:
    hash0 = hash_hex(hash_edge(kedge0))
    hash1 = hash_hex(hash_edge(kedge0))

    assert type(hash0) is str
    assert hash0 == hash1


class TestRetrievalSourceDict:
    def test_hash_retrieval_source(self) -> None:
        source = RetrievalSourceDict(resource_id="some_id", resource_role="some_role")
        hash0 = hash_retrieval_source(source)

        # Changing the resource_id should change the hash
        source["resource_id"] = "some_other_id"
        hash1 = hash_retrieval_source(source)

        assert hash0 != hash1

    def test_update_retrieval_source(self) -> None:
        source0 = RetrievalSourceDict(resource_id="some_id", resource_role="some_role")
        source1 = RetrievalSourceDict(
            resource_id="some_id",
            resource_role="some_role",
            upstream_resource_ids=["some_other_id"],
        )

        assert hash_retrieval_source(source0) == hash_retrieval_source(source1)

        update_retrieval_source(source0, source1)

        assert "upstream_resource_ids" in source0
        assert source0["upstream_resource_ids"][0] == "some_other_id"


def test_hash_node_binding() -> None:
    binding = NodeBindingDict(id="A", attributes=[])
    hash0 = hash_node_binding(binding)

    binding["id"] = "B"
    hash1 = hash_node_binding(binding)

    assert hash0 != hash1


class TestResultDict:
    def test_hash_result(self, result0: ResultDict) -> None:
        hash0 = hash_result(result0)

        result0["node_bindings"]["n02"][0]["id"] = "CURIE:some_other_id"
        hash1 = hash_result(result0)

        assert hash0 != hash1

    def test_merge_results(self, result0: ResultDict) -> None:
        result1 = deepcopy(result0)
        result1["node_bindings"]["n02"][0]["id"] = "CURIE:some_other_id0"

        results = {hash_result(result0): result0}

        merge_results(results, [deepcopy(result0), result1])

        assert len(results) == 2


class TestKnowledgeGraphDict:
    def test_normalize_kgraph(
        self, kgraph1: KnowledgeGraphDict, results: list[ResultDict]
    ) -> None:
        aux_graphs = {"aux0": AuxGraphDict(edges=["e1"], attributes=[])}
        normalize_kgraph(kgraph1, results, aux_graphs)

        for edge_id, edge in kgraph1["edges"].items():
            assert edge_id == hash_hex(hash_edge(edge))
        assert (
            results[0]["analyses"][0]["edge_bindings"]["qe1"][0]["id"]
            in kgraph1["edges"]
        )
        assert aux_graphs["aux0"]["edges"][0] in kgraph1["edges"]

    def test_update_kgraph(
        self, kgraph0: KnowledgeGraphDict, kgraph1: KnowledgeGraphDict
    ) -> None:
        aux_graphs = {"aux0": AuxGraphDict(edges=["e1"], attributes=[])}
        normalize_kgraph(kgraph0, [], aux_graphs)
        normalize_kgraph(kgraph1, [], aux_graphs)
        update_kgraph(kgraph0, kgraph1)

        assert len(kgraph0["nodes"]) == 3
        assert len(kgraph0["edges"]) == 3

        new_edge_id = next(iter(kgraph0["edges"].keys()))

        edge = kgraph0["edges"][new_edge_id]
        assert "attributes" in edge
        assert edge["attributes"][0]["value"] == "some_value"

    def test_prune_kgraph(
        self,
        kgraph1: KnowledgeGraphDict,
        results: list[ResultDict],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        aux_graphs = {"aux0": AuxGraphDict(edges=["e1"], attributes=[])}
        kgraph1["edges"]["e1"]["attributes"] = [
            AttributeDict(attribute_type_id="biolink:support_graphs", value=["aux0"])
        ]
        normalize_kgraph(kgraph1, results, aux_graphs)

        logger = TRAPILogger("somejobid")
        monkeypatch.setattr(TRAPILogger, "_log", mock_inner_log)

        prune_kg(results, kgraph1, aux_graphs, logger)
        assert len(kgraph1["nodes"]) == 2
        assert len(kgraph1["edges"]) == 2


def test_update_node(knode0: NodeDict) -> None:
    knode1 = deepcopy(knode0)
    knode1["name"] = "new_name"
    knode1["categories"] = [*knode1.get("categories", []), "new_category"]
    new_attr = AttributeDict(attribute_type_id="some_id", value="some_value")
    knode1["attributes"] = [*knode1.get("attributes", []), new_attr]

    update_node(knode0, knode1)

    assert "name" in knode0
    assert knode0["name"] == "new_name"
    assert "new_category" in knode0["categories"]
    assert new_attr in knode0["attributes"]

    knode2 = NodeDict(categories=[], attributes=[])

    update_node(knode0, knode2)

    assert len(knode0["categories"]) > 0
    assert len(knode0["attributes"]) > 0
