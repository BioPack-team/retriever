from typing import cast, Any

import pytest

from retriever.data_tiers.tier_1.elasticsearch.transpiler import ElasticsearchTranspiler
from retriever.data_tiers.tier_1.elasticsearch.types import ESPayload, ESHit
from retriever.types.trapi import QueryGraphDict


def qg(d: dict[str, Any]) -> QueryGraphDict:
    """Cast a raw qgraph dict into a QueryGraphDict for type-checking in tests."""
    return cast(QueryGraphDict, cast(object, d))

def esh(d: dict[str, Any]) -> ESHit:
    return cast(ESHit, cast(ESPayload, cast(dict, d)))

SIMPLE_QGRAPH_0: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:4514"], "constraints": []},
        "n1": {"ids": ["UMLS:C1564592"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["subclass_of"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
    },
})

SIMPLE_QGRAPH_1: QueryGraphDict = qg({
        "nodes": {
                "n0": {"categories": ["biolink:Gene"]},
                "n1": {"categories": ["biolink:Disease"], "ids": ["UMLS:C0011847"]}
            },
            "edges": {
                "e01": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:causes"]
                }
            }
})


SIMPLE_QGRAPH_MULTIPLE_IDS: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:3125", "CHEBI:53448"], "constraints": []},
        "n1": {"ids": ["UMLS:C0282090", "CHEBI:10119"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["interacts_with"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
    },
})

SIMPLE_ES_HITS: list[ESHit] = [esh(
{
                    "subject": {
                        "id": "UMLS:C1564592",
                        "name": "Diclomin",
                        "category": "ChemicalEntity",
                        "all_names": [
                            "Diclomin"
                        ],
                        "all_categories": [
                            "Entity",
                            "ChemicalOrDrugOrTreatment",
                            "ChemicalEntity",
                            "PhysicalEssence",
                            "NamedThing",
                            "ChemicalEntityOrGeneOrGeneProduct",
                            "ChemicalEntityOrProteinOrPolypeptide",
                            "PhysicalEssenceOrOccurrent"
                        ],
                        "iri": "https://identifiers.org/umls:C1564592",
                        "description": "UMLS Semantic Type: STY:T109; UMLS Semantic Type: STY:T121",
                        "equivalent_curies": [
                            "UMLS:C1564592"
                        ]
                    },
                    "object": {
                        "id": "CHEBI:4514",
                        "name": "dicyclomine",
                        "category": "SmallMolecule",
                        "all_names": [
                            "dicyclomine hydrochloride 2 MG/ML Oral Solution",
                            "dicyclomine Oral Tablet",
                            "dicyclomine hydrochloride 20 MG [Bentyl]",
                            "Bentyl Oral Product",
                            "dicyclomine Injectable Product",
                            "Dicyclomine",
                            "dicyclomine Oral Solution",
                            "dicycloverin",
                            "dicyclomine Oral Capsule [Bentyl]",
                            "dicyclomine Injection [Bentyl]",
                            "dicyclomine hydrochloride 10 MG/ML [Bentyl]",
                            "dicyclomine Oral Capsule",
                            "dicyclomine hydrochloride 2 MG/ML",
                            "dicyclomine hydrochloride 10 MG/ML",
                            "dicyclomine hydrochloride 20 MG Oral Capsule",
                            "2 ML dicyclomine hydrochloride 10 MG/ML Injection",
                            "dicycloverine",
                            "Bentyl",
                            "bentyl",
                            "DICYCLOMINE",
                            "dicyclomine Oral Product",
                            "Bentyl Pill",
                            "dicyclomine",
                            "dicyclomine hydrochloride 10 MG Oral Capsule",
                            "2 ML dicyclomine hydrochloride 10 MG/ML Injection [Bentyl]",
                            "dicyclomine hydrochloride 10 MG [Bentyl]",
                            "dicyclomine Pill",
                            "dicyclomine hydrochloride 10 MG Oral Tablet",
                            "dicyclomine Oral Liquid Product",
                            "dicyclomine Oral Tablet [Bentyl]",
                            "dicyclomine hydrochloride 10 MG",
                            "Bentyl Injectable Product",
                            "dicyclomine hydrochloride 20 MG",
                            "dicyclomine hydrochloride",
                            "DICYCLOMINE HYDROCHLORIDE",
                            "dicyclomine hydrochloride 10 MG Oral Capsule [Bentyl]",
                            "dicyclomine single use injection",
                            "dicyclomine hydrochloride 20 MG Oral Tablet"
                        ],
                        "all_categories": [
                            "Entity",
                            "ChemicalEntity",
                            "ChemicalOrDrugOrTreatment",
                            "ChemicalMixture",
                            "PhysicalEssence",
                            "MolecularEntity",
                            "NamedThing",
                            "MolecularMixture",
                            "OntologyClass",
                            "Drug",
                            "ChemicalEntityOrGeneOrGeneProduct",
                            "ChemicalEntityOrProteinOrPolypeptide",
                            "PhysicalEssenceOrOccurrent",
                            "SmallMolecule"
                        ],
                        "iri": "http://purl.obolibrary.org/obo/CHEBI_4514",
                        "description": "Dicyclomine is only found in individuals that have used or taken this drug. It is a muscarinic antagonist used as an antispasmodic and in urinary incontinence. It has little effect on glandular secretion or the cardiovascular system. It does have some local anesthetic properties and is used in gastrointestinal, biliary, and urinary tract spasms. [PubChem]Action is achieved via a dual mechanism: (1) a specific anticholinergic effect (antimuscarinic) at the acetylcholine-receptor sites and (2) a direct effect upon smooth muscle (musculotropic).",
                        "equivalent_curies": [
                            "RXNORM:991063",
                            "RXNORM:1722904",
                            "RXNORM:991151",
                            "RXCUI:2657977",
                            "CAS:104959-55-9",
                            "UMLS:C3205694",
                            "CHV:0000044640",
                            "RXNORM:1151168",
                            "RXNORM:1171479",
                            "RXNORM:991069",
                            "UMLS:C0591771",
                            "UMLS:C0976352",
                            "RXCUI:366711",
                            "HMDB:HMDB0014942",
                            "UMLS:C4719851",
                            "UMLS:C3225558",
                            "UMLS:C1245855",
                            "UMLS:C0353803",
                            "UMLS:C4730928",
                            "UMLS:C1245859",
                            "RXNORM:991086",
                            "RXCUI:991081",
                            "RXCUI:991063",
                            "RXNORM:991061",
                            "RXCUI:1151169",
                            "UMLS:C2916892",
                            "UMLS:C0688579",
                            "RXCUI:2649127",
                            "UMLS:C5841834",
                            "NCIT:C61720",
                            "RXNORM:991082",
                            "RXNORM:991064",
                            "KEGG.DRUG:D00717",
                            "UMLS:C5838728",
                            "UMLS:C0709217",
                            "RXCUI:2662657",
                            "RXCUI:2646528",
                            "RXNORM:366711",
                            "RXNORM:991060",
                            "RXCUI:2654863",
                            "UMLS:C2916899",
                            "UMLS:C5838319",
                            "RXCUI:1171479",
                            "RXCUI:1722902",
                            "RXCUI:991087",
                            "UMLS:C2916891",
                            "RXCUI:203018",
                            "UMLS:C0012125",
                            "UMLS:C4060245",
                            "RXCUI:2647822",
                            "MESH:D004025",
                            "UMLS:C3225559",
                            "CAS:77-19-0",
                            "UMLS:C1245857",
                            "RXCUI:991082",
                            "RXCUI:2661688",
                            "UMLS:C5845493",
                            "CHV:0000003899",
                            "RXNORM:203018",
                            "CHEBI:4514",
                            "UMLS:C1270926",
                            "UMLS:C5846109",
                            "RXCUI:2645632",
                            "UMLS:C3205696",
                            "RXCUI:2659854",
                            "RXNORM:991065",
                            "UMLS:C2916902",
                            "NDDF:004711",
                            "UMLS:C5846461",
                            "DRUGBANK:DB00804",
                            "UMLS:C2916901",
                            "UMLS:C0688582",
                            "KEGG.COMPOUND:C06951",
                            "INCHIKEY:CURUTKGFNZGFSE-UHFFFAOYSA-N",
                            "RXNORM:371817",
                            "RXNORM:991616",
                            "RXNORM:1171480",
                            "RXCUI:2654453",
                            "CAS:67-92-5",
                            "UNII:4KV4X8IF6V",
                            "UMLS:C0709215",
                            "RXCUI:991061",
                            "RXNORM:1151166",
                            "RXNORM:371813",
                            "UMLS:C5782099",
                            "RXCUI:1151167",
                            "RXCUI:991086",
                            "RXCUI:371813",
                            "RXNORM:991085",
                            "UMLS:C4050109",
                            "RXCUI:1722904",
                            "GTOPDB:355",
                            "VANDF:4017870",
                            "RXCUI:1171480",
                            "RXCUI:152021",
                            "UMLS:C0305432",
                            "DrugCentral:868",
                            "VANDF:4019716",
                            "RXCUI:991060",
                            "PUBCHEM.COMPOUND:3042",
                            "RXCUI:991062",
                            "RXCUI:3361",
                            "RXCUI:2662305",
                            "RXCUI:991064",
                            "RXCUI:1171477",
                            "RXNORM:371815",
                            "UMLS:C3225556",
                            "UMLS:C1240680",
                            "PUBCHEM.COMPOUND:441344",
                            "RXCUI:991065",
                            "RXNORM:1171477",
                            "RXNORM:991087",
                            "UMLS:C3205695",
                            "UMLS:C0692732",
                            "RXNORM:368081",
                            "UMLS:C0700023",
                            "RXCUI:1151166",
                            "RXNORM:1151169",
                            "RXNORM:991068",
                            "UMLS:C4719825",
                            "UMLS:C2916890",
                            "RXCUI:371817",
                            "RXNORM:991081",
                            "RXCUI:368081",
                            "ATC:A03AA07",
                            "RXNORM:1151167",
                            "RXCUI:991151",
                            "RXNORM:3361",
                            "RXNORM:991062",
                            "RXCUI:991068",
                            "CHEMBL.COMPOUND:CHEMBL1123",
                            "RXCUI:991616",
                            "UMLS:C1242057",
                            "UMLS:C2916894",
                            "RXCUI:991085",
                            "UMLS:C5843663",
                            "RXCUI:371815",
                            "RXCUI:991069",
                            "RXCUI:1151168"
                        ],
                        "publications": [
                            "PMID:22194678",
                            "PMID:23523385",
                            "PMID:3612532",
                            "PMID:3597632",
                            "PMID:2579237",
                            "PMID:18834112",
                            "PMID:24332655",
                            "PMID:1920350",
                            "PMID:14254329",
                            "PMID:22961681"
                        ]
                    },
                    "predicate": "subclass_of",
                    "primary_knowledge_source": "infores:mesh",
                    "kg2_ids": [
                        "UMLS:C1564592---MESH:RN---None---None---None---UMLS:C0012125---umls_source:MSH",
                        "UMLS:C0012125---MESH:RB---None---None---None---UMLS:C1564592---umls_source:MSH"
                    ],
                    "domain_range_exclusion": False,
                    "knowledge_level": "knowledge_assertion",
                    "agent_type": "manual_agent",
                    "id": 1375555,
                    "all_predicates": [
                        "subclass_of",
                        "related_to_at_concept_level",
                        "related_to"
                    ]
                }
)]


@pytest.fixture
def es_transpiler() -> ElasticsearchTranspiler:
    return ElasticsearchTranspiler()


def check_list_fields(reference: list, against: list):
    for ref, ag in zip(reference, against):
        if ref.startswith("biolink:"):
            assert ref[8:] == ag
        else:
            assert ref == ag


def check_single_query_payload(
    q_graph: QueryGraphDict, generated_payload:ESPayload
):
    assert generated_payload is not None

    filter_content = generated_payload["query"]["bool"]["filter"]
    assert filter_content is not None
    assert isinstance(filter_content, list)

    q_edge = next(iter(q_graph["edges"].values()), None)
    in_node = q_graph["nodes"][q_edge["subject"]]
    out_node = q_graph["nodes"][q_edge["object"]]

    for single_filter in filter_content:
        terms = single_filter["terms"]
        if "subject.id" in terms:
            assert in_node["ids"] == terms["subject.id"]
        if "object.id" in terms:
            assert out_node["ids"] == terms["object.id"]

        for field_name in ["predicates", "categories"]:
            variant = f"all_{field_name}"
            if variant in terms:
                check_list_fields(q_edge[field_name], terms[variant])


Q_GRAPH_CASES = (
    "q_graph",
    [SIMPLE_QGRAPH_0, SIMPLE_QGRAPH_1, SIMPLE_QGRAPH_MULTIPLE_IDS],
)

Q_GRAPH_CASES_IDS = ["single id 0","single id 1", "multiple ids"]

@pytest.mark.parametrize(*Q_GRAPH_CASES, ids=Q_GRAPH_CASES_IDS)
def test_convert_triple(q_graph: QueryGraphDict, es_transpiler: ElasticsearchTranspiler) -> None:
    generated_payload = es_transpiler.convert_triple(q_graph)
    check_single_query_payload(q_graph, generated_payload)


@pytest.mark.parametrize(*Q_GRAPH_CASES, ids=Q_GRAPH_CASES_IDS)
def test_convert_batch_triple(q_graph: QueryGraphDict, es_transpiler: ElasticsearchTranspiler) -> None:
    batch_q_graphs = [
        q_graph
        for i in range(10)
    ]

    generated_payload_list = es_transpiler.convert_batch_triple(batch_q_graphs)
    for generated_payload in generated_payload_list:
        check_single_query_payload(q_graph, generated_payload)


@pytest.mark.asyncio
async def test_convert_results(es_transpiler: ElasticsearchTranspiler):
    result = es_transpiler.convert_results(
        SIMPLE_QGRAPH_0,
        SIMPLE_ES_HITS
    )

    assert result is not None

@pytest.mark.asyncio
async def test_convert_batch_results(es_transpiler: ElasticsearchTranspiler):
    results = es_transpiler.convert_batch_results(
        [SIMPLE_QGRAPH_0],
        [SIMPLE_ES_HITS]
    )

    for result in results:
        assert result is not None