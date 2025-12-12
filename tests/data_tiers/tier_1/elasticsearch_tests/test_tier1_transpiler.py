import pytest
from payload.cases import (  # pyright:ignore[reportImplicitRelativeImport]
    Q_GRAPH_CASES,
    Q_GRAPH_CASES_IDS,
)
from payload.es_hits import (  # pyright:ignore[reportImplicitRelativeImport]
    SIMPLE_ES_HITS,
)
from payload.trapi_qgraphs import (  # pyright:ignore[reportImplicitRelativeImport]
    BASE_QGRAPH,
)

from retriever.data_tiers.tier_1.elasticsearch.constraints.types.qualifier_types import (
    ESQueryForSingleQualifierConstraint,
    ESTermClause,
)
from retriever.data_tiers.tier_1.elasticsearch.transpiler import (
    EDGE_FIELDS_MAPPING,
    NODE_FIELDS_MAPPING,
    ElasticsearchTranspiler,
)
from retriever.data_tiers.tier_1.elasticsearch.types import ESPayload
from retriever.types.trapi import QualifierConstraintDict, QualifierDict, QueryGraphDict
from retriever.utils import biolink

# sample generated query


@pytest.fixture
def es_transpiler() -> ElasticsearchTranspiler:
    return ElasticsearchTranspiler()


def check_list_fields(reference: list, against: list):
    for ref, ag in zip(reference, against):
        if ref.startswith("biolink:"):
            assert ref[8:] == ag
        else:
            assert ref == ag


def remove_biolink_prefixes(input: list):
    return list(map(biolink.rmprefix, input))


def verify_es_term_clause(qualifier: QualifierDict, generated_query: ESTermClause):
    assert "term" in generated_query
    term = generated_query["term"]
    qualifier_name = qualifier["qualifier_type_id"]
    qualifier_value = qualifier["qualifier_value"]
    assert term[biolink.rmprefix(qualifier_name)] == biolink.rmprefix(qualifier_value)


def verify_chained_es_term_clauses(
    qualifiers: list[QualifierDict],
    generated_query: ESQueryForSingleQualifierConstraint,
):
    assert "bool" in generated_query
    assert "must" in generated_query["bool"]
    query_terms: list[ESTermClause] = generated_query["bool"]["must"]

    assert len(query_terms) == len(qualifiers)

    for qualifier, terms in zip(qualifiers, query_terms):
        verify_es_term_clause(qualifier, terms)


def check_single_query_payload(q_graph: QueryGraphDict, generated_payload: ESPayload):
    assert generated_payload is not None

    query_content = generated_payload["query"]["bool"]
    filter_content = query_content["filter"]
    assert filter_content is not None
    assert isinstance(filter_content, list)

    q_edge = next(iter(q_graph["edges"].values()), None)
    out_node = q_graph["nodes"][q_edge["subject"]]
    in_node = q_graph["nodes"][q_edge["object"]]

    # qualifier checker
    # 0. check should, if > 1 constraints; make sure minimum_should_match
    # 1. otherwise, check filter; should have one or more filters with `term`

    if "qualifier_constraints" in q_edge:
        qualifier_entries: list[QualifierConstraintDict] = q_edge[
            "qualifier_constraints"
        ]

        if len(qualifier_entries) > 1:
            assert "should" in query_content
            assert "minimum_should_match" in query_content

            should_array = query_content["should"]
            assert len(should_array) == len(qualifier_entries)

            for qualifier_entry, generated_query in zip(
                qualifier_entries, should_array
            ):
                qualifiers = qualifier_entry["qualifier_set"]
                if len(qualifiers) == 1:
                    # assert 'bool' in generated_query and 'query' in generated_query['bool']
                    qualifier = qualifiers[0]
                    verify_es_term_clause(qualifier, generated_query)
                elif len(qualifiers) > 1:
                    verify_chained_es_term_clauses(qualifiers, generated_query)
        elif len(qualifier_entries) == 1:
            # in this case qualifier query is merged with `filter`
            qualifiers = qualifier_entries[0]["qualifier_set"]
            check_set = set(
                [
                    f"{
                        '%'.join(
                            remove_biolink_prefixes(
                                [
                                    qualifier['qualifier_type_id'],
                                    qualifier['qualifier_value'],
                                ]
                            )
                        )
                    }"
                    for qualifier in qualifiers
                ]
            )

            for term in filter_content:
                # `terms` is usually query filter, while `term` is usually qualifier
                if "term" in term:
                    qualifier_query = term["term"]
                    assert isinstance(qualifier_query, dict)
                    assert len(qualifier_query) == 1

                    for key in qualifier_query.keys():
                        assembled_value = "%".join([key, qualifier_query[key]])
                        if assembled_value in check_set:
                            check_set.remove(assembled_value)

            # all qualifiers should have been represented in the query
            assert len(check_set) == 0

    # hacky attribute checking for now
    if q_edge.get("attribute_constraints"):
        assert "must" in query_content
        must = query_content["must"]
        assert must == [
            {"range": {"has_total": {"gt": 2}}},
            {"range": {"has_total": {"lte": 4}}},
        ]

    # generate check targets
    required_fields_to_check = dict()
    # Generated field targets
    # {'predicate_ancestors': ['causes'], 'object.id.keyword': ['UMLS:C0011847'], 'subject.category': ['Gene'], 'object.category': ['Disease']}

    for field in EDGE_FIELDS_MAPPING.keys():
        if field in q_edge:
            required_fields_to_check[EDGE_FIELDS_MAPPING[field]] = (
                remove_biolink_prefixes(q_edge[field])
            )

    for field in NODE_FIELDS_MAPPING.keys():
        if field in out_node:
            required_fields_to_check[f"subject.{NODE_FIELDS_MAPPING[field]}"] = (
                remove_biolink_prefixes(out_node[field])
            )
        if field in in_node:
            required_fields_to_check[f"object.{NODE_FIELDS_MAPPING[field]}"] = (
                remove_biolink_prefixes(in_node[field])
            )

    for single_filter in filter_content:
        if "terms" not in single_filter:
            continue

        terms = single_filter["terms"]

        for term in terms.keys():
            if term in required_fields_to_check:
                assert terms[term] == required_fields_to_check[term]
                required_fields_to_check.pop(term)

    assert not required_fields_to_check


@pytest.mark.parametrize(*Q_GRAPH_CASES, ids=Q_GRAPH_CASES_IDS)
def test_convert_triple(
    q_graph: QueryGraphDict, es_transpiler: ElasticsearchTranspiler
) -> None:
    generated_payload = es_transpiler.convert_triple(q_graph)
    check_single_query_payload(q_graph, generated_payload)


@pytest.mark.parametrize(*Q_GRAPH_CASES, ids=Q_GRAPH_CASES_IDS)
def test_convert_batch_triple(
    q_graph: QueryGraphDict, es_transpiler: ElasticsearchTranspiler
) -> None:
    batch_q_graphs = [q_graph for i in range(10)]

    generated_payload_list = es_transpiler.convert_batch_triple(batch_q_graphs)
    for generated_payload in generated_payload_list:
        check_single_query_payload(q_graph, generated_payload)


@pytest.mark.asyncio
async def test_convert_results(es_transpiler: ElasticsearchTranspiler):
    result = es_transpiler.convert_results(BASE_QGRAPH, SIMPLE_ES_HITS)

    assert result is not None


@pytest.mark.asyncio
async def test_convert_batch_results(es_transpiler: ElasticsearchTranspiler):
    results = es_transpiler.convert_batch_results([BASE_QGRAPH], [SIMPLE_ES_HITS])

    for result in results:
        assert result is not None
