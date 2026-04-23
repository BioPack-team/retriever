from retriever.data_tiers.tier_1.elasticsearch.constraints.types.qualifier_types import (
    ESBoolQueryForExpandedQualifiers,
    ESConstraintsChainedQuery,
    ESEquivalentQualifierPairCollection,
    ESTermClause,
)
from retriever.types.trapi import QualifierConstraintDict
from retriever.utils import biolink
from retriever.utils.biolink import get_descendant_values, get_descendants

ES_QUAL_FIELD = "qualifiers"
ES_QUAL_NAME = "type_id"
ES_QUAL_VAL = "value"


def process_single_entry_result(
    results: list[ESEquivalentQualifierPairCollection],
) -> ESEquivalentQualifierPairCollection | ESBoolQueryForExpandedQualifiers:
    """Wrap chained processed qualifier query."""
    if len(results) == 1:
        return results[0]

    wrapped: ESBoolQueryForExpandedQualifiers = {"bool": {"must": results}}

    return wrapped


def expand_qualifier_pairs(
    pairs: set[tuple[str, str]],
) -> list[ESEquivalentQualifierPairCollection]:
    """Expand qualifier pairs to all possible descendant pairs."""
    must: list[ESEquivalentQualifierPairCollection] = []

    for q_type, q_value in pairs:
        # unique pairs of valid qualifier types and values
        equivalent_pairs: set[tuple[str, str]] = set()

        # get qualifier type descendants
        all_q_type_desc: list[str] = get_descendants(q_type)

        for q_type_desc in all_q_type_desc:
            # for each qualifier type descendant, get value descendants
            all_q_value_desc: set[str] = get_descendant_values(q_type_desc, q_value)
            for q_value_desc in all_q_value_desc:
                equivalent_pairs.add(
                    (biolink.rmprefix(q_type_desc), biolink.rmprefix(q_value_desc))
                )

        should_terms: list[ESTermClause] = [
            {"term": {_type: _value}} for _type, _value in equivalent_pairs
        ]

        must.append(
            {
                "bool": {
                    "should": should_terms,
                    "minimum_should_match": 1,
                }
            }
        )

    return must


def handle_single_constraint(
    constraint: QualifierConstraintDict,
) -> list[ESEquivalentQualifierPairCollection] | None:
    """Generate query terms based on single constraint. One constraint could contain multiple entries in its qualifiers set."""
    qualifiers = constraint["qualifier_set"]

    # empty qualifier set
    if not qualifiers:
        return None

    pairs: set[tuple[str, str]] = set()

    for qualifier in qualifiers:
        qual_type = qualifier["qualifier_type_id"]
        qual_value = qualifier["qualifier_value"]
        pairs.add((qual_type, qual_value))

    # within a qualifier set, it's AND relationship.
    must = expand_qualifier_pairs(pairs)

    return must


def process_qualifier_constraints(
    constraints: list[QualifierConstraintDict] | None,
) -> (
    ESConstraintsChainedQuery
    | ESEquivalentQualifierPairCollection
    | ESBoolQueryForExpandedQualifiers
    | None
):
    """Generate terms for a list of qualifier constraints.

    Example payload

    # ESConstraintsChainedQuery
    {
        "should": [

            # ESTermClause
            {
              "term": { "qualified_predicate": "biolink:causes" }
            },

            # ESQueryForSingleQualifierConstraint
            {
              "bool": {
                "must": [
                  { "term": { "subject_form_or_variant_qualifier": "genetic_variant_form" } },
                  { "term": { "evidence_level": "high" } }
                ]
              }
            }
      ]
    }

    """
    if constraints is None:
        return None

    # if not isinstance(constraints, list):
    #     raise TypeError("qualifier constraints must be a list")

    constraint_queries: list[list[ESEquivalentQualifierPairCollection]] = list(
        filter(None, map(handle_single_constraint, constraints))
    )

    if not constraint_queries:
        return None

    # within qualifier_constraints, it's OR relationship. We will wrap each entry as a separate query and put them in `should`.
    should_array: list[
        ESEquivalentQualifierPairCollection | ESBoolQueryForExpandedQualifiers
    ] = list(map(process_single_entry_result, constraint_queries))

    if len(should_array) == 1:
        # no need to use `should`
        return should_array[0]

    inner_query: ESConstraintsChainedQuery = {"should": should_array}

    return inner_query
