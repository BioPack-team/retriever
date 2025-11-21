from retriever.data_tiers.tier_1.elasticsearch.qualifier_types import  \
    ESQueryForSingleQualifierConstraint, ESConstraintsChainedQuery, ESTermClause
from retriever.types.trapi import QualifierConstraintDict
from retriever.utils import biolink

ES_QUAL_FIELD ="qualifiers"
ES_QUAL_NAME ="type_id"
ES_QUAL_VAL = "value"

def process_single_entry_result(results: list[ESTermClause]) -> ESTermClause | ESQueryForSingleQualifierConstraint:
    if len(results) == 1:
        return results[0]

    wrapped : ESQueryForSingleQualifierConstraint = {
        "bool": {
            "must": results
        }
    }

    return wrapped



def handle_single_constraint(constraint: QualifierConstraintDict) -> list[ESTermClause] | None:
    """Generate query terms based on single constraint. One constraint could contain multiple qualifiers set"""

    qualifiers = constraint["qualifier_set"]

    # empty qualifier set
    if not qualifiers:
        return None

    must: list[ESTermClause] = []

    for qualifier in qualifiers:
        qual_type = biolink.rmprefix(qualifier["qualifier_type_id"])
        qual_value = biolink.rmprefix(qualifier["qualifier_value"])


        must.append({ "term": { qual_type: qual_value } },)

    return must


def process_qualifier_constraints(constraints: list[QualifierConstraintDict] | None) -> ESConstraintsChainedQuery | ESQueryForSingleQualifierConstraint | ESTermClause | None:
    """
    Generate terms for a list of qualifier constraints.

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

    if not isinstance(constraints, list):
        raise TypeError("qualifier constraints must be a list")

    constraint_queries : list[list[ESTermClause]] = list(
        filter(
            None,
            map(handle_single_constraint, constraints)
        )
    )

    if not constraint_queries:
        return None

    should_array = list(
        map(
            process_single_entry_result,
            constraint_queries
        )
    )

    if len(should_array) == 1:
        # no need to use `should`
        return should_array[0]


    inner_query: ESConstraintsChainedQuery = {
        "should": should_array
    }

    return inner_query










