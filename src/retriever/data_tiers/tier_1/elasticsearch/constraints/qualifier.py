from retriever.data_tiers.tier_1.elasticsearch.qualifier_types import ESQueryForOneQualifierEntry, \
    ESQueryForSingleQualifierConstraint, ESConstraintsChainedQuery
from retriever.types.trapi import QualifierConstraintDict
from retriever.utils import biolink

ES_QUAL_FIELD ="qualifiers"
ES_QUAL_NAME ="type_id"
ES_QUAL_VAL = "value"

def process_single_entry_result(results: list[ESQueryForOneQualifierEntry]) -> ESQueryForOneQualifierEntry | ESQueryForSingleQualifierConstraint:
    if len(results) == 1:
        return results[0]

    wrapped : ESQueryForSingleQualifierConstraint = {
        "bool": {
            "must": results
        }
    }

    return wrapped



def handle_single_constraint(constraint: QualifierConstraintDict) -> list[ESQueryForOneQualifierEntry] | None:
    """Generate query terms based on single constraint. One constraint could contain multiple qualifiers set"""

    qualifiers = constraint["qualifier_set"]

    # empty qualifier set
    if not qualifiers:
        return None

    must: list[ESQueryForOneQualifierEntry] = []

    for qualifier in qualifiers:
        qual_type = biolink.rmprefix(qualifier["qualifier_type_id"])
        qual_value = biolink.rmprefix(qualifier["qualifier_value"])

        nested_query: ESQueryForOneQualifierEntry = {
            "nested": {
              "path": ES_QUAL_FIELD,
              "query": {
                "bool":
                    {
                        "must": [
                            { "term": { f"{ES_QUAL_FIELD}.{ES_QUAL_NAME}": qual_type } },
                            { "term": { f"{ES_QUAL_FIELD}.{ES_QUAL_VAL}": qual_value } }
                        ]
                    }
              }
            }
          }

        must.append(nested_query)

    return must


def process_qualifier_constraints(constraints: list[QualifierConstraintDict] | None) -> ESConstraintsChainedQuery | ESQueryForSingleQualifierConstraint | ESQueryForOneQualifierEntry | None:
    """
    Generate terms for a list of qualifier constraints.

    Example payload

    # ESConstraintsChainedQuery
     {
              "should": [

              # ESQueryForSingleQualifierConstraint
                {
                  "bool": {
                    "must": [
                      {
                        "nested": {
                          "path": "qualifiers",
                          "query": {
                            "bool": {
                              "must": [
                                { "term": { "qualifiers.type_id": "qualified_predicate" } },
                                { "term": { "qualifiers.value": "biolink:causes" } }
                              ]
                            }
                          }
                        }
                      },
                      {
                        "nested": {
                          "path": "qualifiers",
                          "query": {
                            "bool": {
                              "must": [
                                { "term": { "qualifiers.type_id": "subject_form_or_variant_qualifier" } },
                                { "term": { "qualifiers.value": "genetic_variant_form" } }
                              ]
                            }
                          }
                        }
                      }
                    ]
                  }
                },

              # ESQueryForOneQualifierEntry
                {
                  "nested": {
                    "path": "qualifiers",
                    "query": {
                      "bool": {
                        "must": [
                          { "term": { "qualifiers.type_id": "qualified_predicate" } },
                          { "term": { "qualifiers.value": "contributes_to" } }
                        ]
                      }
                    }
                  }
                }
              ]
            }
    """

    if constraints is None:
        return None

    if not isinstance(constraints, list):
        raise TypeError("qualifier constraints must be a list")

    constraint_queries : list[list[ESQueryForOneQualifierEntry]] = list(
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










