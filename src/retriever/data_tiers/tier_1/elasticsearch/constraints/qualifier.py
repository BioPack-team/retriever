from retriever.data_tiers.tier_1.elasticsearch.types import ESQueryForSingleQualifierConstraint, \
    ESQueryForOneQualifierEntry
from retriever.types.trapi import QualifierConstraintDict
from retriever.utils import biolink

ES_QUAL_FIELD ="qualifiers"
ES_QUAL_NAME ="type_id"
ES_QUAL_VAL = "value"


def handle_single_qualifier_constraint(constraint: QualifierConstraintDict) -> ESQueryForSingleQualifierConstraint | None:
    """Generate query terms based on single qualifier constraint."""

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


    per_constraint_query: ESQueryForSingleQualifierConstraint = {
        "bool":  {
            "must": must
        }
    }

    return per_constraint_query
