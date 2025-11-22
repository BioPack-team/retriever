# general guidance https://github.com/BioPack-team/retriever/issues/30#issuecomment-3549549249
# scheme reference
# - edge:
#   - https://github.com/NCATSTranslator/ReasonerAPI/blob/master/docs/reference.md#qedge-
#   - field name: attribute_constraints
# - node:
#   - https://github.com/NCATSTranslator/ReasonerAPI/blob/master/docs/reference.md#qnode-
#   - field name: constraints
from retriever.data_tiers.tier_1.elasticsearch.constraints.attributes.meta_info import ATTR_META
from retriever.types.trapi import AttributeConstraintDict
from retriever.utils import biolink


# sample
# {
#   "id": "biolink:publications",
#   "name": "Must have 3+ publications",
#   "operator": ">",
#   "value": [2,3],
# }


def validate_constraint(constraint: AttributeConstraintDict):
    required_fields = [
        "id",
        "operator",
        "value"
    ]

    for field in required_fields:
        if constraint.get(field, None) is None:
            raise AttributeError(f"Attribute constraint must have the field {field}")


def process_single_constraint(constraint: AttributeConstraintDict):
    """Process a single attribute constraint."""

    validate_constraint(constraint)

    # todo unit?

    raw_operator = constraint.get("operator")
    raw_value = constraint.get("value")
    should_negate = constraint.get("not", False)
    target_field_name = biolink.rmprefix(constraint.get("id"))

    field_meta_info = ATTR_META.get(target_field_name, None)

    if field_meta_info is None:
        # todo consider not met?
        return None













    '''
    decision tree:
    
    - check operator
        - matches : evaluate value( a regex) against `target` field or each element thereof, true if any (OR)
        - ===     : evaluate type and strict positional equality (AND) must: 
                        publications.keyword == a [a,b,c]
        
        - ==, >, <, >=, <= : 
                - `target` field is an array (can only be an array of strings)
                        |- yes: 
                            - the `value` is 
                                - an array of strings:
                                        - operator is ==
                                                |- yes: 
                                                    cross check (e.g. "publications", "==",  ["PMID:123", "PMID:234"]), true if any (OR)
                                                |- no: 
                                                    throw error, not meaningful comparison (ES does not support check like "publications", ">",  ["PMID:123", "PMID:234"])
                                - an array of numbers: 
                                        cross check against array length (e.g. "publications", "==" / ">",  [1,2,3]), true if any (OR)
                                - a number: 
                                        evaluate against array length (e.g. "publications", ">=", 2.5)
                                - OTHER:
                                        throw error, not a meaningful comparison (e.g. "publications", ">", [1, "PMID:123", 5.5])"
                                    
                        | - no:
                            - cross check meaningful comparison (i.e. of same type), true if any (OR) 
                                examples:
                                    - meaningful: 
                                        - "original_predicate", "==", "RO:001231
                                        - "original_predicate", ">", ["RO:001231"] (extraneous, but lexically meaningful) 
                                    - not meaningful: 
                                        - "original_predicate", ">=", 5
                                                
  '''











def process_constraints():
    pass