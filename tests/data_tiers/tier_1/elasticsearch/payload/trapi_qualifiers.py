from typing import cast

from retriever.types.trapi import QualifierDict, QualifierConstraintDict

qualifier_specifications = cast(list[QualifierDict],
                                [
                                    {
                                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                                        "qualifier_value": "activity"
                                    },

                                    {
                                        "qualifier_type_id": "biolink:object_modifier_qualifier",
                                        "qualifier_value": "increased"
                                    },

                                    {
                                        "qualifier_type_id": "biolink:qualified_predicate",
                                        "qualifier_value": "biolink:causes"
                                    }
                                ]
)
single_entry_qualifier_set: QualifierConstraintDict = {
    "qualifier_set": qualifier_specifications[:1]
}
multi_entry_qualifier_set: QualifierConstraintDict = {
    "qualifier_set": qualifier_specifications[1:]
}

multiple_qualifier_constraints =  [
                single_entry_qualifier_set,
                multi_entry_qualifier_set
]

single_qualifier_constraint = [
    multi_entry_qualifier_set
]

single_qualifier_constraint_with_single_qualifier_entry = [
    single_entry_qualifier_set
]
