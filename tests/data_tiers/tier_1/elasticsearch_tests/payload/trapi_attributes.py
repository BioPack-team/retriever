from retriever.types.trapi import AttributeConstraintDict

# sample
# {
#   "id": "biolink:publications",
#   "name": "Must have 3+ publications",
#   "operator": ">",
#   "value": [2,3],
# }


base_constraint: AttributeConstraintDict ={
    "id": "biolink:has_total",
    "name": "total value must be greater than 2",
    "operator": ">",
    "value": 2
}

base_negation_constraint: AttributeConstraintDict ={
    "id": "biolink:has_total",
    "name": "total value must not be greater than 4",
    "operator": ">",
    "value": 4,
    "not": True
}


ATTRIBUTE_CONSTRAINTS = [base_constraint, base_negation_constraint]
