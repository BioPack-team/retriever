from translator_tom.v2_0 import Biolink
from translator_tom.v2_0.model_dicts import QualifierSetConstraint


def create_qualifier_constraint(name: str, value: str) -> QualifierSetConstraint:
    """A TRAPI 2.0 QualifierSetConstraint: a flat {qualifier_type_id: value} mapping."""
    return {Biolink.Qualifier(name): value}


sex_qualifier_constraint = create_qualifier_constraint(
    "biolink:sex_qualifier", "PATO:0000383"
)
frequency_qualifier_constraint = create_qualifier_constraint(
    "biolink:frequency_qualifier", "HP:0040280"
)


single_entry_qualifier_set: QualifierSetConstraint = {
    "biolink:object_aspect_qualifier": "activity",
}
multi_entry_qualifier_set: QualifierSetConstraint = {
    "biolink:object_direction_qualifier": "increased",
    "biolink:qualified_predicate": "biolink:causes",
}

multiple_qualifier_constraints = [single_entry_qualifier_set, multi_entry_qualifier_set]

single_qualifier_constraint = [multi_entry_qualifier_set]

single_qualifier_constraint_with_single_qualifier_entry = [single_entry_qualifier_set]
