from typing import cast

from translator_tom import Biolink, Qualifier, QualifierConstraint


def create_qualifier_constraint(name: str, value: str) -> QualifierConstraint:
    return QualifierConstraint(
        qualifier_set=[
            Qualifier(
                qualifier_type_id=Biolink.Qualifier(name),
                qualifier_value=value,
            )
        ]
    )


sex_qualifier_constraint = create_qualifier_constraint(
    "biolink:sex_qualifier", "PATO:0000383"
)
frequency_qualifier_constraint = create_qualifier_constraint(
    "biolink:frequency_qualifier", "HP:0040280"
)


qualifier_specifications = cast(
    list[Qualifier],
    [
        {
            "qualifier_type_id": "biolink:object_aspect_qualifier",
            "qualifier_value": "activity",
        },
        {
            "qualifier_type_id": "biolink:object_direction_qualifier",
            "qualifier_value": "increased",
        },
        {
            "qualifier_type_id": "biolink:qualified_predicate",
            "qualifier_value": "biolink:causes",
        },
    ],
)
single_entry_qualifier_set = QualifierConstraint(
    qualifier_set=qualifier_specifications[:1]
)
multi_entry_qualifier_set = QualifierConstraint(
    qualifier_set=qualifier_specifications[1:]
)

multiple_qualifier_constraints = [single_entry_qualifier_set, multi_entry_qualifier_set]

single_qualifier_constraint = [multi_entry_qualifier_set]

single_qualifier_constraint_with_single_qualifier_entry = [single_entry_qualifier_set]
