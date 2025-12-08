import itertools

from retriever.types.dingo import DINGOMetadata
from retriever.types.metakg import Operation, OperationNode
from retriever.types.trapi import (
    BiolinkEntity,
    Infores,
    MetaAttributeDict,
    MetaEdgeDict,
    MetaKnowledgeGraphDict,
    MetaNodeDict,
    QualifierTypeID,
)
from retriever.types.trapi_pydantic import TierNumber
from retriever.utils import biolink

DINGO_KG_EDGE_TOPLEVEL_VALUES = {
    "binding",
    "direction",
    "predicate",
    "predicate_ancestors",
    "node",
    "sources",
    "source_inforeses",
    "id",
    "subject",
    "object",
    "_index",
    "seq_",
    "negated",  # Should only ever show up as false, field to be removed in future
}


DINGO_KG_NODE_TOPLEVEL_VALUES = {
    "binding",
    "id",
    "name",
    "edges",
    "category",
}


def parse_dingo_metadata(
    metadata: DINGOMetadata, tier: TierNumber, infores: Infores
) -> tuple[list[Operation], dict[BiolinkEntity, OperationNode]]:
    """Parse a DINGO Metadata object to build operations."""
    operations = list[Operation]()
    nodes = dict[BiolinkEntity, OperationNode]()
    for edge in metadata["schema"]["edges"]:
        for sbj, obj in itertools.product(
            edge["subject_category"], edge["object_category"]
        ):
            operations.append(
                Operation(
                    subject=sbj,
                    predicate=edge["predicate"],
                    object=obj,
                    api=infores,
                    tier=tier,
                    attributes=[
                        MetaAttributeDict(attribute_type_id=attr_type)
                        for attr_type in edge["attributes"]
                    ],
                    qualifiers={
                        QualifierTypeID(biolink.ensure_prefix(qual_type)): []
                        for qual_type in edge["qualifiers"]
                    },
                )
            )
    for node in metadata["schema"]["nodes"]:
        for category in node["category"]:
            nodes[category] = OperationNode(
                prefixes={infores: list(node["id_prefixes"].keys())},
                attributes={
                    infores: [
                        MetaAttributeDict(attribute_type_id=attr_type)
                        for attr_type in node["attributes"]
                    ]
                },
            )

    return operations, nodes


def parse_trapi_metakg(
    metakg: MetaKnowledgeGraphDict, tier: TierNumber, infores: Infores
) -> tuple[list[Operation], dict[BiolinkEntity, OperationNode]]:
    """Parse a TRAPI MetaKG to build operations."""
    operations = list[Operation]()
    nodes = dict[BiolinkEntity, OperationNode]()
    for edge in metakg["edges"]:
        edge_dict = MetaEdgeDict(**edge)
        operations.append(
            Operation(
                subject=edge_dict["subject"],
                predicate=edge_dict["predicate"],
                object=edge_dict["object"],
                api=infores,
                tier=tier,
                attributes=edge_dict.get("attributes"),
                qualifiers={
                    qualifier["qualifier_type_id"]: qualifier.get(
                        "applicable_values", []
                    )
                    for qualifier in (edge_dict.get("qualifiers", []) or [])
                },
            )
        )
    for category, node in metakg["nodes"].items():
        node_dict = MetaNodeDict(**node)
        nodes[category] = OperationNode(
            prefixes={infores: node_dict.get("id_prefixes", [])},
            attributes={infores: (node_dict.get("attributes", []) or [])},
        )

    return operations, nodes
