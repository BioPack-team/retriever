from typing import NotRequired, TypedDict

from pydantic import TypeAdapter

from retriever.types.trapi import (
    BiolinkEntity,
    BiolinkPredicate,
    Infores,
    QualifierTypeID,
)

DistributionInfo = TypedDict(
    "DistributionInfo",
    {
        "@id": str,
        "@type": str,
        "contentUrl": str,
        "encodingFormat": str,
        "description": str,
    },
)
"""Information expressing how to obtain the KGX files."""


class LicenseInfo(TypedDict):
    """Information expressing usage license."""

    terms_of_use_url: NotRequired[str]
    license_name: NotRequired[str]
    license_url: NotRequired[str]


class SourceInfo(TypedDict):
    """Information about a knowledge source."""

    id: str
    name: str
    description: str
    license: LicenseInfo | str
    url: list[str]
    version: str


class NodeInfo(TypedDict):
    """Information about a class of nodes present in the knowledge."""

    category: list[BiolinkEntity]
    count: int
    id_prefixes: dict[str, int]
    attributes: dict[str, int]


class NodeSummary(TypedDict):
    """A summary of all nodes present in the knowledge."""

    total_count: int
    id_prefixes: dict[str, int]
    attributes: dict[str, int]


class EdgeInfo(TypedDict):
    """Information about a class of edges present in the knowledge."""

    subject_category: list[BiolinkEntity]
    predicate: BiolinkPredicate
    object_category: list[BiolinkEntity]
    count: int
    primary_knowledge_sources: dict[Infores, int]
    qualifiers: dict[QualifierTypeID, int]
    attributes: dict[str, int]
    subject_id_prefixes: dict[str, int]
    object_id_prefixes: dict[str, int]


class EdgeSummary(TypedDict):
    """A summary of all edges present in the knowledge."""

    total_count: int
    predicates: dict[BiolinkPredicate, int]
    primary_knowledge_sources: dict[Infores, int]
    predicates_by_knowledge_source: dict[Infores, dict[BiolinkPredicate, int]]
    qualifiers: dict[QualifierTypeID, int]
    attributes: dict[str, int]


class SchemaInfo(TypedDict):
    """Information about the given knowledge schema."""

    nodes: list[NodeInfo]
    nodes_summary: NodeSummary
    edges: list[EdgeInfo]
    edges_summary: EdgeSummary


DINGOMetadata = TypedDict(
    "DINGOMetadata",
    {
        "@id": str,
        "@type": str,
        "name": str,
        "description": str,
        "license": str,
        "url": str,
        "version": str,
        "dateCreated": str,
        "biolinkVersion": str,
        "babelVersion": str,
        "distribution": list[DistributionInfo],
        "isBasedOn": list[SourceInfo],
        "schema": SchemaInfo,
    },
)
"""Metadata pertaining to a given DINGO ingest file."""

DINGO_ADAPTER = TypeAdapter(DINGOMetadata)
