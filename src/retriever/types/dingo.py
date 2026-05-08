from typing import Any, TypedDict

from pydantic import TypeAdapter
from translator_tom import Biolink, Infores

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


class SourceInfo(TypedDict):
    """Information about a knowledge source."""

    id: str
    name: str
    description: str
    license: Any
    url: Any
    version: str


class NodeInfo(TypedDict):
    """Information about a class of nodes present in the knowledge."""

    category: list[Biolink.Entity]
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

    subject_category: list[Biolink.Entity]
    predicate: Biolink.Predicate
    object_category: list[Biolink.Entity]
    count: int
    primary_knowledge_sources: dict[Infores, int]
    qualifiers: dict[Biolink.Qualifier, int]
    attributes: dict[str, int]
    subject_id_prefixes: dict[str, int]
    object_id_prefixes: dict[str, int]


class EdgeSummary(TypedDict):
    """A summary of all edges present in the knowledge."""

    total_count: int
    predicates: dict[Biolink.Predicate, int]
    primary_knowledge_sources: dict[Infores, int]
    predicates_by_knowledge_source: dict[Infores, dict[Biolink.Predicate, int]]
    qualifiers: dict[Biolink.Qualifier, int]
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
