from typing import Any, NamedTuple

from retriever.types.trapi import (
    BiolinkEntity,
    BiolinkPredicate,
    MetaAttributeDict,
    QualifierTypeID,
)
from retriever.types.trapi_pydantic import TierNumber

TripleName = str
OpHash = str


class OperationNode(NamedTuple):
    """A node that specifies information as separable by API."""

    prefixes: dict[str, list[str]]  # ID prefixes by API
    attributes: dict[str, list[MetaAttributeDict]]  # attributes by API


class UnhashedOperation(NamedTuple):
    """A single unit of operable subquerying, prior to having a hash attached."""

    tier: TierNumber
    subject: BiolinkEntity
    predicate: BiolinkPredicate
    object: BiolinkEntity
    api: str
    association: str | None = None
    attributes: list[MetaAttributeDict] | None = None
    qualifiers: dict[QualifierTypeID, list[str]] | None = None
    access_metadata: Any | None = None


class Operation(NamedTuple):
    """A single unit of operable subquerying."""

    hash: OpHash
    tier: TierNumber
    subject: BiolinkEntity
    predicate: BiolinkPredicate
    object: BiolinkEntity
    api: str
    association: str | None = None
    attributes: list[MetaAttributeDict] | None = None
    qualifiers: dict[QualifierTypeID, list[str]] | None = None
    access_metadata: Any | None = None


SortedOperations = dict[
    BiolinkEntity, dict[BiolinkPredicate, dict[BiolinkEntity, list[Operation]]]
]

FlatOperations = dict[OpHash, Operation]


class OperationTable(NamedTuple):
    """A table of operations and related node information."""

    operations_sorted: SortedOperations
    operations_flat: FlatOperations
    nodes: dict[BiolinkEntity, OperationNode]
