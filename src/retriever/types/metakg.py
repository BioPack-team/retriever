from typing import Any, NamedTuple

from retriever.types.trapi import (
    BiolinkEntity,
    BiolinkPredicate,
    MetaAttributeDict,
    QualifierTypeID,
)

TripleName = str


class OperationNode(NamedTuple):
    """A node that specifies information as separable by API."""

    prefixes: dict[str, list[str]]  # ID prefixes by API
    attributes: dict[str, list[MetaAttributeDict]]  # attributes by API


class Operation(NamedTuple):
    """A single unit of operable subquerying."""

    subject: BiolinkEntity
    predicate: BiolinkPredicate
    object: BiolinkEntity
    api: str
    tier: int
    association: str | None = None
    tier_meta: Any | None = None
    attributes: list[MetaAttributeDict] | None = None
    qualifiers: dict[QualifierTypeID, list[str]] | None = None


class OperationTable(NamedTuple):
    """A table of operations and related node information."""

    operations: dict[TripleName, list[Operation]]
    nodes: dict[BiolinkEntity, OperationNode]
