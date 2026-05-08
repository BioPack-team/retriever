from typing import Any, NamedTuple

from translator_tom import (
    Biolink,
    MetaAttribute,
)

from retriever.types.trapi_overrides import TierNumber

TripleName = str
OpHash = str


class OperationNode(NamedTuple):
    """A node that specifies information as separable by API."""

    prefixes: dict[str, list[str]]  # ID prefixes by API
    attributes: dict[str, list[MetaAttribute]]  # attributes by API


class UnhashedOperation(NamedTuple):
    """A single unit of operable subquerying, prior to having a hash attached."""

    tier: TierNumber
    subject: Biolink.Entity
    predicate: Biolink.Predicate
    object: Biolink.Entity
    api: str
    association: str | None = None
    attributes: list[MetaAttribute] | None = None
    qualifiers: dict[Biolink.Qualifier, list[str]] | None = None
    access_metadata: Any | None = None


class Operation(NamedTuple):
    """A single unit of operable subquerying."""

    hash: OpHash
    tier: TierNumber
    subject: Biolink.Entity
    predicate: Biolink.Predicate
    object: Biolink.Entity
    api: str
    association: str | None = None
    attributes: list[MetaAttribute] | None = None
    qualifiers: dict[Biolink.Qualifier, list[str]] | None = None
    access_metadata: Any | None = None


SortedOperations = dict[
    Biolink.Entity, dict[Biolink.Predicate, dict[Biolink.Entity, list[Operation]]]
]

FlatOperations = dict[OpHash, Operation]


class OperationTable(NamedTuple):
    """A table of operations and related node information."""

    operations_sorted: SortedOperations
    operations_flat: FlatOperations
    nodes: dict[Biolink.Entity, dict[TierNumber, OperationNode]]
