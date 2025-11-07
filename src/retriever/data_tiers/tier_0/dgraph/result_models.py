from __future__ import annotations

import json
import re
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Literal, Self, TypeGuard, cast

# Regex to find the node binding, ignoring an optional batch prefix like "q0_"
# It captures the part after the optional prefix and "node_"
NODE_KEY_PATTERN = re.compile(r"(?:q\d+_)?node_(\w+)")


def _strip_prefix(d: Mapping[str, Any], prefix: str | None) -> Mapping[str, Any]:
    if not prefix:
        return d
    return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in d.items() }

# -----------------
# Dataclasses
# -----------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Edge:
    """Represents a directed edge with its properties and a target node."""

    binding: str
    direction: Literal["in"] | Literal["out"]
    predicate: str
    node: Node
    primary_knowledge_source: str | None = None
    knowledge_level: str | None = None
    agent_type: str | None = None
    kg2_ids: list[str] = field(default_factory=list)
    domain_range_exclusion: bool | None = None
    edge_id: str | None = None
    qualified_object_aspect: str | None = None
    qualified_object_direction: str | None = None
    qualified_predicate: str | None = None
    publications_info: str | None = None
    publications: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        edge_dict: Mapping[str, Any],
        binding: str,
        direction: str,
        prefix: str | None = None,
    ) -> Self:
        """Parse an edge mapping into an Edge dataclass (handles versioned keys)."""
        norm = _strip_prefix(edge_dict, prefix)

        node_val: Any = next(
            (v for k, v in norm.items() if k.startswith("node_")),
            cast(Mapping[str, Any], {}),
        )
        node_binding = next(
            (k.split("_", 1)[1] for k in norm if k.startswith("node_")), ""
        )

        return cls(
            binding=binding,
            direction="in" if direction == "in" else "out",
            predicate=str(norm.get("predicate", "")),
            node=Node.from_dict(node_val, binding=node_binding, prefix=prefix),
            primary_knowledge_source=(
                str(norm["primary_knowledge_source"])
                if "primary_knowledge_source" in norm
                else None
            ),
            knowledge_level=(
                str(norm["knowledge_level"]) if "knowledge_level" in norm else None
            ),
            agent_type=(
                str(norm["agent_type"]) if "agent_type" in norm else None
            ),
            kg2_ids=_to_str_list(norm.get("kg2_ids")),
            domain_range_exclusion=(
                bool(norm["domain_range_exclusion"])
                if "domain_range_exclusion" in norm
                else None
            ),
            edge_id=binding,
            qualified_object_aspect=(
                str(norm["qualified_object_aspect"])
                if "qualified_object_aspect" in norm
                else None
            ),
            qualified_object_direction=(
                str(norm["qualified_object_direction"])
                if "qualified_object_direction" in norm
                else None
            ),
            qualified_predicate=(
                str(norm["qualified_predicate"])
                if "qualified_predicate" in norm
                else None
            ),
            publications=_to_str_list(norm.get("publications")),
            publications_info=(
                str(norm["publications_info"])
                if "publications_info" in norm
                else None
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Node:
    """Represents a node in the graph, with its properties and connected edges."""

    binding: str
    id: str
    name: str
    category: str
    edges: list[Edge] = field(default_factory=list)
    all_names: list[str] = field(default_factory=list)
    all_categories: list[str] = field(default_factory=list)
    iri: str | None = None
    equivalent_curies: list[str] = field(default_factory=list)
    description: str | None = None
    publications: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, node_dict: Mapping[str, Any], binding: str, prefix: str | None = None) -> Self:
        """Parse a node mapping into a Node dataclass (handles versioned keys)."""
        norm = _strip_prefix(node_dict, prefix)

        edges: list[Edge] = []
        for key, value in norm.items():
            if key.startswith("in_edges_"):
                edge_binding = key.split("_", 2)[2]
                if isinstance(value, list):
                    edges.extend(
                        Edge.from_dict(e, binding=edge_binding, direction="in", prefix=prefix)
                        for e in filter(_is_mapping, cast(list[Any], value))
                    )
            elif key.startswith("out_edges_"):
                edge_binding = key.split("_", 2)[2]
                if isinstance(value, list):
                    edges.extend(
                        Edge.from_dict(e, binding=edge_binding, direction="out", prefix=prefix)
                        for e in filter(_is_mapping, cast(list[Any], value))
                    )

        return cls(
            binding=binding,
            id=str(norm.get("id", "")),
            name=str(norm.get("name", "")),
            category=str(norm.get("category", "")),
            edges=edges,
            all_names=_to_str_list(norm.get("all_names")),
            all_categories=_to_str_list(norm.get("all_categories")),
            iri=(str(norm["iri"]) if "iri" in norm else None),
            equivalent_curies=_to_str_list(norm.get("equivalent_curies")),
            description=(str(norm["description"]) if "description" in norm else None),
            publications=_to_str_list(norm.get("publications")),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DgraphResponse:
    """Parsed Dgraph response mapping query names to lists of Node objects."""

    data: dict[str, list[Node]]

    @classmethod
    def parse(
        cls,
        raw: str | bytes | bytearray | Mapping[str, Any],
        *,
        prefix: str | None = None,
    ) -> Self:
        """Parse a raw Dgraph response into a structured DgraphResponse object.

        Args:
            raw: JSON string/bytes or already-parsed mapping from Dgraph.
            prefix: Explicit version prefix (e.g. 'v3_' or 'schema2025_'). If None, no stripping.
        """
        # Use ternary operator per ruff (SIM108)
        parsed_data: dict[str, Any] = (
            dict(raw) if isinstance(raw, Mapping) else json.loads(raw)
        )

        processed_data: dict[str, list[Node]] = {}
        # The top-level keys are the query aliases (e.g., "q0_node_n0")
        for query_alias, results in parsed_data.items():
            match = NODE_KEY_PATTERN.match(query_alias)
            if not match:
                continue

            # Extract the TRAPI node binding (e.g., "n0") from the key
            node_binding = match.group(1)

            # Determine the batch query key (e.g., "q0")
            query_key = "q0"
            if "_" in query_alias:
                prefix_part = query_alias.split("_")[0]
                if prefix_part.startswith("q") and prefix_part[1:].isdigit():
                    query_key = prefix_part

            if query_key not in processed_data:
                processed_data[query_key] = []

            if not isinstance(results, list):
                continue

            for node_data in filter(_is_mapping, cast(list[Any], results)):
                with suppress(Exception):
                    processed_data[query_key].append(
                        Node.from_dict(node_data, binding=node_binding, prefix=prefix)
                    )

        return cls(data=processed_data)


# -----------------
# Parsing helpers
# -----------------


def _to_str_list(value: Any) -> list[str]:
    """Coerce a scalar or list value to a list[str]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in cast(list[Any], value)]
    # Accept single strings/bytes by wrapping
    if isinstance(value, str | bytes | bytearray):
        return [str(value)]
    return []


def _is_mapping(item: Any) -> TypeGuard[Mapping[str, Any]]:
    """A TypeGuard to check if an item is a mapping."""
    return isinstance(item, Mapping)
