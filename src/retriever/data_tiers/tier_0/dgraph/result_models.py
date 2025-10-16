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

    @classmethod
    def from_dict(
        cls, edge_dict: Mapping[str, Any], binding: str, direction: str
    ) -> Self:
        """Parse an edge mapping into an Edge dataclass."""
        # Find the nested node and its binding (e.g., "node_n1")
        node_val: Any = next(
            (v for k, v in edge_dict.items() if k.startswith("node_")),
            cast(Mapping[str, Any], {}),
        )
        node_binding = next(
            (k.split("_", 1)[1] for k in edge_dict if k.startswith("node_")), ""
        )

        return cls(
            binding=binding,
            direction="in" if direction == "in" else "out",
            predicate=str(edge_dict.get("predicate", "")),
            node=Node.from_dict(node_val, binding=node_binding),
            primary_knowledge_source=(
                str(edge_dict["primary_knowledge_source"])
                if "primary_knowledge_source" in edge_dict
                else None
            ),
            knowledge_level=(
                str(edge_dict["knowledge_level"])
                if "knowledge_level" in edge_dict
                else None
            ),
            agent_type=(
                str(edge_dict["agent_type"]) if "agent_type" in edge_dict else None
            ),
            kg2_ids=_to_str_list(edge_dict.get("kg2_ids")),
            domain_range_exclusion=(
                bool(edge_dict["domain_range_exclusion"])
                if "domain_range_exclusion" in edge_dict
                else None
            ),
            # The edge_id is the binding from the dynamic key (e.g., "e0" from "in_edges_e0").
            edge_id=binding,
            qualified_object_aspect=(
                str(edge_dict["qualified_object_aspect"])
                if "qualified_object_aspect" in edge_dict
                else None
            ),
            qualified_object_direction=(
                str(edge_dict["qualified_object_direction"])
                if "qualified_object_direction" in edge_dict
                else None
            ),
            qualified_predicate=(
                str(edge_dict["qualified_predicate"])
                if "qualified_predicate" in edge_dict
                else None
            ),
            publications_info=(
                str(edge_dict["publications_info"])
                if "publications_info" in edge_dict
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
    def from_dict(cls, node_dict: Mapping[str, Any], binding: str) -> Self:
        """Parse a node mapping into a Node dataclass, normalizing dynamic keys."""
        edges: list[Edge] = []
        for key, value in node_dict.items():
            if key.startswith("in_edges_"):
                edge_binding = key.split("_", 2)[2]
                if isinstance(value, list):
                    # Explicitly cast the list to satisfy the static checker
                    edges.extend(
                        Edge.from_dict(e, binding=edge_binding, direction="in")
                        for e in filter(_is_mapping, cast(list[Any], value))
                    )
            elif key.startswith("out_edges_"):
                edge_binding = key.split("_", 2)[2]
                if isinstance(value, list):
                    # Explicitly cast the list to satisfy the static checker
                    edges.extend(
                        Edge.from_dict(e, binding=edge_binding, direction="out")
                        for e in filter(_is_mapping, cast(list[Any], value))
                    )

        return cls(
            binding=binding,
            id=str(node_dict.get("id", "")),
            name=str(node_dict.get("name", "")),
            category=str(node_dict.get("category", "")),
            edges=edges,
            all_names=_to_str_list(node_dict.get("all_names")),
            all_categories=_to_str_list(node_dict.get("all_categories")),
            iri=(str(node_dict["iri"]) if "iri" in node_dict else None),
            equivalent_curies=_to_str_list(node_dict.get("equivalent_curies")),
            description=(
                str(node_dict["description"]) if "description" in node_dict else None
            ),
            publications=_to_str_list(node_dict.get("publications")),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DgraphResponse:
    """Parsed Dgraph response mapping query names to lists of Node objects."""

    data: dict[str, list[Node]]

    @classmethod
    def parse(cls, raw: str | bytes | bytearray | Mapping[str, Any]) -> Self:
        """Parse a raw Dgraph response into a structured DgraphResponse object."""
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
                prefix = query_alias.split("_")[0]
                if prefix.startswith("q") and prefix[1:].isdigit():
                    query_key = prefix

            if query_key not in processed_data:
                processed_data[query_key] = []

            if not isinstance(results, list):
                continue

            # Explicitly cast the list to satisfy the static checker
            for node_data in filter(_is_mapping, cast(list[Any], results)):
                with suppress(Exception):
                    processed_data[query_key].append(
                        Node.from_dict(node_data, binding=node_binding)
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
