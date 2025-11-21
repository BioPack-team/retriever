from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Self, TypeGuard, cast

import orjson

from retriever.data_tiers.utils import (
    DINGO_KG_EDGE_TOPLEVEL_VALUES,
    DINGO_KG_NODE_TOPLEVEL_VALUES,
)
from retriever.utils import biolink

# Regex to find the node binding, ignoring an optional batch prefix like "q0_"
# It captures the part after the optional prefix and "node_"
NODE_KEY_PATTERN = re.compile(r"(?:q\d+_)?node_(\w+)")


def _strip_prefix(d: Mapping[str, Any], prefix: str | None) -> Mapping[str, Any]:
    if not prefix:
        return d
    return {(k.removeprefix(prefix)): v for k, v in d.items()}


# -----------------
# Dataclasses
# -----------------


# It represents the expected number of parts when splitting edge keys by underscore
# Format: "direction_edges_id" or "direction_edges-symmetric_id"
# Example: "in_edges_e0" splits to ["in", "edges", "e0"] = 3 parts
_EDGE_KEY_PARTS = 3


@dataclass(frozen=True, slots=True, kw_only=True)
class Source:
    """Represents a single source with its resource ID and role."""

    resource_id: str
    resource_role: str
    upstream_resource_ids: list[str] = field(default_factory=list)
    source_record_urls: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls, source_dict: Mapping[str, Any], prefix: str | None = None
    ) -> Self:
        """Parse a source mapping into a Source dataclass."""
        norm = _strip_prefix(source_dict, prefix)
        return cls(
            resource_id=str(norm.get("resource_id", "")),
            resource_role=str(norm.get("resource_role", "")),
            upstream_resource_ids=_to_str_list(norm.get("upstream_resource_ids")),
            source_record_urls=_to_str_list(norm.get("source_record_urls")),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Edge:
    """Represents a directed edge with its properties and a target node."""

    binding: str
    direction: Literal["in"] | Literal["out"]
    predicate: str
    node: Node
    agent_type: str | None = None
    knowledge_level: str | None = None
    publications: list[str] = field(default_factory=list)
    qualified_predicate: str | None = None
    predicate_ancestors: list[str] = field(default_factory=list)
    source_inforeses: list[str] = field(default_factory=list)
    subject_form_or_variant_qualifier: str | None = None
    disease_context_qualifier: str | None = None
    frequency_qualifier: str | None = None
    onset_qualifier: str | None = None
    sex_qualifier: str | None = None
    original_subject: str | None = None
    original_predicate: str | None = None
    original_object: str | None = None
    allelic_requirement: str | None = None
    update_date: str | None = None
    z_score: float | None = None
    has_evidence: list[str] = field(default_factory=list)
    has_confidence_score: float | None = None
    has_count: float | None = None
    has_total: float | None = None
    has_percentage: float | None = None
    has_quotient: float | None = None
    sources: list[Source] = field(default_factory=list)
    id: str | None = None
    category: list[str] = field(default_factory=list)

    def get_attributes(self) -> dict[str, Any]:
        """Return all fields which correspond to TRAPI attributes as a dict."""
        attrs = dict[str, Any]()
        for data_field in fields(self):
            if (
                data_field.name not in DINGO_KG_EDGE_TOPLEVEL_VALUES
                and not biolink.is_qualifier(data_field.name)
            ):
                attrs[data_field.name] = getattr(self, data_field.name)
        return attrs

    def get_qualifiers(self) -> dict[str, Any]:
        """Return all fields which correspond to TRAPI qualfiers as a dict."""
        qualifiers = dict[str, Any]()
        for data_field in fields(self):
            if biolink.is_qualifier(data_field.name):
                qualifiers[data_field.name] = getattr(self, data_field.name)
        return qualifiers

    @classmethod
    def from_dict(
        cls,
        edge_dict: Mapping[str, Any],
        binding: str,
        direction: str,
        prefix: str | None = None,
        edge_id_map: Mapping[str, str] | None = None,
    ) -> Self:
        """Parse an edge mapping into an Edge dataclass (handles versioned keys).

        Args:
            edge_dict: Raw edge data from Dgraph response
            binding: Edge binding (already converted to original ID by Node.from_dict)
            direction: Edge direction ('in' or 'out')
            prefix: Schema version prefix (e.g., 'vC_'), or None for no prefix
            edge_id_map: Optional mapping from normalized edge IDs to original IDs

        Returns:
            Parsed Edge instance with connected node
        """
        norm = _strip_prefix(edge_dict, prefix)

        node_val: Any = next(
            (v for k, v in norm.items() if k.startswith("node_")),
            cast(Mapping[str, Any], {}),
        )
        normalized_node_binding = next(
            (k.split("_", 1)[1] for k in norm if k.startswith("node_")), ""
        )

        # Note: We don't convert node binding here because node_id_map isn't passed down
        # The node binding conversion happens at the top level in DgraphResponse.parse()
        node_binding = normalized_node_binding

        # --- Parse sources ---
        sources_val = norm.get("sources")
        parsed_sources: list[Source] = []
        if isinstance(sources_val, list):
            safe_sources_list = cast(list[Any], sources_val)
            parsed_sources = [
                Source.from_dict(cast(Mapping[str, Any], source_item), prefix=prefix)
                for source_item in safe_sources_list
                if isinstance(source_item, Mapping)
            ]

        return cls(
            binding=binding,
            direction="in" if direction == "in" else "out",
            predicate=str(norm.get("predicate", "")),
            node=Node.from_dict(
                node_val,
                binding=node_binding,
                prefix=prefix,
                edge_id_map=edge_id_map,
            ),
            agent_type=str(norm["agent_type"]) if "agent_type" in norm else None,
            knowledge_level=str(norm["knowledge_level"])
            if "knowledge_level" in norm
            else None,
            publications=_to_str_list(norm.get("publications")),
            qualified_predicate=str(norm["qualified_predicate"])
            if "qualified_predicate" in norm
            else None,
            predicate_ancestors=_to_str_list(norm.get("predicate_ancestors")),
            source_inforeses=_to_str_list(norm.get("source_inforeses")),
            subject_form_or_variant_qualifier=str(
                norm["subject_form_or_variant_qualifier"]
            )
            if "subject_form_or_variant_qualifier" in norm
            else None,
            disease_context_qualifier=str(norm["disease_context_qualifier"])
            if "disease_context_qualifier" in norm
            else None,
            frequency_qualifier=str(norm["frequency_qualifier"])
            if "frequency_qualifier" in norm
            else None,
            onset_qualifier=str(norm["onset_qualifier"])
            if "onset_qualifier" in norm
            else None,
            sex_qualifier=str(norm["sex_qualifier"])
            if "sex_qualifier" in norm
            else None,
            original_subject=str(norm["original_subject"])
            if "original_subject" in norm
            else None,
            original_predicate=str(norm["original_predicate"])
            if "original_predicate" in norm
            else None,
            original_object=str(norm["original_object"])
            if "original_object" in norm
            else None,
            allelic_requirement=str(norm["allelic_requirement"])
            if "allelic_requirement" in norm
            else None,
            update_date=str(norm["update_date"]) if "update_date" in norm else None,
            z_score=float(norm["z_score"]) if "z_score" in norm else None,
            has_evidence=_to_str_list(norm.get("has_evidence")),
            has_confidence_score=float(norm["has_confidence_score"])
            if "has_confidence_score" in norm
            else None,
            has_count=float(norm["has_count"]) if "has_count" in norm else None,
            has_total=float(norm["has_total"]) if "has_total" in norm else None,
            has_percentage=float(norm["has_percentage"])
            if "has_percentage" in norm
            else None,
            has_quotient=float(norm["has_quotient"])
            if "has_quotient" in norm
            else None,
            sources=parsed_sources,
            id=str(norm["eid"]) if "eid" in norm else None,
            category=_to_str_list(norm.get("ecategory")),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Node:
    """Represents a node in the graph, with its properties and connected edges."""

    binding: str
    id: str
    name: str
    edges: list[Edge] = field(default_factory=list)
    category: list[str] = field(default_factory=list)
    in_taxon: list[str] = field(default_factory=list)
    information_content: float | None = None
    inheritance: str | None = None
    provided_by: list[str] = field(default_factory=list)
    description: str | None = None
    equivalent_identifiers: list[str] = field(default_factory=list)

    def get_attributes(self) -> dict[str, Any]:
        """Return all fields which correspond to TRAPI attributes as a dict."""
        attrs = dict[str, Any]()
        for data_field in fields(self):
            if data_field.name not in DINGO_KG_NODE_TOPLEVEL_VALUES:
                attrs[data_field.name] = getattr(self, data_field.name)
        return attrs

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        binding: str = "",
        prefix: str | None = None,
        edge_id_map: Mapping[str, str] | None = None,
    ) -> Self:
        """Parse a node mapping into a Node dataclass (handles versioned keys).

        Args:
            data: Raw node data from Dgraph response
            binding: Node binding (normalized ID like 'n0')
            prefix: Schema version prefix (e.g., 'vC_'), or None for no prefix
            edge_id_map: Optional mapping from normalized edge IDs to original IDs

        Returns:
            Parsed Node instance with edges having original bindings
        """
        norm = _strip_prefix(data, prefix)

        edges: list[Edge] = []
        for key, value in norm.items():
            # Parse incoming edges (where this node is the OBJECT)
            # Handle both "in_edges_e0" and "in_edges-symmetric_e0"
            if key.startswith("in_edges"):
                # Remove the "in_edges" prefix and any "-symmetric" suffix
                # Then extract the edge ID
                # Examples:
                #   "in_edges_e0" -> "in_edges" + "_e0" -> parts = ["in", "edges", "e0"]
                #   "in_edges-symmetric_e0" -> "in_edges-symmetric" + "_e0" -> parts = ["in", "edges-symmetric", "e0"]
                parts = key.split("_", 2)  # Split into max 3 parts
                if len(parts) >= _EDGE_KEY_PARTS:
                    # parts[2] is the edge ID (e.g., "e0")
                    normalized_edge_binding = parts[2]
                    # Convert back to original edge ID if mapping provided
                    edge_binding = (
                        edge_id_map.get(normalized_edge_binding, normalized_edge_binding)
                        if edge_id_map
                        else normalized_edge_binding
                    )
                else:
                    # Fallback for unexpected format
                    edge_binding = binding

                if isinstance(value, list):
                    edges.extend(
                        Edge.from_dict(
                            e,
                            binding=edge_binding,
                            direction="in",
                            prefix=prefix,
                            edge_id_map=edge_id_map,
                        )
                        for e in filter(_is_mapping, cast(list[Any], value))
                    )
            # Parse outgoing edges (where this node is the SUBJECT)
            # Handle both "out_edges_e0" and "out_edges-symmetric_e0"
            elif key.startswith("out_edges"):
                # Same logic as incoming edges
                parts = key.split("_", 2)  # Split into max 3 parts
                if len(parts) >= _EDGE_KEY_PARTS:
                    # parts[2] is the edge ID (e.g., "e0")
                    normalized_edge_binding = parts[2]
                    # Convert back to original edge ID if mapping provided
                    edge_binding = (
                        edge_id_map.get(normalized_edge_binding, normalized_edge_binding)
                        if edge_id_map
                        else normalized_edge_binding
                    )
                else:
                    # Fallback for unexpected format
                    edge_binding = binding

                if isinstance(value, list):
                    edges.extend(
                        Edge.from_dict(
                            e,
                            binding=edge_binding,
                            direction="out",
                            prefix=prefix,
                            edge_id_map=edge_id_map,
                        )
                        for e in filter(_is_mapping, cast(list[Any], value))
                    )

        return cls(
            binding=binding,
            id=str(norm.get("id", "")),
            name=str(norm.get("name", "")),
            edges=edges,
            category=_to_str_list(norm.get("category")),
            in_taxon=_to_str_list(norm.get("in_taxon")),
            information_content=float(norm["information_content"])
            if "information_content" in norm
            else None,
            inheritance=str(norm["inheritance"]) if "inheritance" in norm else None,
            provided_by=_to_str_list(norm.get("provided_by")),
            description=str(norm["description"]) if "description" in norm else None,
            equivalent_identifiers=_to_str_list(norm.get("equivalent_identifiers")),
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
        node_id_map: Mapping[str, str] | None = None,
        edge_id_map: Mapping[str, str] | None = None,
    ) -> Self:
        """Parse a raw Dgraph response into a structured DgraphResponse object.

        Args:
            raw: JSON string/bytes or already-parsed mapping from Dgraph.
            prefix: Explicit version prefix (e.g. 'v3_' or 'schema2025_'). If None, no stripping.
            node_id_map: Mapping from normalized node IDs (e.g., 'n0') to original IDs (e.g., 'n0_test')
            edge_id_map: Mapping from normalized edge IDs (e.g., 'e0') to original IDs (e.g., 'e0_test')

        Returns:
            Parsed DgraphResponse with original node/edge bindings restored
        """
        parsed_data: dict[str, Any] = (
            dict(raw) if isinstance(raw, Mapping) else orjson.loads(raw)
        )

        processed_data: dict[str, list[Node]] = {}
        for query_alias, results in parsed_data.items():
            match = NODE_KEY_PATTERN.match(query_alias)
            if not match:
                continue

            # Extract the normalized node binding from the key
            normalized_node_binding = match.group(1)

            # Convert back to original node ID if mapping provided
            node_binding = (
                node_id_map.get(normalized_node_binding, normalized_node_binding)
                if node_id_map
                else normalized_node_binding
            )

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
                        Node.from_dict(
                            node_data,
                            binding=node_binding,
                            prefix=prefix,
                            edge_id_map=edge_id_map,
                        )
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
