from __future__ import annotations

import base64
import re
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Self, TypeGuard, cast

import msgpack
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


def _decode_msgpack_base64(value: Any) -> Any | None:
    """Decode a base64 string containing msgpack data into a Python object.

    Returns None if value is missing or decoding fails.
    """
    if not isinstance(value, str | bytes | bytearray):
        return None
    try:
        raw_bytes = base64.b64decode(value)
        # raw=False decodes msgpack bytes into str for map keys/values
        return msgpack.unpackb(raw_bytes, raw=False)
    except Exception:
        return None


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
    source_id: str
    source_category: list[str] = field(default_factory=list)

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
            source_id=str(norm.get("source_id", "")),
            source_category=_to_str_list(norm.get("source_category")),
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
    anatomical_context_qualifier: list[str] = field(default_factory=list)
    causal_mechanism_qualifier: str | None = None
    species_context_qualifier: str | None = None
    object_aspect_qualifier: str | None = None
    object_direction_qualifier: str | None = None
    subject_aspect_qualifier: str | None = None
    subject_direction_qualifier: str | None = None
    qualifiers: list[str] = field(default_factory=list)
    FDA_regulatory_approvals: list[str] = field(default_factory=list)
    clinical_approval_status: str | None = None
    max_research_phase: str | None = None
    p_value: float | None = None
    adjusted_p_value: float | None = None
    number_of_cases: int | None = None
    dgidb_evidence_score: float | None = None
    dgidb_interaction_score: float | None = None
    has_supporting_studies_raw: str | None = None
    has_supporting_studies: Any | None = None

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
    def from_dict(  # noqa: PLR0913
        cls,
        edge_dict: Mapping[str, Any],
        binding: str,
        direction: str,
        prefix: str | None = None,
        edge_id_map: Mapping[str, str] | None = None,
        node_id_map: Mapping[str, str] | None = None,
    ) -> Self:
        """Parse an edge mapping into an Edge dataclass (handles versioned keys).

        Args:
            edge_dict: Raw edge data from Dgraph response
            binding: Edge binding (already converted to original ID by Node.from_dict)
            direction: Edge direction ('in' or 'out')
            prefix: Schema version prefix (e.g., 'vC_'), or None for no prefix
            edge_id_map: Optional mapping from normalized edge IDs to original IDs
            node_id_map: Optional mapping from normalized node IDs to original IDs

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

        # Convert node binding back to original ID
        node_binding = (
            node_id_map.get(normalized_node_binding, normalized_node_binding)
            if node_id_map
            else normalized_node_binding
        )

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

        # Decode supporting studies (base64 → msgpack → Python)
        raw_supporting = cast(str | None, norm.get("has_supporting_studies"))
        decoded_supporting = _decode_msgpack_base64(raw_supporting)

        return cls(
            binding=binding,
            direction="in" if direction == "in" else "out",
            predicate=str(norm.get("predicate", "")),
            node=Node.from_dict(
                node_val,
                binding=node_binding,
                prefix=prefix,
                edge_id_map=edge_id_map,
                node_id_map=node_id_map,
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
            anatomical_context_qualifier=_to_str_list(
                norm.get("anatomical_context_qualifier")
            ),
            causal_mechanism_qualifier=str(norm["causal_mechanism_qualifier"])
            if "causal_mechanism_qualifier" in norm
            else None,
            species_context_qualifier=str(norm["species_context_qualifier"])
            if "species_context_qualifier" in norm
            else None,
            object_aspect_qualifier=str(norm["object_aspect_qualifier"])
            if "object_aspect_qualifier" in norm
            else None,
            object_direction_qualifier=str(norm["object_direction_qualifier"])
            if "object_direction_qualifier" in norm
            else None,
            subject_aspect_qualifier=str(norm["subject_aspect_qualifier"])
            if "subject_aspect_qualifier" in norm
            else None,
            subject_direction_qualifier=str(norm["subject_direction_qualifier"])
            if "subject_direction_qualifier" in norm
            else None,
            qualifiers=_to_str_list(norm.get("qualifiers")),
            FDA_regulatory_approvals=_to_str_list(
                norm.get("FDA_regulatory_approvals")
            ),
            clinical_approval_status=str(norm["clinical_approval_status"])
            if "clinical_approval_status" in norm
            else None,
            max_research_phase=str(norm["max_research_phase"])
            if "max_research_phase" in norm
            else None,
            p_value=float(norm["p_value"]) if "p_value" in norm else None,
            adjusted_p_value=float(norm["adjusted_p_value"])
            if "adjusted_p_value" in norm
            else None,
            number_of_cases=int(norm["number_of_cases"])
            if "number_of_cases" in norm
            else None,
            dgidb_evidence_score=float(norm["dgidb_evidence_score"])
            if "dgidb_evidence_score" in norm
            else None,
            dgidb_interaction_score=float(norm["dgidb_interaction_score"])
            if "dgidb_interaction_score" in norm
            else None,
            has_supporting_studies_raw=raw_supporting,
            has_supporting_studies=decoded_supporting,
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
    full_name: str | None = None
    symbol: str | None = None
    synonym: list[str] = field(default_factory=list)
    xref: list[str] = field(default_factory=list)
    taxon: str | None = None
    chembl_availability_type: str | None = None
    chembl_black_box_warning: str | None = None
    chembl_natural_product: bool | None = None
    chembl_prodrug: bool | None = None

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
        node_id_map: Mapping[str, str] | None = None,
    ) -> Self:
        """Parse a node mapping into a Node dataclass (handles versioned keys).

        Args:
            data: Raw node data from Dgraph response
            binding: Node binding (normalized ID like 'n0')
            prefix: Schema version prefix (e.g., 'vC_'), or None for no prefix
            edge_id_map: Optional mapping from normalized edge IDs to original IDs
            node_id_map: Optional mapping from normalized node IDs to original IDs

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
                        edge_id_map.get(
                            normalized_edge_binding, normalized_edge_binding
                        )
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
                            node_id_map=node_id_map,
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
                        edge_id_map.get(
                            normalized_edge_binding, normalized_edge_binding
                        )
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
                            node_id_map=node_id_map,
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
            full_name=str(norm["full_name"]) if "full_name" in norm else None,
            symbol=str(norm["symbol"]) if "symbol" in norm else None,
            synonym=_to_str_list(norm.get("synonym")),
            xref=_to_str_list(norm.get("xref")),
            taxon=str(norm["taxon"]) if "taxon" in norm else None,
            chembl_availability_type=str(norm["chembl_availability_type"])
            if "chembl_availability_type" in norm
            else None,
            chembl_black_box_warning=str(norm["chembl_black_box_warning"])
            if "chembl_black_box_warning" in norm
            else None,
            chembl_natural_product=bool(norm["chembl_natural_product"])
            if "chembl_natural_product" in norm
            else None,
            chembl_prodrug=bool(norm["chembl_prodrug"])
            if "chembl_prodrug" in norm
            else None,
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
                            node_id_map=node_id_map,
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
