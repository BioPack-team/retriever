import json
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, cast

# -----------------
# Dataclasses
# -----------------

@dataclass(frozen=True)
class Node:
    """Node attributes returned by Dgraph (target or nested edge node)."""
    id: str
    name: str
    category: str
    all_names: list[str] = field(default_factory=list)
    all_categories: list[str] = field(default_factory=list)
    iri: str | None = None
    equivalent_curies: list[str] = field(default_factory=list)
    description: str | None = None
    publications: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InEdge:
    """Inbound edge with predicate, qualifiers, and a nested target Node."""
    predicate: str
    node: Node
    primary_knowledge_source: str | None = None
    knowledge_level: str | None = None
    agent_type: str | None = None
    kg2_ids: list[str] = field(default_factory=list)
    domain_range_exclusion: bool | None = None
    edge_id: str | None = None


@dataclass(frozen=True)
class NodeResult:
    """Root node result for each query key, including its inbound edges."""
    id: str
    name: str
    category: str
    in_edges: list[InEdge]
    all_names: list[str] = field(default_factory=list)
    all_categories: list[str] = field(default_factory=list)
    iri: str | None = None
    equivalent_curies: list[str] = field(default_factory=list)
    description: str | None = None
    publications: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DgraphResponse:
    """Parsed Dgraph response mapping query names to lists of NodeResult."""
    data: dict[str, list[NodeResult]]


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


def parse_node(node_dict: Mapping[str, Any]) -> Node:
    """Parse a node mapping into a Node dataclass."""
    return Node(
        id=str(node_dict.get("id", "")),
        name=str(node_dict.get("name", "")),
        category=str(node_dict.get("category", "")),
        all_names=_to_str_list(node_dict.get("all_names")),
        all_categories=_to_str_list(node_dict.get("all_categories")),
        iri=(str(node_dict["iri"]) if "iri" in node_dict else None),
        equivalent_curies=_to_str_list(node_dict.get("equivalent_curies")),
        description=(str(node_dict["description"]) if "description" in node_dict else None),
        publications=_to_str_list(node_dict.get("publications")),
    )


def parse_inedge(edge_dict: Mapping[str, Any]) -> InEdge:
    """Parse an edge mapping into an InEdge dataclass (including nested Node)."""
    node_val: Any = edge_dict.get("node", {})
    node_mapping: dict[str, Any] = {}
    if hasattr(node_val, "keys") and hasattr(node_val, "values"):
        with suppress(TypeError, ValueError):
            node_mapping = dict(node_val)

    return InEdge(
        predicate=str(edge_dict.get("predicate", "")),
        node=parse_node(node_mapping),
        primary_knowledge_source=(str(edge_dict["primary_knowledge_source"]) if "primary_knowledge_source" in edge_dict else None),
        knowledge_level=(str(edge_dict["knowledge_level"]) if "knowledge_level" in edge_dict else None),
        agent_type=(str(edge_dict["agent_type"]) if "agent_type" in edge_dict else None),
        kg2_ids=_to_str_list(edge_dict.get("kg2_ids")),
        domain_range_exclusion=(bool(edge_dict["domain_range_exclusion"]) if "domain_range_exclusion" in edge_dict else None),
        edge_id=(str(edge_dict["edge_id"]) if "edge_id" in edge_dict else None),
    )

def parse_noderesult(node_dict: Mapping[str, Any]) -> NodeResult:
    """Parse a root node mapping (with in_edges) into a NodeResult dataclass."""
    in_edges_raw: Any = node_dict.get("in_edges", [])
    in_edges: list[InEdge] = []
    if isinstance(in_edges_raw, list):
        items: list[Any] = cast(list[Any], in_edges_raw)
        mapped_edges: list[dict[str, Any]] = []
        for e in items:
            if hasattr(e, "keys") and hasattr(e, "values"):
                with suppress(TypeError, ValueError):
                    mapped_edges.append(dict(e))
        in_edges = [parse_inedge(e) for e in mapped_edges]
    return NodeResult(
        id=str(node_dict.get("id", "")),
        name=str(node_dict.get("name", "")),
        category=str(node_dict.get("category", "")),
        in_edges=in_edges,
        all_names=_to_str_list(node_dict.get("all_names")),
        all_categories=_to_str_list(node_dict.get("all_categories")),
        iri=(str(node_dict["iri"]) if "iri" in node_dict else None),
        equivalent_curies=_to_str_list(node_dict.get("equivalent_curies")),
        description=(str(node_dict["description"]) if "description" in node_dict else None),
        publications=_to_str_list(node_dict.get("publications")),
    )

def _parse_json_to_dict(raw_data: str | bytes | bytearray) -> dict[str, Any]:
    """Parse JSON string or bytes to dict."""
    if isinstance(raw_data, bytes | bytearray):
        tmp = json.loads(raw_data.decode("utf-8"))
    else:
        tmp = json.loads(raw_data)

    if not isinstance(tmp, dict):
        raise ValueError("Response is not a JSON object")
    return cast(dict[str, Any], tmp)

def _extract_data_map(obj_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract data map from response object."""
    if "data" in obj_dict:
        data_val = obj_dict["data"]
        if hasattr(data_val, "items"):
            data_map_dict: dict[str, Any] = {}
            with suppress(TypeError, ValueError):
                for k, v in cast(Mapping[str, Any], data_val).items():
                    data_map_dict[str(k)] = v
            return data_map_dict
        raise ValueError("'data' field is not a mapping")
    elif all(isinstance(v, list) for v in obj_dict.values()) or not obj_dict:
        return obj_dict
    else:
        raise ValueError("Response missing 'data' field or it is not a mapping")

def _parse_node_items(node_list: list[Any]) -> list[NodeResult]:
    """Parse a list of raw nodes into NodeResult objects."""
    node_results: list[NodeResult] = []
    for n_raw in node_list:
        n: Any = n_raw
        if n is not None and hasattr(n, "items"):
            with suppress(Exception):
                node_dict: dict[str, Any] = {}
                for k, v in cast(Mapping[str, Any], n).items():
                    node_dict[str(k)] = v
                node_results.append(parse_noderesult(node_dict))
    return node_results

def parse_response(raw: str | bytes | bytearray | Mapping[str, Any]) -> DgraphResponse:
    """Parse a Dgraph response into dataclasses.

    Supports both:
    - HTTP shape: {"data": {...}}
    - gRPC shape: {...} (top-level is already the data map)
    """
    # Load into a concrete dict for precise typing
    obj_dict: dict[str, Any]

    if isinstance(raw, str | bytes | bytearray):
        obj_dict = _parse_json_to_dict(raw)
    elif hasattr(raw, "items"):
        obj_dict = {}
        for k, v in raw.items():
            obj_dict[str(k)] = v
    else:
        raise TypeError(f"Unsupported response type: {type(raw)}. Expected str, bytes, or a mapping.")

    # Determine where the data map lives
    data_map_dict = _extract_data_map(obj_dict)

    # Parse each query result
    parsed: dict[str, list[NodeResult]] = {}
    for query_name, nodes in data_map_dict.items():
        if isinstance(nodes, list):
            parsed[str(query_name)] = _parse_node_items(cast(list[Any], nodes))
        else:
            parsed[str(query_name)] = []

    return DgraphResponse(data=parsed)
