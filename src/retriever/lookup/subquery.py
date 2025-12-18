import asyncio
import math
import time
from typing import Any

from opentelemetry import trace

from retriever.data_tiers import tier_manager
from retriever.data_tiers.base_transpiler import Transpiler
from retriever.lookup.branch import Branch
from retriever.types.general import BackendResult
from retriever.types.trapi import (
    BiolinkPredicate,
    Infores,
    KnowledgeGraphDict,
    LogEntryDict,
    QEdgeDict,
    QueryGraphDict,
)
from retriever.utils import biolink
from retriever.utils.logs import TRAPILogger
from retriever.utils.trapi import (
    append_aggregator_source,
    normalize_kgraph,
)

tracer = trace.get_tracer("lookup.execution.tracer")


def make_payloads(
    branch: Branch, qg: QueryGraphDict, job_log: TRAPILogger
) -> tuple[list[QueryGraphDict], list[Transpiler], list[Any]]:
    """Convert the existing branch edge to query payloads.

    Produces multiple if symmetric predicates are present.
    """
    edge_id = branch.current_edge
    current_edge = qg["edges"][edge_id]
    subject_node = qg["nodes"][current_edge["subject"]]
    object_node = qg["nodes"][current_edge["object"]]
    if not branch.reversed:
        subject_node["ids"] = [branch.input_curie]
    else:
        object_node["ids"] = [branch.input_curie]

    qgraph = QueryGraphDict(
        nodes={
            current_edge["subject"]: subject_node,
            current_edge["object"]: object_node,
        },
        edges={edge_id: current_edge},
    )

    subject_info = (
        subject_node.get("ids", None) or subject_node.get("categories", []) or []
    )
    if "biolink:NamedThing" in subject_info:
        subject_info = ["biolink:NamedThing"]
    object_info = (
        object_node.get("ids", None) or object_node.get("categories", []) or []
    )
    if "biolink:NamedThing" in object_info:
        object_info = ["biolink:NamedThing"]
    predicate_info = current_edge.get("predicates", ["biolink:related_to"]) or [
        "biolink:related_to"
    ]
    if "biolink:related_to" in predicate_info:
        predicate_info = ["biolink:related_to"]
    job_log.debug(
        f"Subquerying Tier 1 for {subject_info} -{predicate_info}-> {object_info}..."
    )

    # Check the symmetric predicate case
    symmetrics = list[BiolinkPredicate]()
    for predicate in current_edge.get("predicates") or [
        BiolinkPredicate("biolink:related_to")
    ]:
        if biolink.is_symmetric(str(predicate)):
            symmetrics.append(predicate)

    transpiler = tier_manager.get_transpiler(1)
    query_payload = transpiler.process_qgraph(qgraph)
    job_log.trace(str(query_payload))
    qgraphs = [qgraph]
    queries = [query_payload]
    transpilers = [transpiler]  # keep transpilers in case they store anything

    if len(symmetrics):
        symmetrics_info = (
            symmetrics
            if "biolink:related_to" not in symmetrics
            else ["biolink:related_to"]
        )
        job_log.debug("Symmetric predicates found, adding reverse subquery.")
        job_log.debug(
            f"Subquerying Tier 1 for {object_info} -{symmetrics_info}-> {subject_info}..."
        )
        reverse_edge = QEdgeDict(
            subject=current_edge["subject"],
            object=current_edge["object"],
            predicates=current_edge.get(
                "predicates", [BiolinkPredicate("biolink:related_to")]
            ),
        )
        if qualifiers := current_edge.get("qualifier_constraints"):
            reverse_edge["qualifier_constraints"] = (
                biolink.reverse_qualifier_constraints(qualifiers)
            )
        # BUG: doesn't reverse attribute constraints. But this is vanishingly unlikely.
        reverse_qg = QueryGraphDict(
            nodes={
                current_edge["subject"]: object_node,
                current_edge["object"]: subject_node,
            },
            edges={edge_id: reverse_edge},
        )
        transpiler = tier_manager.get_transpiler(1)  # Transpiler isn't singleton
        reverse_query_payload = transpiler.process_qgraph(reverse_qg)
        qgraphs.append(reverse_qg)
        queries.append(reverse_query_payload)
        transpilers.append(transpiler)

    return qgraphs, transpilers, queries


async def run_queries(
    qgraphs: list[QueryGraphDict],
    transpilers: list[Transpiler],
    query_payloads: list[Any],
    job_log: TRAPILogger,
) -> list[BackendResult]:
    """Given a set of query payloads, run them and combine their results."""
    query_driver = tier_manager.get_driver(1)

    subqueries = [
        asyncio.create_task(query_driver.run_query(payload))
        for payload in query_payloads
    ]

    response_records = await asyncio.gather(*subqueries, return_exceptions=True)
    results = list[BackendResult]()
    for i, record in enumerate(response_records):
        if isinstance(record, Exception):
            continue  # Exception will come from driver, which should have logged.
        result = transpilers[i].convert_results(qgraphs[i], record)

        # Add Retriever to the provenance chain
        for edge_id, edge in result["knowledge_graph"]["edges"].items():
            try:
                append_aggregator_source(edge, Infores("infores:retriever"))
            except ValueError:
                job_log.warning(f"Edge f{edge_id} has an invalid provenance chain.")

        results.append(result)

    # Have to do this for each
    for result in results:
        normalize_kgraph(
            result["knowledge_graph"], result["results"], result["auxiliary_graphs"]
        )

    return results


@tracer.start_as_current_span("subquery")
async def subquery(
    job_id: str, branch: Branch, qg: QueryGraphDict
) -> tuple[KnowledgeGraphDict, list[LogEntryDict]]:
    """Basic subquery function for retrieving from Tier 1.

    Intended to be replaced by subquery dispatcher in the future.

    Assumptions:
        Returns KEdges in the direction of the QEdge
    """
    try:
        job_log: TRAPILogger = TRAPILogger(job_id)
        start = time.time()

        # branch comes in execution direction
        # edge it refers to is in query direction
        qgraphs, transpilers, query_payloads = make_payloads(branch, qg, job_log)
        results = await run_queries(qgraphs, transpilers, query_payloads, job_log)

        if len(results) == 0:
            kg = KnowledgeGraphDict(nodes={}, edges={})
        else:
            kg = results[0]["knowledge_graph"]
        if len(results) > 1:
            for result in results[1:]:
                # We can only do this because we can guarantee both graphs are disjoint,
                # except for nodes, but any two nodes that are the same ID should be
                # exactly the same.
                # update_kgraph() should be used in all other cases
                kg["nodes"].update(result["knowledge_graph"]["nodes"])
                kg["edges"].update(result["knowledge_graph"]["edges"])
        end = time.time()
        job_log.debug(
            f"Subquery got {len(kg['edges'])} records in {math.ceil((end - start) * 1000)}ms"
        )

        return kg, job_log.get_logs()

    except asyncio.CancelledError:
        return KnowledgeGraphDict(nodes={}, edges={}), []
