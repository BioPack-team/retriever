import asyncio
import math
import time
from collections.abc import Callable
from typing import Any, NamedTuple, cast

from loguru import logger
from opentelemetry import trace

from retriever.data_tiers import tier_manager
from retriever.data_tiers.base_transpiler import Transpiler
from retriever.data_tiers.tier_1.elasticsearch.types import ESEdge
from retriever.lookup.branch import Branch, SuperpositionHop
from retriever.types.general import BackendResult
from retriever.types.trapi import (
    CURIE,
    BiolinkPredicate,
    EdgeDict,
    Infores,
    KnowledgeGraphDict,
    LogEntryDict,
    QEdgeDict,
    QueryGraphDict,
)
from retriever.utils import biolink
from retriever.utils.general import BatchedAction
from retriever.utils.logs import TRAPILogger
from retriever.utils.trapi import (
    append_aggregator_source,
    hash_edge,
    hash_hex,
    normalize_kgraph,
)

tracer = trace.get_tracer("lookup.execution.tracer")


class SubqContext(NamedTuple):
    """All the context required to generate a subquery."""

    job: str
    branch: Branch


class SubqueryDispatcher(BatchedAction):
    """A simple batcher for queries going to Tier 1/2.

    Currently does nothing intelligent for multi-tier use.
    Will have to refactor for this.
    """

    queue_delay: float = 0.05
    # Essentially should flush every interval
    batch_size: int = 100
    flush_time: float = 0

    subscriptions: dict[
        int, list[Callable[[tuple[KnowledgeGraphDict, list[LogEntryDict]]], None]]
    ]

    def __init__(self) -> None:
        """Initialize an instance."""
        self.subscriptions = {}
        super().__init__()

    async def subquery(
        self, job_id: str, branch: Branch
    ) -> tuple[KnowledgeGraphDict, list[LogEntryDict]]:
        """Make a subquery."""
        subq_id = hash((job_id, branch.superposition_id))
        job_log = TRAPILogger(job_id)
        try:
            start = time.time()
            job_log.debug(
                f"Subquery against Tier 1 {branch.superposition_id_to_name(branch.superposition_id[-2:])}"
            )
            future = asyncio.Future[tuple[KnowledgeGraphDict, list[LogEntryDict]]]()

            def return_result(
                subq_result: tuple[KnowledgeGraphDict, list[LogEntryDict]],
            ) -> None:
                self.subscriptions[subq_id].remove(return_result)
                # Have to clean up or else memory leak since superpositions are MANY
                if len(self.subscriptions[subq_id]) == 0:
                    del self.subscriptions[subq_id]

                end = time.time()
                kg, logs = subq_result
                job_log.debug(
                    f"Subquery got {len(kg['edges'])} records as part of batched query in {math.ceil((end - start) * 1000)}ms"
                )
                logs.extend(job_log.get_logs())

                future.set_result(subq_result)

            if subq_id not in self.subscriptions:
                self.subscriptions[subq_id] = []
            self.subscriptions[subq_id].append(return_result)

            self.put("batch_subquery", SubqContext(job=job_id, branch=branch))

            return await future
        except asyncio.CancelledError:
            return KnowledgeGraphDict(nodes={}, edges={}), []

    async def batch_subquery(self, batch: list[SubqContext]) -> None:
        """Produce query payloads and make them as a single batch query to the backend(s)."""
        loggers = dict[int, TRAPILogger]()

        query_mapping = list[tuple[SubqContext, QueryGraphDict, Transpiler]]()
        payloads = list[Any]()
        for subq in batch:
            subq_id = hash((subq.job, subq.branch.superposition_id))
            if subq_id not in loggers:
                loggers[subq_id] = TRAPILogger(subq.job)

            new_qgraphs, new_transpilers, new_payloads = self.make_payloads(
                subq.branch, loggers[subq_id]
            )
            payloads.extend(new_payloads)
            query_mapping.extend(
                [
                    (subq, qg, trans)
                    for qg, trans in zip(new_qgraphs, new_transpilers, strict=True)
                ]
            )

        start = time.time()
        logger.info(
            f"Subquerying Tier 1 with batch of {len(payloads)} subqueries (originating from batch of {len(batch)})..."
        )
        query_driver = tier_manager.get_driver(1)
        try:
            response_records = cast(
                list[list[ESEdge]], await query_driver.run_query(payloads)
            )
            split = time.time()
            logger.success(f"Got results in {math.ceil((split - start) * 1000)}ms")

            results = dict[int, BackendResult]()
            for record, (subq, qgraph, transpiler) in zip(
                response_records, query_mapping, strict=True
            ):
                subq_id = hash((subq.job, subq.branch.superposition_id))

                result = transpiler.convert_results(qgraph, record)

                # Add Retriever to the provenance chain
                for edge_id, edge in result["knowledge_graph"]["edges"].items():
                    try:
                        append_aggregator_source(edge, Infores("infores:retriever"))
                    except ValueError:
                        loggers[subq_id].warning(
                            f"Edge f{edge_id} has an invalid provenance chain."
                        )

                # Normalize the result kgraph for merging
                normalize_kgraph(
                    result["knowledge_graph"],
                    result["results"],
                    result["auxiliary_graphs"],
                )

                if subq_id not in results:
                    results[subq_id] = result
                    continue

                # We can only do this because we can guarantee both graphs are disjoint,
                # except for nodes, but any two nodes that are the same ID should be
                # exactly the same.
                # Additionally, we're doing nothing to update aux/results simple because
                # we know they'll never be used in Tier 1/2
                results[subq_id]["knowledge_graph"]["nodes"].update(
                    result["knowledge_graph"]["nodes"]
                )
                results[subq_id]["knowledge_graph"]["edges"].update(
                    result["knowledge_graph"]["edges"]
                )

            for subq_id, result in results.items():
                for callback in self.subscriptions.get(subq_id, []):
                    callback((result["knowledge_graph"], loggers[subq_id].get_logs()))
            end = time.time()
            logger.success(
                f"Transformed results and sent to original callers in {math.ceil((end - split) * 1000)}ms"
            )

        except Exception as e:
            for subq_id, job_log in loggers.items():
                job_log.with_exception(
                    "An unhandled error occurred in the query driver.", exception=e
                )
                for callback in self.subscriptions.get(subq_id, []):
                    callback(
                        (KnowledgeGraphDict(nodes={}, edges={}), job_log.get_logs())
                    )

    def make_payloads(
        self, branch: Branch, job_log: TRAPILogger
    ) -> tuple[list[QueryGraphDict], list[Transpiler], list[Any]]:
        """Convert the existing branch edge to query payloads.

        Produces multiple if symmetric predicates are present.
        """
        current_edge = branch.qgraph["edges"][branch.current_edge]
        subject_node = branch.qgraph["nodes"][current_edge["subject"]]
        object_node = branch.qgraph["nodes"][current_edge["object"]]
        if not branch.reversed:
            subject_node["ids"] = [branch.input_curie]
        else:
            object_node["ids"] = [branch.input_curie]

        qgraph = QueryGraphDict(
            nodes={
                current_edge["subject"]: subject_node,
                current_edge["object"]: object_node,
            },
            edges={branch.current_edge: current_edge},
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
            job_log.debug("Symmetric predicates found, adding reverse subquery.")
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
                edges={branch.current_edge: reverse_edge},
            )
            transpiler = tier_manager.get_transpiler(1)  # Transpiler isn't singleton
            reverse_query_payload = transpiler.process_qgraph(reverse_qg)
            qgraphs.append(reverse_qg)
            queries.append(reverse_query_payload)
            transpilers.append(transpiler)

        return qgraphs, transpilers, queries

    async def run_queries(
        self,
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
                job_log.with_exception(
                    "An unhandled error occurred in the query driver.", exception=record
                )
                continue
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

    @staticmethod
    async def get_subgraph(
        branch: Branch,
        key: SuperpositionHop,
        kedges: dict[SuperpositionHop, list[EdgeDict]],
        kgraph: KnowledgeGraphDict,
    ) -> tuple[KnowledgeGraphDict, list[LogEntryDict]]:
        """Get a subgraph from a given set of kedges.

        Used to replace subquerying when a given hop has already been completed.
        """
        edges = kedges[key]
        curies = list[str]()
        for edge in edges:
            if not branch.reversed:
                curies.append(edge["object"])
            else:
                curies.append(edge["subject"])

        kg = KnowledgeGraphDict(
            edges={hash_hex(hash_edge(edge)): edge for edge in edges},
            nodes={CURIE(curie): kgraph["nodes"][CURIE(curie)] for curie in curies},
        )

        return kg, list[LogEntryDict]()
