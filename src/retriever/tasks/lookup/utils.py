import asyncio
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
)

import bmt
from reasoner_pydantic import (
    CURIE,
    BiolinkEntity,
    BiolinkPredicate,
    HashableSequence,
    QEdge,
    QueryGraph,
)

from retriever.tasks.lookup.branch import Branch
from retriever.types.general import (
    AdjacencyGraph,
    QEdgeIDMap,
    SuperpositionHop,
)
from retriever.types.trapi import EdgeDict, KnowledgeGraphDict, LogEntryDict
from retriever.utils.logs import TRAPILogger
from retriever.utils.trapi import hash_edge, hash_hex

biolink = bmt.Toolkit()


def expand_qgraph(qg: QueryGraph, job_log: TRAPILogger) -> QueryGraph:
    """Ensure all nodes in qgraph have all descendent categories.

    See https://biolink.github.io/biolink-model/categories.html
    """
    # biolink functions are already LRU cached :)
    for qnode_id, qnode in qg.nodes.items():
        categories = set(qnode.categories or {})
        new_categories = set[BiolinkEntity]()
        if len(categories) == 0:
            categories.add("biolink:NamedThing")
        # Can probably handle this in subquery dispatcher/abstraction layers
        # for category in categories:
        #     new_categories.update(biolink.get_descendants(category, formatted=True))

        if len(new_categories):
            job_log.info(
                f"QNode {qnode_id}: Added descendent categories {new_categories}."
            )
        qnode.categories = HashableSequence[BiolinkEntity](
            [*categories, *new_categories]
        )

    for qedge_id, qedge in qg.edges.items():
        predicates = set(qedge.predicates or {})
        new_predicates = set[BiolinkPredicate]()
        if len(predicates) == 0:
            predicates.add("biolink:related_to")
        # Can probably handle this in subquery dispatcher/abstraction layers
        # for predicate in predicates:
        #     new_predicates.update(biolink.get_descendants(predicate, formatted=True))

        if len(new_predicates):
            job_log.info(
                f"QEdge {qedge_id}: Added descendent predicates {new_predicates}."
            )

    return qg


def make_mappings(qg: QueryGraph) -> tuple[AdjacencyGraph, QEdgeIDMap]:
    """Make an undirected QGraph representation in which edges are presented by their nodes."""
    agraph: AdjacencyGraph = {}
    edge_id_map: QEdgeIDMap = {}
    for edge_id, edge in qg.edges.items():
        edge_id_map[edge] = edge_id
        if edge.subject not in agraph:
            agraph[edge.subject] = dict[str, list[QEdge]]()
        if edge.object not in agraph:
            agraph[edge.object] = dict[str, list[QEdge]]()
        if edge.object not in agraph[edge.subject]:
            agraph[edge.subject][edge.object] = list[QEdge]()
        if edge.subject not in agraph[edge.object]:
            agraph[edge.object][edge.subject] = list[QEdge]()
        agraph[edge.subject][edge.object].append(edge)
        agraph[edge.object][edge.subject].append(edge)

    return agraph, edge_id_map


async def await_next[T](iterator: AsyncIterator[T]) -> T:
    """Create a coroutine awaiting next value in iterator."""
    return await iterator.__anext__()


def as_task[T](iterator: AsyncIterator[T]) -> asyncio.Task[T]:
    """Create a task which resolves to the iterator's next result."""
    return asyncio.create_task(await_next(iterator))


async def merge_iterators[T](*iterators: AsyncIterator[T]) -> AsyncIterable[T]:
    """Merge multiple async iterators, yielding values as they are completed.

    Based on Alex Peter's solution: https://stackoverflow.com/a/76643550
    """
    next_tasks = {iterator: as_task(iterator) for iterator in iterators}
    backmap = {v: k for k, v in next_tasks.items()}

    while next_tasks:
        done, _ = await asyncio.wait(
            next_tasks.values(), return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            iterator = backmap[task]

            try:
                yield task.result()
            except StopAsyncIteration:
                del next_tasks[iterator]
                del backmap[task]
            except Exception as e:
                raise e
            else:
                next_task = as_task(iterator)
                next_tasks[iterator] = next_task
                backmap[next_task] = iterator


async def get_subgraph(
    branch: Branch,
    key: tuple[CURIE, str],
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
        nodes={curie: kgraph["nodes"][curie] for curie in curies},
    )

    return kg, list[LogEntryDict]()
