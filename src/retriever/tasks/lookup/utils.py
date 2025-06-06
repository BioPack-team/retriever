import asyncio
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
)

import bmt
from reasoner_pydantic import (
    CURIE,
    BiolinkEntity,
    Edge,
    HashableMapping,
    HashableSequence,
    KnowledgeGraph,
    LogEntry,
    QEdge,
    QueryGraph,
)
from reasoner_pydantic.shared import EdgeIdentifier

from retriever.tasks.lookup.branch import Branch
from retriever.type_defs import (
    AdjacencyGraph,
    QEdgeIDMap,
    SuperpositionHop,
)
from retriever.utils.logs import TRAPILogger

biolink = bmt.Toolkit()


def expand_qnode_categories(qg: QueryGraph, job_log: TRAPILogger) -> QueryGraph:
    """Ensure all nodes in qgraph have all descendent categories."""
    for qnode_id, qnode in qg.nodes.items():
        if qnode.categories is None or len(qnode.categories) == 0:
            qnode.categories = HashableSequence[BiolinkEntity](["biolink:NamedThing"])
        categories = set(qnode.categories)
        for category in set(qnode.categories):
            categories.update(biolink.get_descendants(category, formatted=True))

        if new_categories := categories.difference(set(qnode.categories)):
            job_log.debug(
                f"QNode {qnode_id}: Added descendent categories {new_categories}"
            )
        qnode.categories = HashableSequence[BiolinkEntity](list(categories))

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
                # TODO: handle exceptions in an appropriate manner
                raise e
            else:
                next_task = as_task(iterator)
                next_tasks[iterator] = next_task
                backmap[next_task] = iterator


# Probably not used, remove if so
# async def amap[InputType, **ArgType, OutputType](
#     iterable: AsyncIterable[InputType],
#     func: Callable[Concatenate[InputType, ArgType], Awaitable[OutputType]],
#     *args: ArgType.args,
#     **kwargs: ArgType.kwargs,
# ) -> AsyncGenerator[OutputType]:
#     """Apply the given func to every output of the given async iterable, yielding the results."""
#     async for item in iterable:
#         yield await func(item, *args, **kwargs)


async def get_subgraph(
    branch: Branch,
    key: tuple[CURIE, str],
    kedges: dict[SuperpositionHop, list[Edge]],
    kgraph: KnowledgeGraph,
) -> tuple[KnowledgeGraph, list[LogEntry]]:
    """Get a subgraph from a given set of kedges.

    Used to replace subquerying when a given hop has already been completed.
    """
    edges = kedges[key]
    curies = list[CURIE]()
    for edge in edges:
        if not branch.reversed:
            curies.append(edge.object)
        else:
            curies.append(edge.subject)

    kg = KnowledgeGraph(
        edges=HashableMapping(
            {EdgeIdentifier(str(hash(edge))): edge for edge in edges}
        ),
        nodes=HashableMapping({curie: kgraph.nodes[curie] for curie in curies}),
    )

    return kg, list[LogEntry]()
