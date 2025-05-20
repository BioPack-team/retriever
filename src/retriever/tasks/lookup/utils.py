import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from typing import AsyncGenerator, Awaitable, Callable, Concatenate

from reasoner_pydantic import (
    CURIE,
    Edge,
    HashableMapping,
    KnowledgeGraph,
    Node,
    QEdge,
    QueryGraph,
)
from reasoner_pydantic.shared import EdgeIdentifier

from retriever.tasks.lookup.branch import Branch
from retriever.type_defs import (
    AdjacencyGraph,
    EdgeIDMap,
    SuperpositionHop,
)


def initialize_kgraph(qgraph: QueryGraph) -> KnowledgeGraph:
    """Initialize a knowledge graph, using nodes from the query graph."""
    kgraph = KnowledgeGraph()
    for qnode in qgraph.nodes.values():
        if qnode.ids is None:
            continue
        for curie in qnode.ids:
            kgraph.nodes[curie] = Node.model_validate(
                {
                    "categories": qnode.categories or [],
                    "attributes": [],
                }
            )
    return kgraph


def make_mappings(qg: QueryGraph) -> tuple[AdjacencyGraph, EdgeIDMap]:
    """Make an undirected QGraph representation in which edges are presented by their nodes."""
    agraph: AdjacencyGraph = {}
    edge_id_map: EdgeIDMap = {}
    for edge_id, edge in qg.edges.items():
        edge_id_map[edge_id] = edge
        edge_id_map[edge] = edge_id
        if edge.subject not in agraph:
            agraph[edge.subject] = dict[str, QEdge]()
        if edge.object not in agraph:
            agraph[edge.object] = dict[str, QEdge]()
        agraph[edge.subject][edge.object] = edge
        agraph[edge.object][edge.subject] = edge

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


async def amap[InputType, **ArgType, OutputType](
    iterable: AsyncIterable[InputType],
    func: Callable[Concatenate[InputType, ArgType], Awaitable[OutputType]],
    *args: ArgType.args,
    **kwargs: ArgType.kwargs,
) -> AsyncGenerator[OutputType]:
    """Apply the given func to every output of the given async iterable, yielding the results."""
    async for item in iterable:
        yield await func(item, *args, **kwargs)


async def get_subgraph(
    branch: Branch,
    key: tuple[CURIE, str],
    kedges: dict[SuperpositionHop, list[Edge]],
    kgraph: KnowledgeGraph,
) -> KnowledgeGraph:
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

    return KnowledgeGraph(
        edges=HashableMapping(
            {EdgeIdentifier(str(hash(edge))): edge for edge in edges}
        ),
        nodes=HashableMapping({curie: kgraph.nodes[curie] for curie in curies}),
    )
