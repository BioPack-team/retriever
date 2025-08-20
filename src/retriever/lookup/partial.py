from typing import override

import orjson
from reasoner_pydantic import (
    CURIE,
)

from retriever.types.general import KAdjacencyGraph
from retriever.types.trapi import (
    AnalysisDict,
    EdgeBindingDict,
    NodeBindingDict,
    ResultDict,
)


class Partial:
    """A partial result that can be back-propagated for branch reconciliation."""

    def __init__(
        self,
        node_bindings: list[tuple[str, CURIE]],
        edge_bindings: list[tuple[str, CURIE, CURIE]],
    ) -> None:
        """Initialize a partial result."""
        self.node_bindings: list[tuple[str, CURIE]] = node_bindings
        self.edge_bindings: set[tuple[str, CURIE, CURIE]] = set(edge_bindings)

    @override
    def __str__(self) -> str:
        return orjson.dumps(
            {
                "node_bindings": {
                    qnode: str(node) for qnode, node in sorted(self.node_bindings)
                },
                "edge_bindings": sorted(
                    (qedge_id, str(in_curie), str(out_curie))
                    for (qedge_id, in_curie, out_curie) in self.edge_bindings
                ),
            }
        ).decode()

    @override
    def __hash__(self) -> int:
        """Hash a Partial by its str representation."""
        nodes = sorted(f"{qnode_id}:{curie}" for qnode_id, curie in self.node_bindings)
        edges = sorted(
            f"{qedge_id}:{in_curie}:{out_curie}"
            for (qedge_id, in_curie, out_curie) in self.edge_bindings
        )
        return hash(f"{','.join(nodes)};{','.join(edges)}")

    @override
    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, Partial):
            return False
        return self.__hash__() == value.__hash__()

    def clone(self) -> "Partial":
        """Return a clone of the Partial with no mutable overlap."""
        return Partial(self.node_bindings.copy(), list(self.edge_bindings.copy()))

    def combine(self, other: "Partial") -> "Partial":
        """Return a new Partial that combines self and other."""
        return Partial(
            [*self.node_bindings, *other.node_bindings],
            [*self.edge_bindings, *other.edge_bindings],
        )

    def reconcile(self, other: "Partial") -> "Partial | None":
        """Reconcile a Partial with another Partial, returning their combination.

        Returns:
            Partial if the two are reconcilable; None if they weren't.
        """
        combined_nodes = dict(self.node_bindings)
        for qnode, node in other.node_bindings:
            if combined_nodes.get(qnode, node) != node:
                return None
            else:
                combined_nodes[qnode] = node

        return Partial(
            list(combined_nodes.items()), [*self.edge_bindings, *other.edge_bindings]
        )

    async def as_result(self, k_agraph: KAdjacencyGraph) -> ResultDict:
        """Return a result generated from the Partial's node and edge bindings."""
        return ResultDict(
            node_bindings={
                qnode: [NodeBindingDict(id=curie, attributes=[])]
                for qnode, curie in self.node_bindings
            },
            analyses=[
                AnalysisDict(
                    resource_id="infores:retriever",
                    edge_bindings={
                        qedge_id: [
                            EdgeBindingDict(id=kedge_id, attributes=[])
                            for kedge_id in k_agraph[qedge_id][in_id][out_id]
                        ]
                        for qedge_id, in_id, out_id in self.edge_bindings
                    },
                )
            ],
        )
        # return Result.model_construct(
        #     node_bindings=HashableMapping(
        #         {
        #             qnode: HashableSet(
        #                 {NodeBinding(id=curie, attributes=HashableSet())}
        #             )
        #             for qnode, curie in self.node_bindings
        #         }
        #     ),
        #     analyses=HashableSet(
        #         {
        #             Analysis(
        #                 resource_id=CURIE("infores:retriever"),
        #                 edge_bindings=HashableMapping(
        #                     {
        #                         kedge_key[0]: HashableSet(
        #                             {
        #                                 EdgeBinding(
        #                                     id=EdgeIdentifier(str(hash(kedge))),
        #                                     attributes=HashableSet(),
        #                                 )
        #                                 for kedge in k_agraph[kedge_key[0]][
        #                                     kedge_key[1]
        #                                 ][kedge_key[2]]
        #                             }
        #                         )
        #                         for kedge_key in self.edge_bindings
        #                     }
        #                 ),
        #             )
        #         }
        #     ),
        # )
        # return Result.model_validate(result_dict)
