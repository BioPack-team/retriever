from abc import ABC, abstractmethod
from typing import Any

from retriever.types.general import BackendResult
from retriever.types.trapi import QEdgeDict, QNodeDict, QueryGraphDict


class Transpiler(ABC):
    """A class which handles converting a given TRAPI query graph to a target query language.

    Also handles back-converting the response to TRAPI.
    """

    def process_qgraph(self, qgraph: QueryGraphDict) -> Any:
        """Take in a TRAPI query graph and convert it to the target query language."""
        batch = any(len(node.get("ids", []) or []) for node in qgraph["nodes"].values())
        if len(qgraph["nodes"]) == 2:  # noqa:PLR2004 This number will not change unless the laws of nature do :P
            edge = next(iter(qgraph["edges"].values()))
            subject_node = qgraph["nodes"][edge["subject"]]
            object_node = qgraph["nodes"][edge["object"]]
            return (
                self.convert_triple(subject_node, edge, object_node)
                if not batch
                else self.convert_batch_triple(subject_node, edge, object_node)
            )
        else:
            return (
                self._convert_multihop(qgraph)
                if not batch
                else self._convert_batch_multihop([qgraph])
            )

    @abstractmethod
    def convert_triple(
        self, in_node: QNodeDict, edge: QEdgeDict, out_node: QNodeDict
    ) -> Any:
        """Convert a single hop where each node has at most 1 ID.

        In Tier 1 cases, will be used outside of process_qgraph.
        """

    @abstractmethod
    def convert_batch_triple(
        self, in_node: QNodeDict, edge: QEdgeDict, out_node: QNodeDict
    ) -> Any:
        """Convert a single hop where either node may have multiple IDs.

        In Tier 1 cases, will be used outside of process_qgraph.
        """

    @abstractmethod
    def _convert_multihop(self, qgraph: QueryGraphDict) -> Any:
        """Convert a multi-hop graph query."""

    @abstractmethod
    def _convert_batch_multihop(self, qgraphs: list[QueryGraphDict]) -> Any:
        """Convert a list of multi-hop graph queries."""

    @abstractmethod
    def convert_results(self, qgraph: QueryGraphDict, results: Any) -> BackendResult:
        """Convert the backend response back to TRAPI, mapping it to the qgraph."""
