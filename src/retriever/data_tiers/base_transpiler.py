from abc import ABC, abstractmethod
from typing import Any

from retriever.types.general import BackendResult
from retriever.types.trapi import QueryGraphDict


class Transpiler(ABC):
    """A class which handles converting a given TRAPI query graph to a target query language.

    Also handles back-converting the response to TRAPI.
    """

    def process_qgraph(
        self, qgraph: QueryGraphDict, *additional_qgraphs: QueryGraphDict
    ) -> Any:
        """Take in a TRAPI query graph and convert it to the target query language."""
        is_tier0 = isinstance(self, Tier0Transpiler)
        is_tier1 = isinstance(self, Tier1Transpiler)
        multihop = any(len(qg["nodes"]) > 2 for qg in (qgraph, *additional_qgraphs))  # noqa: PLR2004

        if not (is_tier0 or is_tier1):
            raise NotImplementedError(
                f"Class {self.__class__.__name__} must be abstract, no transpiling methods are implemented."
            )

        if multihop and not is_tier0:
            raise ValueError("Multi-hop query graphs are not supported by this class.")

        if (multihop and is_tier0) or not is_tier1:
            convert = self.convert_multihop
            batch_convert = self.convert_batch_multihop
        else:
            convert = self.convert_triple
            batch_convert = self.convert_batch_triple
        if len(additional_qgraphs) > 0:
            return batch_convert([qgraph, *additional_qgraphs])
        return convert(qgraph)

    @abstractmethod
    def convert_results(self, qgraph: QueryGraphDict, results: Any) -> BackendResult:
        """Convert the backend response back to TRAPI, mapping it to the qgraph."""


class Tier1Transpiler(Transpiler, ABC):
    """A transpiler class that handles converting single-hop query graphs."""

    @abstractmethod
    def convert_triple(self, qgraph: QueryGraphDict) -> Any:
        """Convert a single-hop query graph."""

    @abstractmethod
    def convert_batch_triple(self, qgraphs: list[QueryGraphDict]) -> Any:
        """Convert a list of single-hop query graphs."""


class Tier0Transpiler(Transpiler, ABC):
    """A transpiler class that handles converting multi-hop query graphs."""

    @abstractmethod
    def convert_multihop(self, qgraph: QueryGraphDict) -> Any:
        """Convert a multi-hop query graph."""

    @abstractmethod
    def convert_batch_multihop(self, qgraphs: list[QueryGraphDict]) -> Any:
        """Convert a list of multi-hop query graphs."""
