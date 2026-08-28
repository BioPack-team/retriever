from translator_tom.v1_6 import (
    Biolink,
    Curie,
    Message,
    QEdge,
    QEdgeID,
    QNode,
    QNodeID,
    QueryGraph,
)

from retriever.types.trapi import AsyncQuery, Parameters, Query

parameters = Parameters(tier=0)
submitter = "someone-looking-at-examples"
message = Message(
    query_graph=QueryGraph(
        nodes={
            QNodeID("n0"): QNode(categories=[Biolink.Entity("biolink:Gene")]),
            QNodeID("n2"): QNode(
                categories=[Biolink.Entity("biolink:Disease")],
                ids=[Curie("MONDO", "0011382")],
            ),
        },
        edges={
            QEdgeID("e02"): QEdge(
                subject=QNodeID("n0"),
                object=QNodeID("n2"),
                predicates=[Biolink.Predicate("biolink:causes")],
            ),
        },
    )
)

EXAMPLE_QUERY = Query(parameters=parameters, submitter=submitter, message=message)
EXAMPLE_ASYNCQUERY = AsyncQuery(
    parameters=parameters,
    submitter=submitter,
    message=message,
    callback="https://<your-callback-url-here>",
)
