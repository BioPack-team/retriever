from retriever.types.trapi import (
    CURIE,
    URL,
    AsyncQueryDict,
    BiolinkEntity,
    BiolinkPredicate,
    MessageDict,
    ParametersDict,
    QEdgeDict,
    QEdgeID,
    QNodeDict,
    QNodeID,
    QueryDict,
    QueryGraphDict,
)

EXAMPLE_QUERY = QueryDict(
    parameters=ParametersDict(tiers=[1]),
    submitter="bte-dev-tester-manual",
    message=MessageDict(
        query_graph=QueryGraphDict(
            nodes={
                QNodeID("n0"): QNodeDict(categories=[BiolinkEntity("biolink:Gene")]),
                QNodeID("n2"): QNodeDict(
                    categories=[BiolinkEntity("biolink:Disease")],
                    ids=[CURIE("MONDO:0011382")],
                ),
            },
            edges={
                QEdgeID("e02"): QEdgeDict(
                    subject=QNodeID("n0"),
                    object=QNodeID("n2"),
                    predicates=[BiolinkPredicate("biolink:causes")],
                ),
            },
        )
    ),
)

EXAMPLE_ASYNCQUERY = AsyncQueryDict(
    **EXAMPLE_QUERY, callback=URL("https://<your-callback-url-here>")
)
