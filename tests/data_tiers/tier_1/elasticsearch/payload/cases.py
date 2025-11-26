from tests.data_tiers.tier_1.elasticsearch.payload.trapi_qgraphs import Q_GRAPHS_WITH_QUALIFIER_CONSTRAINTS, \
    QGRAPH_MULTIPLE_IDS, COMPREHENSIVE_QGRAPH, Q_GRAPH_WITH_ATTRIBUTE_CONSTRAINTS

Q_GRAPH_CASES = (
    "q_graph",
    [
        Q_GRAPH_WITH_ATTRIBUTE_CONSTRAINTS,
        *Q_GRAPHS_WITH_QUALIFIER_CONSTRAINTS,
        QGRAPH_MULTIPLE_IDS,
        COMPREHENSIVE_QGRAPH,
    ]
)

qualifier_case_ids = [
    f'with_qualifier_{case_num}' for case_num, _ in enumerate(Q_GRAPHS_WITH_QUALIFIER_CONSTRAINTS)
]

Q_GRAPH_CASES_IDS = ["with attribute", *qualifier_case_ids, "multiple ids", "comprehensive qgraph"]
