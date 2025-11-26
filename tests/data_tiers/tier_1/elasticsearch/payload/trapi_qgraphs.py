from retriever.types.trapi import QueryGraphDict
from tests.data_tiers.tier_1.elasticsearch.payload.cases import qg

SIMPLE_QGRAPH_0: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:4514"], "constraints": []},
        "n1": {"ids": ["UMLS:C1564592"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["subclass_of"],
            "attribute_constraints": [],
            "qualifier_constraints": [
                {
                    "qualifier_set": [
                            {
                                "qualifier_type_id": "biolink:object_aspect_qualifier",
                                "qualifier_value": "activity"
                            },
                    ]
                },
                {
                    "qualifier_set": [
                            {
                                "qualifier_type_id": "biolink:object_modifier_qualifier",
                                "qualifier_value": "increased"
                              },
                              {
                                "qualifier_type_id": "biolink:qualified_predicate",
                                "qualifier_value": "biolink:causes"
                              }
                    ]
                }
            ],
        },
    },
})
sample_generated_query = {
  "query": {
    "bool": {
      "filter": [
        { "terms": { "subject.id": ["UMLS:C1564592"] } },
        { "terms": { "object.id": ["CHEBI:4514"] } },
        { "terms": { "predicate_ancestors": ["subclass_of"] } }
      ],
      "should": [
        { "term": { "object_aspect_qualifier": "activity" } },
        {
          "bool": {
            "must": [
              { "term": { "object_modifier_qualifier": "increased" } },
              { "term": { "qualified_predicate": "causes" } }
            ]
          }
        }
      ],
      "minimum_should_match": 1
    }
  }
}
SIMPLE_QGRAPH_1: QueryGraphDict = qg({
        "nodes": {
                "n0": {"categories": ["biolink:Gene"]},
                "n1": {"categories": ["biolink:Disease"], "ids": ["UMLS:C0011847"]}
            },
            "edges": {
                "e01": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:causes"]
                }
            }
})
SIMPLE_QGRAPH_MULTIPLE_IDS: QueryGraphDict = qg({
    "nodes": {
        "n0": {"ids": ["CHEBI:3125", "CHEBI:53448"], "constraints": []},
        "n1": {"ids": ["UMLS:C0282090", "CHEBI:10119"], "constraints": []},
    },
    "edges": {
        "e0": {
            "object": "n0",
            "subject": "n1",
            "predicates": ["interacts_with"],
            "attribute_constraints": [],
            "qualifier_constraints": [],
        },
    },
})
