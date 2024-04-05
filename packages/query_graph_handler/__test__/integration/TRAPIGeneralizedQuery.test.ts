import TRAPIQueryHandler from '../../src/index';

describe('Testing TRAPI QueryHandler Generalized Query Handling', () => {
  const OneHopQuery = {
    nodes: {
      n0: {
        ids: ['PUBCHEM.COMPOUND:2662'],
        categories: ['biolink:SmallMolecule'],
      },
      n1: {
        categories: ['biolink:Gene'],
        ids: ['HGNC:9604'],
      },
    },
    edges: {
      e01: {
        subject: 'n0',
        object: 'n1',
      },
    },
  };

  describe('Testing query function', () => {
    test.skip('One hop has only nodes specified', async (done) => {
      const queryHandler = new TRAPIQueryHandler();
      queryHandler.setQueryGraph(OneHopQuery);
      await queryHandler.query();
      const res = queryHandler.getResponse();
      await expect(queryHandler.knowledgeGraph.kg).toHaveProperty('nodes');
      await expect(Object.keys(res.message.knowledge_graph.nodes).length).toBe(2);
      await expect(res.message.knowledge_graph.nodes).toHaveProperty('NCBIGene:5742');
      await expect(res.message.knowledge_graph.nodes).toHaveProperty('PUBCHEM.COMPOUND:2662');
      done();
    });
  });

  // const TwoHopQuery = {
  //     "nodes": {
  //         "n0": {
  //             "ids":["PUBCHEM.COMPOUND:2662"],
  //             "categories":["biolink:SmallMolecule"]
  //         },
  //         "n1": {
  //             "categories":["biolink:Disease"]
  //         },
  //             "n2": {
  //                 "categories":["biolink:Gene"],
  //                 "ids":["HGNC:9604"]
  //         }
  //     },
  //     "edges": {
  //         "e0": {
  //             "subject": "n0",
  //             "object": "n1"
  //         },
  //         "e1": {
  //             "subject": "n2",
  //             "object": "n1"
  //         }
  //     }
  // };

  // describe("Testing query function", () => {
  //     test("Two hop has edges connecting end to end", async () => {
  //         const queryHandler = new TRAPIQueryHandler.TRAPIQueryHandler();
  //         queryHandler.setQueryGraph(TwoHopQuery);
  //         await queryHandler.query();
  //         let res = queryHandler.getResponse();
  //         expect(Object.keys(res.message.knowledge_graph.nodes).length).toBeGreaterThan(2);
  //         expect(res.message.knowledge_graph.nodes).toHaveProperty("NCBIGene:5742");
  //         expect(res.message.knowledge_graph.nodes).toHaveProperty("CHEBI:41423");
  //         expect(res.message.knowledge_graph.edges).toHaveProperty("CHEBI:41423-biolink:treats-MONDO:0018874");
  //         expect(res.message.knowledge_graph.edges).toHaveProperty("NCBIGene:5742-biolink:related_to-MONDO:001887");
  //     })
  // })

  // const ThreeHopQuery = {
  //     "message": {
  //         "query_graph": {
  //             "nodes": {
  //                 "n0": {
  //                     "ids":["PUBCHEM.COMPOUND:2662"],
  //                     "categories":["biolink:SmallMolecule"]
  //                 },
  //                 "n1": {
  //                     "categories":["biolink:Disease"]
  //                },
  //                "n2": {
  //                     "categories":["biolink:Pathway"]
  //                },
  //                 "n3": {
  //                     "categories":["biolink:Gene"],
  //                        "ids":["HGNC:17947"]
  //                }
  //             },
  //             "edges": {
  //                 "e0": {
  //                     "subject": "n0",
  //                     "object": "n1"
  //                 },
  //                 "e1": {
  //                     "subject": "n1",
  //                     "object": "n2"
  //                 },
  //                 "e2": {
  //                     "subject": "n2",
  //                     "object": "n3"
  //                 }
  //             }
  //         }
  //     }
  // }

  // describe("Testing query function", () => {
  //     test("Three hop has edges connecting end to end", async () => {
  //         const queryHandler = new TRAPIQueryHandler.TRAPIQueryHandler();
  //         queryHandler.setQueryGraph(ThreeHopQuery);
  //         await queryHandler.query();
  //         let res = queryHandler.getResponse();
  //         expect(Object.keys(res.message.knowledge_graph.nodes).length).toBeGreaterThan(4);
  //         expect(res.message.knowledge_graph.nodes).toHaveProperty("NCBIGene:117145");
  //         expect(res.message.knowledge_graph.nodes).toHaveProperty("REACT:R-HSA-109704");
  //         expect(res.message.knowledge_graph.nodes).toHaveProperty("MONDO:0018874");
  //         expect(res.message.knowledge_graph.nodes).toHaveProperty("CHEBI:41423");
  //         expect(res.message.knowledge_graph.edges).toHaveProperty("CHEBI:41423-biolink:related_to-MONDO:0002974");
  //         expect(res.message.knowledge_graph.edges).toHaveProperty("MONDO:0002974-biolink:related_to-REACT:R-HSA-109704");
  //         expect(res.message.knowledge_graph.edges).toHaveProperty("REACT:R-HSA-109704-biolink:has_participant-NCBIGene:117145");
  //     })
  // })

  // const BroadCategoryQuery = {
  //     "message": {
  //         "query_graph": {
  //             "nodes": {
  //                 "n0": {
  //                     "categories": ["biolink:DiseaseOrPhenotypicFeature"]
  //                 },
  //                 "n1": {
  //                     "ids": ["HGNC:6284"],
  //                     "categories":["biolink:Gene"]
  //                 }
  //             },
  //             "edges": {
  //                 "e0": {
  //                     "subject": "n0",
  //                     "object": "n1"
  //                 }
  //             }
  //         }
  //     }
  // }

  // describe("Testing query function", () => {
  //     test("Broad category to known entity with all expected branching entities present", async () => {
  //         const queryHandler = new TRAPIQueryHandler.TRAPIQueryHandler();
  //         queryHandler.setQueryGraph(BroadCategoryQuery);
  //         await queryHandler.query();
  //         let res = queryHandler.getResponse();
  //         expect(res.message.knowledge_graph.nodes).toHaveProperty("NCBIGene:3778");
  //         expect(res.message.knowledge_graph.nodes).toHaveProperty("MONDO:0005247");
  //         expect(res.message.knowledge_graph.nodes).toHaveProperty("HP:0002465");
  //         expect(res.message.knowledge_graph.edges).toHaveProperty("MONDO:0005247-biolink:related_to-NCBIGene:3778");
  //         expect(res.message.knowledge_graph.edges).toHaveProperty("HP:0002465-biolink:related_to-NCBIGene:3778");
  //     })
  // })

  // const PredictQuery = {
  //     "message": {
  //         "query_graph": {
  //             "nodes": {
  //                 "n0": {
  //                     "ids": ["NCBIGene:3778"],
  //                     "categories": ["biolink:Gene"]
  //                 },
  //                 "n1": {
  //                     "categories": [
  //                         "biolink:Disease",
  //                         "biolink:BiologicalProcess",
  //                         "biolink:Pathway"
  //                     ]
  //                 }
  //             },
  //             "edges": {
  //                 "e01": {
  //                     "subject": "n0",
  //                     "object": "n1"
  //                 }
  //             }
  //         }
  //     }
  // }

  // describe("Testing query function", () => {
  //     test("Predict known entity to general category to have all nodes expected", async () => {
  //         const queryHandler = new TRAPIQueryHandler.TRAPIQueryHandler();
  //         queryHandler.setQueryGraph(PredictQuery);
  //         await queryHandler.query();
  //         let res = queryHandler.getResponse();
  //         expect(res.message.knowledge_graph.nodes).toHaveProperty("MONDO:0005030");
  //         expect(res.message.knowledge_graph.nodes).toHaveProperty("NCBIGene:3778");
  //         expect(res.message.knowledge_graph.nodes).toHaveProperty("GO:0001666");
  //         expect(res.message.knowledge_graph.nodes).toHaveProperty("REACT:R-HSA-109582");
  //     })
  // })
});
