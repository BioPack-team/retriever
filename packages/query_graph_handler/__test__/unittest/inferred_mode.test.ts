import TRAPIQueryHandler, { TrapiQueryGraph, TrapiResponse, TrapiResult } from '../../src/index';
import path from 'path';
import fs from 'fs';
const smartAPIPAth = path.resolve(__dirname, '../../../bte-trapi/data/smartapi_specs.json');
const predicatesPath = path.resolve(__dirname, '../../../bte-trapi/data/predicates.json');
import _ from 'lodash';
import { StampedLog, TrapiLog } from '@biothings-explorer/utils';
import InferredQueryHandler, { CombinedResponse } from '../../src/inferred_mode/inferred_mode';
import { MatchedTemplate } from '../../src/inferred_mode/template_lookup';

const queryGraph1 = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, '../data/chemicalEntity_treats_acanthosis.json'), { encoding: 'utf8' }),
).message.query_graph as TrapiQueryGraph;

describe('Test InferredQueryHandler', () => {
  test('queryIsValid', () => {
    const noCategories = _.cloneDeep(queryGraph1);
    Object.values(noCategories.nodes).forEach((node) => {
      node.categories = undefined;
    });

    const queryGraphHandler = new TRAPIQueryHandler();

    let handler = new InferredQueryHandler(queryGraphHandler, noCategories, [], {}, smartAPIPAth, predicatesPath, true);

    expect(handler.queryIsValid).toBeFalsy();

    const missingID = _.cloneDeep(queryGraph1);
    Object.values(missingID.nodes).forEach((node) => {
      node.ids = undefined;
    });

    handler = new InferredQueryHandler(queryGraphHandler, missingID, [], {}, smartAPIPAth, predicatesPath, true);

    expect(handler.queryIsValid).toBeFalsy();

    const missingPredicate = _.cloneDeep(queryGraph1);
    Object.values(missingPredicate.edges).forEach((edge) => {
      edge.predicates = undefined;
    });

    handler = new InferredQueryHandler(queryGraphHandler, missingPredicate, [], {}, smartAPIPAth, predicatesPath, true);

    expect(handler.queryIsValid).toBeFalsy();

    const missingPredicateType2 = _.cloneDeep(queryGraph1);
    Object.values(missingPredicateType2.edges).forEach((edge) => {
      edge.predicates = [];
    });

    handler = new InferredQueryHandler(
      queryGraphHandler,
      missingPredicateType2,
      [],
      {},
      smartAPIPAth,
      predicatesPath,
      true,
    );

    expect(handler.queryIsValid).toBeFalsy();

    const tooManyIDs = _.cloneDeep(queryGraph1);
    Object.values(tooManyIDs.nodes).forEach((node) => {
      node.ids = ['id0', 'id1'];
    });

    handler = new InferredQueryHandler(queryGraphHandler, tooManyIDs, [], {}, smartAPIPAth, predicatesPath, true);

    expect(handler.queryIsValid).toBeFalsy();

    const multiplePredicates = _.cloneDeep(queryGraph1);
    Object.values(multiplePredicates.edges).forEach((edge) => {
      edge.predicates = ['pred0', 'pred1'];
    });

    handler = new InferredQueryHandler(
      queryGraphHandler,
      multiplePredicates,
      [],
      {},
      smartAPIPAth,
      predicatesPath,
      true,
    );

    expect(handler.queryIsValid).toBeFalsy();

    handler = new InferredQueryHandler(
      queryGraphHandler,
      queryGraph1,
      [],
      { smartAPIID: 'test', teamName: 'test' },
      smartAPIPAth,
      predicatesPath,
      true,
    );

    expect(handler.queryIsValid).toBeFalsy();

    handler = new InferredQueryHandler(queryGraphHandler, queryGraph1, [], {}, smartAPIPAth, predicatesPath, true);

    expect(handler.queryIsValid).toBeTruthy();
  });

  test('getQueryParts', () => {
    const queryGraphHandler = new TRAPIQueryHandler();
    const handler = new InferredQueryHandler(
      queryGraphHandler,
      queryGraph1,
      [],
      {},
      smartAPIPAth,
      predicatesPath,
      true,
    );

    const queryParts = handler.getQueryParts();
    expect(queryParts).toHaveProperty('qEdgeID');
    expect(queryParts).toHaveProperty('qEdge');
    expect(queryParts).toHaveProperty('qSubject');
    expect(queryParts).toHaveProperty('qObject');
    const { qEdgeID, qEdge, qSubject, qObject } = queryParts;
    expect(qEdgeID).toEqual('e01');
    expect(qEdge).toStrictEqual(queryGraph1.edges.e01);
    expect(qSubject).toStrictEqual(queryGraph1.nodes.n01);
    expect(qObject).toStrictEqual(queryGraph1.nodes.n02);
  });

  describe('findTemplates', () => {
    test('find templates', async () => {
      // may need updates if biolink changes significantly
      const queryGraphHandler = new TRAPIQueryHandler();
      const handler = new InferredQueryHandler(
        queryGraphHandler,
        queryGraph1,
        [],
        {},
        smartAPIPAth,
        predicatesPath,
        true,
      );
      const { qEdgeID, qEdge, qSubject, qObject } = handler.getQueryParts();

      const templates = await handler.findTemplates(qEdge, qSubject, qObject);
      // console.log(templates);
      expect(templates.length).toBeGreaterThan(1);
      expect(
        templates.every(({ queryGraph }) => {
          return 'creativeQuerySubject' in queryGraph.nodes && 'creativeQueryObject' in queryGraph.nodes;
        }),
      ).toBeTruthy();
    });

    test("don't find templates", async () => {
      // may need updates if biolink changes significantly
      const badQuery = _.cloneDeep(queryGraph1);
      badQuery.nodes.n01.categories = ['biolink:disease'];
      const logs: StampedLog[] = [];

      const queryGraphHandler = new TRAPIQueryHandler();
      const handler = new InferredQueryHandler(
        queryGraphHandler,
        badQuery,
        logs,
        {},
        smartAPIPAth,
        predicatesPath,
        true,
      );
      const { qEdgeID, qEdge, qSubject, qObject } = handler.getQueryParts();

      const templates = await handler.findTemplates(qEdge, qSubject, qObject);
      expect(templates).toHaveLength(0);
      expect(logs[logs.length - 1].message).toMatch('Your query terminates');
    });
  });

  test('createQueries', async () => {
    const spy = jest.spyOn(InferredQueryHandler.prototype, 'findTemplates');

    const templates: MatchedTemplate[] = [
      {
        queryGraph: {
          nodes: {
            creativeQueryObject: {
              categories: [],
              ids: [],
            },
            creativeQuerySubject: {
              categories: [],
            },
          },
          edges: {
            e01: {
              subject: 'creativeQuerySubject',
              object: 'creativeQueryObject',
              predicates: [],
              knowledge_type: 'inferred',
            },
          },
        },
        template: 'Chem-treats-DoP.json',
      },
    ];

    const queryGraphHandler = new TRAPIQueryHandler();
    const handler = new InferredQueryHandler(
      queryGraphHandler,
      queryGraph1,
      [],
      {},
      smartAPIPAth,
      predicatesPath,
      true,
    );
    const { qEdgeID, qEdge, qSubject, qObject } = handler.getQueryParts();
    const subQueries = await handler.createQueries(qEdge, qSubject, qObject);
    expect(spy).toHaveBeenCalled();

    subQueries.forEach(({ template, queryGraph }) => {
      expect(queryGraph.nodes.creativeQuerySubject.categories).toContain('biolink:ChemicalEntity');
      expect(queryGraph.nodes.creativeQueryObject.categories).toContain('biolink:Disease');
      expect(queryGraph.nodes.creativeQueryObject.ids).toContain('MONDO:0007035');
      expect(queryGraph.nodes.creativeQuerySubject.ids).toBeUndefined();
    });

    // check that undefined deletion works
    const handler2 = new InferredQueryHandler(
      queryGraphHandler,
      {
        nodes: {
          n02: {
            categories: [],
            ids: [],
          },
          n01: {
            categories: [],
          },
        },
        edges: {
          e01: {
            subject: 'n01',
            object: 'n02',
            predicates: [],
            knowledge_type: 'inferred',
          },
        },
      },
      [],
      {},
      smartAPIPAth,
      predicatesPath,
      true,
    );
    const { qEdgeID: qEdgeID1, qEdge: qEdge1, qSubject: qSubject1, qObject: qObject1 } = handler2.getQueryParts();
    spy.mockResolvedValueOnce(templates);
    const subQueries1 = await handler2.createQueries(qEdge1, qSubject1, qObject1);
    expect(spy).toHaveBeenCalled();
    expect(subQueries1[0].queryGraph.nodes.creativeQuerySubject.categories).toBeUndefined();
    expect(subQueries1[0].queryGraph.nodes.creativeQueryObject.categories).toBeUndefined();
    expect(subQueries1[0].queryGraph.nodes.creativeQuerySubject.ids).toBeUndefined();
    expect(subQueries1[0].queryGraph.nodes.creativeQueryObject.ids).toBeUndefined();
  });

  test('combineResponse', () => {
    const options = {
      provenanceUsesServiceProvider: false,
    };
    const queryGraphHandler = new TRAPIQueryHandler(options);
    const inferredQueryHandler = new InferredQueryHandler(
      queryGraphHandler,
      queryGraph1,
      [],
      options,
      smartAPIPAth,
      predicatesPath,
      true,
    );
    inferredQueryHandler.CREATIVE_LIMIT = 3;
    const trapiQueryHandler0 = new TRAPIQueryHandler();
    trapiQueryHandler0.logs.push({
      message: 'new fake log',
    } as StampedLog);
    trapiQueryHandler0.getResponse = () => ({
      workflow: [{ id: 'lookup' }],
      message: {
        query_graph: {
          nodes: {
            creativeQuerySubject: {
              categories: ['biolink:SmallMolecule'],
            },
            creativeQueryObject: {
              categories: ['biolink:Disease'],
              ids: ['fakeDiseaseID'],
            },
          },
          edges: {
            e01: {
              subject: 'creativeQuerySubject',
              object: 'creativeQueryObject',
              predicates: ['biolink:treats'],
            },
          },
        },
        knowledge_graph: {
          nodes: {
            fakeCompound1: {
              categories: ['biolink:SmallMolecule'],
              name: 'fakeCompound1',
            },
            fakeCompound2: {
              categories: ['biolink:SmallMolecule'],
              name: 'fakeCompound2',
            },
            fakeCompound3: {
              categories: ['biolink:SmallMolecule'],
              name: 'fakeCompound3',
            },
            fakeDisease1: {
              categories: ['biolink:Disease'],
              name: 'fakeDisease1',
            },
          },
          edges: {
            edgeHash1: {
              predicate: 'biolink:treats',
              subject: 'fakeCompound2',
              object: 'fakeDisease1',
              sources: [],
            },
            edgeHash2: {
              predicate: 'biolink:treats',
              subject: 'fakeCompound1',
              object: 'fakeDisease1',
              sources: [],
            },
            edgeHash3: {
              predicate: 'biolink:treats',
              subject: 'fakeCompound3',
              object: 'fakeDisease1',
              sources: [],
            },
          },
        },
        results: [
          {
            node_bindings: {
              creativeQuerySubject: [
                {
                  id: 'fakeCompound2',
                },
              ],
              creativeQueryObject: [
                {
                  id: 'fakeDisease1',
                },
              ],
            },
            analyses: [
              {
                resource_id: 'infores:biothings_explorer',
                edge_bindings: {
                  e0: [
                    {
                      id: 'edgeHash1',
                    },
                  ],
                },
                score: 0.5,
              },
            ],
          },
          {
            node_bindings: {
              creativeQuerySubject: [
                {
                  id: 'fakeCompound1',
                },
              ],
              creativeQueryObject: [
                {
                  id: 'fakeDisease1',
                },
              ],
            },
            analyses: [
              {
                resource_id: 'infores:biothings_explorer',
                edge_bindings: {
                  e0: [
                    {
                      id: 'edgeHash2',
                    },
                  ],
                },
                score: 0.25,
              },
            ],
          },
          {
            node_bindings: {
              creativeQuerySubject: [
                {
                  id: 'fakeCompound3',
                },
              ],
              creativeQueryObject: [
                {
                  id: 'fakeDisease1',
                },
              ],
            },
            analyses: [
              {
                resource_id: 'infores:biothings_explorer',
                edge_bindings: {
                  e0: [
                    {
                      id: 'edgeHash3',
                    },
                  ],
                },
                score: 0.2,
              },
            ],
          },
        ],
      },
      logs: [
        {
          message: 'new fake log',
        } as TrapiLog,
      ],
    });

    const combinedResponse: CombinedResponse = {
      workflow: [{ id: 'lookup' }],
      message: {
        auxiliary_graphs: {},
        query_graph: queryGraph1,
        knowledge_graph: {
          nodes: {
            fakeCompound1: {
              categories: ['biolink:SmallMolecule'],
              name: 'fakeCompound1',
            },
            fakeCompound3: {
              categories: ['biolink:SmallMolecule'],
              name: 'fakeCompound3',
            },
            fakeDisease1: {
              categories: ['biolink:Disease'],
              name: 'fakeDisease1',
            },
          },
          edges: {
            edgeHash1: {
              predicate: 'biolink:treats',
              subject: 'fakeCompound1',
              object: 'fakeDisease1',
              sources: [],
            },
            edgeHash2: {
              predicate: 'biolink:treats',
              subject: 'fakeCompound3',
              object: 'fakeDisease1',
              sources: [],
            },
          },
        },
        results: {
          'fakeCompound1-fakeDisease1': {
            node_bindings: {
              n01: [
                {
                  id: 'fakeCompound1',
                },
              ],
              n02: [
                {
                  id: 'fakeDisease1',
                },
              ],
            },
            analyses: [
              {
                resource_id: 'infores:biothings_explorer',
                edge_bindings: {
                  e01: [
                    {
                      id: 'edgeHash1',
                    },
                  ],
                },
                score: 0.75,
              },
            ],
          },
          'fakeCompound3-fakeDisease1': {
            node_bindings: {
              n01: [
                {
                  id: 'fakeCompound3',
                },
              ],
              n02: [
                {
                  id: 'fakeDisease1',
                },
              ],
            },
            analyses: [
              {
                edge_bindings: {
                  e01: [
                    {
                      id: 'edgeHash2',
                    },
                  ],
                },
                score: undefined,
              },
            ],
          },
        },
      },
      logs: [
        {
          message: 'fake initial log',
        } as StampedLog,
      ],
    };

    const { qEdgeID, qEdge, qSubject, qObject } = inferredQueryHandler.getQueryParts();

    const report = inferredQueryHandler.combineResponse(1, trapiQueryHandler0, qEdgeID, qEdge, combinedResponse);

    expect(report).toHaveProperty('querySuccess');
    expect(report).toHaveProperty('queryHadResults');
    expect(report).toHaveProperty('mergedResults');
    expect(report).toHaveProperty('creativeLimitHit');

    const { querySuccess, queryHadResults, mergedResults, creativeLimitHit } = report;
    expect(querySuccess).toBeTruthy();
    expect(queryHadResults).toBeTruthy();
    expect(Object.keys(mergedResults)).toHaveLength(2);
    expect(Object.values(mergedResults)[0]).toEqual(1);
    expect(creativeLimitHit).toBeTruthy();
    expect(Object.keys(combinedResponse.message.results)).toHaveLength(3);
    expect(combinedResponse.message.results['fakeCompound1-fakeDisease1'].analyses[0].score).toEqual(
      0.8421052631578949,
    );
    expect(combinedResponse.message.results['fakeCompound3-fakeDisease1'].analyses[0].score).toEqual(0.2);
    expect(combinedResponse.logs).toHaveLength(3);
    expect(combinedResponse.logs[1].message).toMatch('[Template-2]: new fake log');

    const trapiQueryHandler1 = new TRAPIQueryHandler();
    trapiQueryHandler1.getResponse = () => ({
      workflow: [{ id: 'lookup' }],
      message: {
        query_graph: {
          nodes: {
            creativeQuerySubject: {
              categories: ['biolink:SmallMolecule'],
            },
            creativeQueryObject: {
              categories: ['biolink:Disease'],
              ids: ['fakeDiseaseID'],
            },
            n01: {
              categories: ['biolink:Gene'],
              ids: ['aGeneThatCausesFakeDisease1'],
            },
          },
          edges: {
            e0: {
              subject: 'creativeQuerySubject',
              object: 'creativeQueryObject',
              predicate: ['biolink:treats'],
            },
            e01: {
              subject: 'creativeQuerySubject',
              object: 'someIntermediateNode',
              predicate: ['biolink:negatively_regulates'],
            },
          },
        },
        knowledge_graph: {
          nodes: {
            fakeGene1: {
              categories: ['biolink:Gene'],
              name: 'fakeGene1',
            },
            fakeCompound4: {
              categories: ['biolink:SmallMolecule'],
              name: 'fakeCompound4',
            },
            fakeCompound1: {
              categories: ['biolink:SmallMolecule'],
              name: 'fakeCompound1',
            },
            fakeDisease1: {
              categories: ['biolink:Disease'],
              name: 'fakeDisease1',
            },
          },
          edges: {
            edgeHash1: {
              predicate: 'biolink:negatively_regulates',
              subject: 'fakeCompound4',
              object: 'fakeGene1',
              sources: [],
            },
            edgeHash2: {
              predicate: 'biolink:causes',
              subject: 'fakeGene1',
              object: 'fakeDisease1',
              sources: [],
            },
            edgeHash3: {
              predicate: 'biolink:treats',
              subject: 'fakeCompound1',
              object: 'fakeDisease1',
              sources: [],
            },
          },
        },
        results: [
          {
            node_bindings: {
              creativeQuerySubject: [
                {
                  id: 'fakeCompound4',
                },
              ],
              creativeQueryObject: [
                {
                  id: 'fakeDisease1',
                },
              ],
              n01: [
                {
                  id: 'fakeGene1',
                },
              ],
            },
            analyses: [
              {
                resource_id: 'infores:biothings_explorer',
                edge_bindings: {
                  e0: [
                    {
                      id: 'edgeHash1',
                    },
                  ],
                  e01: [
                    {
                      id: 'edgeHash2',
                    },
                  ],
                },
                score: 0.99,
              },
            ],
          } as TrapiResult,
          {
            node_bindings: {
              creativeQuerySubject: [
                {
                  id: 'fakeCompound1',
                },
              ],
              creativeQueryObject: [
                {
                  id: 'fakeDisease1',
                },
              ],
            },
            analyses: [
              {
                resource_id: 'infores:biothings_explorer',
                edge_bindings: {
                  e0: [
                    {
                      id: 'edgeHash3',
                    },
                  ],
                },
                score: 0,
              },
            ],
          } as TrapiResult,
        ],
      },
      logs: [
        {
          message: 'new fake log',
        } as TrapiLog,
      ],
    });

    const {
      querySuccess: querySuccess1,
      queryHadResults: queryHadResults1,
      mergedResults: mergedResults1,
      creativeLimitHit: creativeLimitHit1,
    } = inferredQueryHandler.combineResponse(2, trapiQueryHandler1, qEdgeID, qEdge, combinedResponse);

    expect(querySuccess1).toBeTruthy();
    expect(queryHadResults1).toBeTruthy();
    expect(Object.keys(mergedResults1)).toHaveLength(1);
    expect(creativeLimitHit1).toBeTruthy();
    expect(combinedResponse.message.results['fakeCompound1-fakeDisease1'].analyses[0].score).toEqual(
      0.8421052631578949,
    );
  });

  test('pruneKnowledgeGraph', () => {
    const queryGraphHandler = new TRAPIQueryHandler();
    const handler = new InferredQueryHandler(
      queryGraphHandler,
      queryGraph1,
      [],
      {},
      smartAPIPAth,
      predicatesPath,
      true,
    );
    const combinedResponse: TrapiResponse = {
      workflow: [{ id: 'lookup' }],
      message: {
        auxiliary_graphs: {},
        query_graph: queryGraph1,
        knowledge_graph: {
          nodes: {
            fakeCompound1: {
              categories: ['biolink:SmallMolecule'],
              name: 'fakeCompound1',
            },
            fakeCompound3: {
              categories: ['biolink:SmallMolecule'],
              name: 'fakeCompound3',
            },
            fakeDisease1: {
              categories: ['biolink:Disease'],
              name: 'fakeDisease1',
            },
          },
          edges: {
            edgeHash1: {
              predicate: 'biolink:treats',
              subject: 'fakeCompound1',
              object: 'fakeDisease1',
              sources: [],
              attributes: [
                {
                  attribute_type_id: 'biolink:support_graphs',
                  value: [],
                },
              ],
            },
            edgeHash2: {
              predicate: 'biolink:treats',
              subject: 'fakeCompound3',
              object: 'fakeDisease1',
              sources: [],
              attributes: [
                {
                  attribute_type_id: 'biolink:support_graphs',
                  value: [],
                },
              ],
            },
          },
        },
        results: [
          {
            node_bindings: {
              n01: [
                {
                  id: 'fakeCompound1',
                },
              ],
              n02: [
                {
                  id: 'fakeDisease1',
                },
              ],
            },
            analyses: [
              {
                resource_id: 'infores:biothings_explorer',
                edge_bindings: {
                  e01: [
                    {
                      id: 'edgeHash1',
                    },
                  ],
                },
                score: 0.75,
              },
            ],
          } as TrapiResult,
        ],
      },
      logs: [
        {
          message: 'fake initial log',
        } as StampedLog,
      ],
    };

    handler.pruneKnowledgeGraph(combinedResponse);
    expect(combinedResponse.message.knowledge_graph.nodes).not.toHaveProperty('fakeCompounds3');
    expect(combinedResponse.message.knowledge_graph.edges).not.toHaveProperty('edgeHash2');
  });

  test('query', async () => {
    const querySpy = jest.spyOn(TRAPIQueryHandler.prototype, 'query');
    querySpy.mockImplementation(async () => undefined);
    const responseSpy = jest.spyOn(TRAPIQueryHandler.prototype, 'getResponse');
    responseSpy.mockImplementation(() => {
      return {
        workflow: [{ id: 'lookup' }],
        message: {
          query_graph: {
            nodes: {
              creativeQuerySubject: {
                categories: ['biolink:SmallMolecule'],
              },
              creativeQueryObject: {
                categories: ['biolink:Disease'],
                ids: ['fakeDisease1'],
              },
            },
            edges: {
              e01: {
                subject: 'creativeQuerySubject',
                object: 'creativeQueryObject',
                predicates: ['biolink:treats'],
              },
            },
          },
          knowledge_graph: {
            nodes: {
              creativeQuerySubject: {
                categories: ['biolink:SmallMolecule'],
                name: 'fakeCompound1',
              },
              creativeQueryObject: {
                categories: ['biolink:Disease'],
                name: 'fakeDisease1',
              },
            },
            edges: {
              edgeHash1: {
                predicate: 'biolink:treats',
                subject: 'creativeQuerySubject',
                object: 'creativeQueryObject',
                knowledge_type: 'inferred',
                sources: [],
                attributes: [
                  {
                    attribute_type_id: 'biolink:support_graphs',
                    value: [],
                  },
                ],
              },
            },
          },
          results: [
            {
              node_bindings: {
                creativeQuerySubject: [
                  {
                    id: 'creativeQuerySubject',
                  },
                ],
                creativeQueryObject: [
                  {
                    id: 'creativeQueryObject',
                  },
                ],
              },
              analyses: [
                {
                  edge_bindings: {
                    e01: [
                      {
                        id: 'edgeHash1',
                      },
                    ],
                  },
                  score: 0.75,
                },
              ],
            },
          ],
        },
        logs: [
          {
            message: 'fake initial log',
          } as TrapiLog,
        ],
      };
    });
    const queryIsValid = jest.spyOn(InferredQueryHandler.prototype, 'queryIsValid', 'get');
    const getQueryParts = jest.spyOn(InferredQueryHandler.prototype, 'getQueryParts');
    const findTemplates = jest.spyOn(InferredQueryHandler.prototype, 'findTemplates');
    const createQueries = jest.spyOn(InferredQueryHandler.prototype, 'createQueries');
    const combineResponse = jest.spyOn(InferredQueryHandler.prototype, 'combineResponse');
    const pruneKnowledgeGraph = jest.spyOn(InferredQueryHandler.prototype, 'pruneKnowledgeGraph');

    const parentHandler = new TRAPIQueryHandler();

    const handler = new InferredQueryHandler(
      parentHandler,
      {
        nodes: {
          creativeQuerySubject: {
            categories: ['biolink:SmallMolecule'],
          },
          creativeQueryObject: {
            categories: ['biolink:Disease'],
            ids: ['fakeDisease1'],
          },
        },
        edges: {
          e0: {
            subject: 'creativeQuerySubject',
            object: 'creativeQueryObject',
            predicates: ['biolink:treats'],
            knowledge_type: 'inferred',
          },
        },
      },
      [],
      {},
      smartAPIPAth,
      predicatesPath,
      true,
    );
    handler.CREATIVE_LIMIT = 1;

    const response = await handler.query();

    expect(queryIsValid).toHaveBeenCalled();
    expect(getQueryParts).toHaveBeenCalled();
    expect(findTemplates).toHaveBeenCalled();
    expect(createQueries).toHaveBeenCalled();
    expect(combineResponse).toHaveBeenCalled();
    expect(pruneKnowledgeGraph).toHaveBeenCalled();

    expect(response).toBeTruthy();
    expect(Object.keys(response.logs).length).toBeGreaterThan(0);
    expect(response.message.results).toHaveLength(1);
    expect(response.message.knowledge_graph.edges).toHaveProperty('edgeHash1');
    expect(response.message.knowledge_graph.nodes).toHaveProperty('creativeQuerySubject');
    expect(response.message.knowledge_graph.nodes).toHaveProperty('creativeQueryObject');
    expect(response.message.results[0].node_bindings).toHaveProperty('creativeQuerySubject');
    expect(response.message.results[0].node_bindings).toHaveProperty('creativeQueryObject');
    expect(response.logs.map((log) => log.message)).toContain(
      'Addition of 1 results from Template 1 meets creative result maximum of 1 (reaching 1 merged). Response will be truncated to top-scoring 1 results. Skipping remaining 2 templates.',
    );
  });

  test('supportedLookups', async () => {
    const { supportedLookups } = require('../../src/inferred_mode/template_lookup');
    const supported = await supportedLookups();
    expect(supported).toContainEqual({
      subject: 'biolink:Drug',
      predicate: 'biolink:treats',
      object: 'biolink:Disease',
      qualifiers: undefined,
    });
    expect(supported).toContainEqual({
      subject: 'biolink:SmallMolecule',
      predicate: 'biolink:treats',
      object: 'biolink:PhenotypicFeature',
      qualifiers: undefined,
    });
    expect(supported.length).toBeGreaterThanOrEqual(5 * 2 * 3);
  });
});
