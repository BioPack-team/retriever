import Debug from 'debug';
import { LogEntry, StampedLog, Telemetry } from '@biothings-explorer/utils';
import * as utils from '../utils';
import async from 'async';
import biolink from '../biolink';
import { getTemplates, MatchedTemplate, TemplateLookup } from './template_lookup';
import { scaled_sigmoid, inverse_scaled_sigmoid } from '../results_assembly/score';
import TRAPIQueryHandler from '../index';
import { QueryHandlerOptions } from '@biothings-explorer/types';
import {
  CompactQualifiers,
  TrapiAuxGraphCollection,
  TrapiEdgeBinding,
  TrapiKnowledgeGraph,
  TrapiQEdge,
  TrapiQNode,
  TrapiQueryGraph,
  TrapiResponse,
  TrapiResult,
} from '../types';
const debug = Debug('bte:biothings-explorer-trapi:inferred-mode');

export interface CombinedResponse {
  description?: string;
  workflow?: { id: string }[];
  message: {
    query_graph: TrapiQueryGraph;
    knowledge_graph: TrapiKnowledgeGraph;
    auxiliary_graphs?: TrapiAuxGraphCollection;
    results: {
      [resultID: string]: TrapiResult;
    };
  };
  logs: StampedLog[];
}

export interface CombinedResponseReport {
  querySuccess: number;
  queryHadResults: boolean;
  mergedResults: { [resultID: string]: number };
  creativeLimitHit: boolean | number;
}

// MatchedTemplate, but with IDs, etc. filled in
export type FilledTemplate = MatchedTemplate;

export default class InferredQueryHandler {
  parent: TRAPIQueryHandler;
  queryGraph: TrapiQueryGraph;
  logs: StampedLog[];
  options: QueryHandlerOptions;
  path: string;
  predicatePath: string;
  includeReasoner: boolean;
  CREATIVE_LIMIT: number;
  constructor(
    parent: TRAPIQueryHandler,
    queryGraph: TrapiQueryGraph,
    logs: StampedLog[],
    options: QueryHandlerOptions,
    path: string,
    predicatePath: string,
    includeReasoner: boolean,
  ) {
    this.parent = parent;
    this.queryGraph = queryGraph;
    this.logs = logs;
    this.options = options;
    this.path = path;
    this.predicatePath = predicatePath;
    this.includeReasoner = includeReasoner;
    this.CREATIVE_LIMIT = process.env.CREATIVE_LIMIT ? parseInt(process.env.CREATIVE_LIMIT) : 500;
  }

  get queryIsValid(): boolean {
    const nodeMissingCategory = Object.values(this.queryGraph.nodes).some((node) => {
      return !node.categories || node.categories.length === 0;
    });
    if (nodeMissingCategory) {
      const message = 'All nodes in Inferred Mode edge must have categories. Your query terminates.';
      this.logs.push(new LogEntry('WARNING', null, message).getLog());
      debug(message);
      return false;
    }

    const nodeMissingID = !Object.values(this.queryGraph.nodes).some((node) => {
      return node.ids && node.ids.length > 0;
    });
    if (nodeMissingID) {
      const message = 'At least one node in Inferred Mode edge must have at least 1 ID. Your query terminates.';
      this.logs.push(new LogEntry('WARNING', null, message).getLog());
      debug(message);
      return false;
    }

    const edgeMissingPredicate =
      !Object.values(this.queryGraph.edges)[0].predicates ||
      Object.values(this.queryGraph.edges)[0].predicates.length < 1;
    if (edgeMissingPredicate) {
      const message = 'Inferred Mode edge must specify a predicate. Your query terminates.';
      this.logs.push(new LogEntry('WARNING', null, message).getLog());
      debug(message);
      return false;
    }

    const tooManyIDs =
      1 <
      Object.values(this.queryGraph.nodes).reduce((sum, node) => {
        return typeof node.ids !== 'undefined' ? sum + node.ids.length : sum;
      }, 0);
    if (tooManyIDs) {
      const message = 'Inferred Mode queries with multiple IDs are not supported. Your query terminates.';
      this.logs.push(new LogEntry('WARNING', null, message).getLog());
      debug(message);
      return false;
    }

    const multiplePredicates =
      Object.values(this.queryGraph.edges).reduce((sum, edge) => {
        return edge.predicates ? sum + edge.predicates.length : sum;
      }, 0) > 1;
    if (multiplePredicates) {
      const message = 'Inferred Mode queries with multiple predicates are not supported. Your query terminates.';
      this.logs.push(new LogEntry('WARNING', null, message).getLog());
      debug(message);
      return false;
    }

    if (this.options.smartAPIID || this.options.teamName) {
      const message = 'Inferred Mode on smartapi/team-specific endpoints not supported. Your query terminates.';
      this.logs.push(new LogEntry('WARNING', null, message).getLog());
      debug(message);
      return false;
    }

    return true;
  }

  getQueryParts(): {
    qEdgeID: string;
    qEdge: TrapiQEdge;
    qSubject: TrapiQNode;
    qObject: TrapiQNode;
  } {
    const qEdgeID = Object.keys(this.queryGraph.edges)[0];
    const qEdge = this.queryGraph.edges[qEdgeID];
    const qSubject = this.queryGraph.nodes[qEdge.subject];
    const qObject = this.queryGraph.nodes[qEdge.object];
    return {
      qEdgeID: qEdgeID,
      qEdge: qEdge,
      qSubject: qSubject,
      qObject: qObject,
    };
  }

  async findTemplates(qEdge: TrapiQEdge, qSubject: TrapiQNode, qObject: TrapiQNode): Promise<MatchedTemplate[]> {
    debug('Looking up query Templates');
    const expandedSubject = qSubject.categories.reduce((arr: string[], subjectCategory: string) => {
      return utils.getUnique([...arr, ...biolink.getDescendantClasses(utils.removeBioLinkPrefix(subjectCategory))]);
    }, [] as string[]);
    const expandedPredicates = qEdge.predicates.reduce((arr: string[], predicate: string) => {
      return utils.getUnique([...arr, ...biolink.getDescendantPredicates(utils.removeBioLinkPrefix(predicate))]);
    }, [] as string[]);
    const expandedObject = qObject.categories.reduce((arr: string[], objectCategory: string) => {
      return utils.getUnique([...arr, ...biolink.getDescendantClasses(utils.removeBioLinkPrefix(objectCategory))]);
    }, [] as string[]);
    const qualifierConstraints = (qEdge.qualifier_constraints || []).map((qualifierSetObj) => {
      return Object.fromEntries(
        qualifierSetObj.qualifier_set.map(({ qualifier_type_id, qualifier_value }) => [
          qualifier_type_id.replace('biolink:', ''),
          Array.isArray(qualifier_value)
            ? qualifier_value.map((string) => string.replace('biolink:', ''))
            : qualifier_value.replace('biolink:', ''),
        ]),
      ) as CompactQualifiers;
    });
    if (qualifierConstraints.length === 0) qualifierConstraints.push({});
    const lookupObjects = expandedSubject.reduce((arr: TemplateLookup[], subjectCategory) => {
      const objectCombos = expandedObject.reduce((arr2: TemplateLookup[], objectCategory) => {
        const qualifierCombos = qualifierConstraints.reduce(
          (arr3: TemplateLookup[], qualifierSet: CompactQualifiers) => [
            ...arr3,
            ...expandedPredicates.map((predicate) => {
              return {
                subject: utils.removeBioLinkPrefix(subjectCategory),
                object: utils.removeBioLinkPrefix(objectCategory),
                predicate: utils.removeBioLinkPrefix(predicate),
                qualifiers: qualifierSet,
              };
            }),
          ],
          [],
        );
        return [...arr2, ...qualifierCombos];
      }, []);
      return [...arr, ...objectCombos];
    }, []);
    const templates = await getTemplates(lookupObjects);

    const logMessage = `Got ${templates.length} inferred query templates.`;
    debug(logMessage);
    this.logs.push(new LogEntry('INFO', null, logMessage).getLog());

    if (!templates.length) {
      const logMessage = `No Templates matched your inferred-mode query. Your query terminates.`;
      debug(logMessage);
      this.logs.push(new LogEntry('WARNING', null, logMessage).getLog());
    }

    return templates;
  }

  async createQueries(qEdge: TrapiQEdge, qSubject: TrapiQNode, qObject: TrapiQNode): Promise<FilledTemplate[]> {
    const templates = await this.findTemplates(qEdge, qSubject, qObject);
    // combine creative query with templates
    const subQueries = templates.map(({ template, queryGraph }) => {
      queryGraph.nodes.creativeQuerySubject.categories = [
        ...new Set([...queryGraph.nodes.creativeQuerySubject.categories, ...qSubject.categories]),
      ];
      const creativeQuerySubjectIDs = qSubject.ids ? qSubject.ids : [];
      queryGraph.nodes.creativeQuerySubject.ids = queryGraph.nodes.creativeQuerySubject.ids
        ? [...new Set([...queryGraph.nodes.creativeQuerySubject.ids, ...creativeQuerySubjectIDs])]
        : creativeQuerySubjectIDs;

      queryGraph.nodes.creativeQueryObject.categories = [
        ...new Set([...queryGraph.nodes.creativeQueryObject.categories, ...qObject.categories]),
      ];
      const qEdgeObjectIDs = qObject.ids ? qObject.ids : [];
      queryGraph.nodes.creativeQueryObject.ids = queryGraph.nodes.creativeQueryObject.ids
        ? [...new Set([...queryGraph.nodes.creativeQueryObject.ids, ...qEdgeObjectIDs])]
        : qEdgeObjectIDs;

      if (!queryGraph.nodes.creativeQuerySubject.categories.length) {
        delete queryGraph.nodes.creativeQuerySubject.categories;
      }
      if (!queryGraph.nodes.creativeQueryObject.categories.length) {
        delete queryGraph.nodes.creativeQueryObject.categories;
      }
      if (!queryGraph.nodes.creativeQuerySubject.ids.length) {
        delete queryGraph.nodes.creativeQuerySubject.ids;
      }
      if (!queryGraph.nodes.creativeQueryObject.ids.length) {
        delete queryGraph.nodes.creativeQueryObject.ids;
      }

      return { template, queryGraph };
    });

    return subQueries;
  }

  combineResponse(
    queryNum: number,
    handler: TRAPIQueryHandler,
    qEdgeID: string,
    qEdge: TrapiQEdge,
    combinedResponse: CombinedResponse,
  ): CombinedResponseReport {
    const span = Telemetry.startSpan({ description: 'creativeCombineResponse' });
    const newResponse = handler.getResponse();
    const report: CombinedResponseReport = {
      querySuccess: 0,
      queryHadResults: false,
      mergedResults: {},
      creativeLimitHit: false,
    };
    let mergedThisTemplate = 0;
    const resultIDsFromPrevious = new Set(Object.keys(combinedResponse.message.results));
    // add non-duplicate nodes
    Object.entries(newResponse.message.knowledge_graph.nodes).forEach(([curie, node]) => {
      if (!(curie in combinedResponse.message.knowledge_graph.nodes)) {
        combinedResponse.message.knowledge_graph.nodes[curie] = node;
      }
    });
    // add non-duplicate edges
    Object.entries(newResponse.message.knowledge_graph.edges).forEach(([recordHash, edge]) => {
      if (!(recordHash in combinedResponse.message.knowledge_graph.edges)) {
        combinedResponse.message.knowledge_graph.edges[recordHash] = edge;
      }
    });
    Object.entries(newResponse.message.auxiliary_graphs ?? {}).forEach(([auxGraphID, auxGraph]) => {
      if (!(auxGraphID in combinedResponse.message.auxiliary_graphs)) {
        combinedResponse.message.auxiliary_graphs[auxGraphID] = auxGraph;
      }
    });
    // add results
    newResponse.message.results.forEach((result) => {
      const translatedResult: TrapiResult = {
        node_bindings: {
          [qEdge.subject]: [{ id: result.node_bindings.creativeQuerySubject[0].id }],
          [qEdge.object]: [{ id: result.node_bindings.creativeQueryObject[0].id }],
        },
        pfocr: result.pfocr?.length ? result.pfocr : undefined,
        analyses: [
          {
            resource_id: result.analyses[0].resource_id,
            edge_bindings: {},
            score: result.analyses[0].score,
          },
        ],
      };
      const resultCreativeSubjectID = translatedResult.node_bindings[qEdge.subject]
        .map((binding) => binding.id)
        .join(',');
      const resultCreativeObjectID = translatedResult.node_bindings[qEdge.object]
        .map((binding) => binding.id)
        .join(',');
      const resultID = `${resultCreativeSubjectID}-${resultCreativeObjectID}`;

      // Direct edge answers stand on their own, not as an inferred edge.
      if (Object.keys(result.node_bindings).length == 2) {
        const boundEdgeID = Object.values(result.analyses[0].edge_bindings)[0][0].id;
        translatedResult.analyses[0].edge_bindings = { [qEdgeID]: [{ id: boundEdgeID }] };
      } else {
        // Create an aux graph using the result and associate it with an inferred Edge
        const inferredEdgeID = `inferred-${resultCreativeSubjectID}-${qEdge.predicates[0].replace(
          'biolink:',
          '',
        )}-${resultCreativeObjectID}`;
        translatedResult.analyses[0].edge_bindings = { [qEdgeID]: [{ id: inferredEdgeID }] };
        if (!combinedResponse.message.knowledge_graph.edges[inferredEdgeID]) {
          combinedResponse.message.knowledge_graph.edges[inferredEdgeID] = {
            subject: resultCreativeSubjectID,
            object: resultCreativeObjectID,
            predicate: qEdge.predicates[0],
            sources: [
              {
                resource_id: this.parent.options.provenanceUsesServiceProvider
                  ? 'infores:service-provider-trapi'
                  : 'infores:biothings-explorer',
                resource_role: 'primary_knowledge_source',
              },
            ],
            attributes: [{ attribute_type_id: 'biolink:support_graphs', value: [] }],
          };
        }
        let auxGraphSuffix = 0;
        while (
          Object.keys(combinedResponse.message.auxiliary_graphs).includes(`${inferredEdgeID}-support${auxGraphSuffix}`)
        ) {
          auxGraphSuffix += 1;
        }
        const auxGraphID = `${inferredEdgeID}-support${auxGraphSuffix}`;
        (combinedResponse.message.knowledge_graph.edges[inferredEdgeID].attributes[0].value as string[]).push(
          auxGraphID,
        );
        combinedResponse.message.auxiliary_graphs[auxGraphID] = {
          edges: Object.values(result.analyses[0].edge_bindings).reduce(
            (arr: string[], bindings: TrapiEdgeBinding[]) => {
              bindings.forEach((binding) => arr.push(binding.id));
              return arr;
            },
            [] as string[],
          ),
        };
      }

      if (resultID in combinedResponse.message.results) {
        report.mergedResults[resultID] = report.mergedResults[resultID] ? report.mergedResults[resultID] + 1 : 1;
        mergedThisTemplate += 1;
        Object.entries(translatedResult.analyses[0].edge_bindings).forEach(([edgeID, bindings]) => {
          const combinedBindings = combinedResponse.message.results[resultID].analyses[0].edge_bindings[edgeID];
          bindings.forEach((binding) => {
            if (combinedBindings.some((combinedBinding) => combinedBinding.id === binding.id)) return;
            combinedResponse.message.results[resultID].analyses[0].edge_bindings[edgeID].push(binding);
          });
        });

        // Combine, re-sort, and truncate to 20 any pfocr figures
        if (combinedResponse.message.results[resultID].pfocr || translatedResult.pfocr) {
          let reSort = false;
          if (combinedResponse.message.results[resultID].pfocr && translatedResult.pfocr) reSort = true;
          let newFigures = [
            ...(combinedResponse.message.results[resultID].pfocr ?? []),
            ...(translatedResult.pfocr ?? []),
          ];
          if (reSort) {
            newFigures = newFigures.sort((figA, figB) => figB.score - figA.score).slice(0, 20);
          }
          combinedResponse.message.results[resultID].pfocr = newFigures;
        }

        const resScore = translatedResult.analyses[0].score;
        if (typeof combinedResponse.message.results[resultID].analyses[0].score !== 'undefined') {
          combinedResponse.message.results[resultID].analyses[0].score = resScore
            ? scaled_sigmoid(
              inverse_scaled_sigmoid(combinedResponse.message.results[resultID].analyses[0].score) +
              inverse_scaled_sigmoid(resScore),
            )
            : combinedResponse.message.results[resultID].analyses[0].score;
        } else {
          combinedResponse.message.results[resultID].analyses[0].score = resScore;
        }
      } else {
        combinedResponse.message.results[resultID] = translatedResult;
      }
    });
    const mergedWithinTemplate = Object.entries(report.mergedResults).reduce((count, [resultID, merged]) => {
      return !resultIDsFromPrevious.has(resultID) ? count + merged : count;
    }, 0);

    // fix/combine logs
    handler.logs.forEach((log) => {
      log.message = `[Template-${queryNum + 1}]: ${log.message}`;

      combinedResponse.logs.push(log);
    });

    const mergeMessage = [
      `(${mergedWithinTemplate}) results from Template-${queryNum + 1} `,
      `were merged with other results from the template. `,
      `(${mergedThisTemplate - mergedWithinTemplate}) results `,
      `were merged with existing results from previous templates. `,
      `Current result count is ${Object.keys(combinedResponse.message.results).length} `,
      `(+${newResponse.message.results.length - mergedThisTemplate})`,
    ].join('');
    debug(mergeMessage);
    combinedResponse.logs.push(new LogEntry('INFO', null, mergeMessage).getLog());

    if (newResponse.message.results.length) {
      report.queryHadResults = true;
    }
    report.querySuccess = 1;

    if (Object.keys(combinedResponse.message.results).length >= this.CREATIVE_LIMIT && !report.creativeLimitHit) {
      report.creativeLimitHit = Object.keys(newResponse.message.results).length;
    }
    span.finish();
    return report;
  }

  pruneKnowledgeGraph(combinedResponse: TrapiResponse): void {
    debug('pruning creative combinedResponse nodes/edges...');
    const edgeBoundNodes: Set<string> = new Set();
    const resultsBoundEdges: Set<string> = new Set();
    const resultBoundAuxGraphs: Set<string> = new Set();

    // Handle nodes and edges bound to results directly
    combinedResponse.message.results.forEach((result) => {
      Object.entries(result.analyses[0].edge_bindings).forEach(([, bindings]) => {
        bindings.forEach((binding) => resultsBoundEdges.add(binding.id));
      });
    });

    // Handle edges bound via auxiliary graphs
    // This will iterate over new edges as they're added
    resultsBoundEdges.forEach((edgeID) => {
      edgeBoundNodes.add(combinedResponse.message.knowledge_graph.edges[edgeID].subject);
      edgeBoundNodes.add(combinedResponse.message.knowledge_graph.edges[edgeID].object);
      combinedResponse.message.knowledge_graph.edges[edgeID].attributes.find(({ attribute_type_id, value }) => {
        if (attribute_type_id === 'biolink:support_graphs') {
          (value as string[]).forEach((auxGraphID) => {
            resultBoundAuxGraphs.add(auxGraphID);
            combinedResponse.message.auxiliary_graphs[auxGraphID].edges.forEach((auxGraphEdgeID) => {
              edgeBoundNodes.add(combinedResponse.message.knowledge_graph.edges[auxGraphEdgeID].subject);
              edgeBoundNodes.add(combinedResponse.message.knowledge_graph.edges[auxGraphEdgeID].object);
              resultsBoundEdges.add(auxGraphEdgeID);
            });
          });
          return true;
        }
      });
    });

    const nodesToDelete = Object.keys(combinedResponse.message.knowledge_graph.nodes).filter(
      (nodeID) => !edgeBoundNodes.has(nodeID),
    );
    nodesToDelete.forEach((unusedBTENodeID) => delete combinedResponse.message.knowledge_graph.nodes[unusedBTENodeID]);
    const edgesToDelete = Object.keys(combinedResponse.message.knowledge_graph.edges).filter(
      (edgeID) => !resultsBoundEdges.has(edgeID),
    );
    edgesToDelete.forEach((unusedEdgeID) => delete combinedResponse.message.knowledge_graph.edges[unusedEdgeID]);
    const auxGraphsToDelete = Object.keys(combinedResponse.message.auxiliary_graphs).filter(
      (auxGraphID) => !resultBoundAuxGraphs.has(auxGraphID),
    );
    auxGraphsToDelete.forEach((unusedAuxGraphID) => delete combinedResponse.message.auxiliary_graphs[unusedAuxGraphID]);
    debug(
      `pruned ${nodesToDelete.length} nodes, ${edgesToDelete.length} edges, ${auxGraphsToDelete.length} auxGraphs from combinedResponse.`,
    );
  }

  async query(): Promise<TrapiResponse> {
    // TODO (eventually) check for flipped predicate cases
    // e.g. Drug -treats-> Disease OR Disease -treated_by-> Drug
    const logMessage = 'Query proceeding in Inferred Mode.';
    debug(logMessage);
    this.logs.push(new LogEntry('INFO', null, logMessage).getLog());

    if (!this.queryIsValid) {
      return;
    }

    const { qEdgeID, qEdge, qSubject, qObject } = this.getQueryParts();
    const subQueries = await this.createQueries(qEdge, qSubject, qObject);
    const combinedResponse = {
      status: 'Success',
      description: '',
      schema_version: global.SCHEMA_VERSION,
      biolink_version: global.BIOLINK_VERSION,
      workflow: [{ id: 'lookup' }],
      message: {
        query_graph: this.queryGraph,
        knowledge_graph: {
          nodes: {},
          edges: {},
        },
        auxiliary_graphs: {},
        results: {},
      },
      logs: this.logs,
    } as CombinedResponse;
    // add/combine nodes
    const resultQueries = [];
    let successfulQueries = 0;
    let stop = false;
    const mergedResultsCount: {
      [resultID: string]: number;
    } = {};

    await async.eachOfSeries(subQueries, async ({ template, queryGraph }, i) => {
      const span = Telemetry.startSpan({ description: 'creativeTemplate' });
      span.setData('template', (i as number) + 1);
      i = i as number;
      if (stop) {
        span.finish();
        return;
      }
      if (global.queryInformation?.queryGraph) {
        global.queryInformation.isCreativeMode = true;
        global.queryInformation.creativeTemplate = template;
      }
      const handler = new TRAPIQueryHandler(this.options, this.path, this.predicatePath, this.includeReasoner);
      try {
        // make query and combine results/kg/logs/etc
        handler.setQueryGraph(queryGraph);
        await handler.query();
        const { querySuccess, queryHadResults, mergedResults, creativeLimitHit } = this.combineResponse(
          i,
          handler,
          qEdgeID,
          qEdge,
          combinedResponse,
        );
        // update values used in logging
        successfulQueries += querySuccess;
        if (queryHadResults) resultQueries.push(i);
        Object.entries(mergedResults).forEach(([result, countMerged]) => {
          mergedResultsCount[result] =
            result in mergedResultsCount ? mergedResultsCount[result] + countMerged : countMerged;
        });
        // log to user if we should stop
        if (creativeLimitHit) {
          stop = true;
          const message = [
            `Addition of ${creativeLimitHit} results from Template ${i + 1}`,
            Object.keys(combinedResponse.message.results).length === this.CREATIVE_LIMIT ? ' meets ' : ' exceeds ',
            `creative result maximum of ${this.CREATIVE_LIMIT} (reaching ${Object.keys(combinedResponse.message.results).length
            } merged). `,
            `Response will be truncated to top-scoring ${this.CREATIVE_LIMIT} results. Skipping remaining ${subQueries.length - (i + 1)
            } `,
            subQueries.length - (i + 1) === 1 ? `template.` : `templates.`,
          ].join('');
          debug(message);
          combinedResponse.logs.push(new LogEntry(`INFO`, null, message).getLog());
        }
        span.finish();
      } catch (error) {
        handler.logs.forEach((log) => {
          combinedResponse.logs.push(log);
        });
        const message = `ERROR:  Template-${i + 1} failed due to error ${error}`;
        debug(message);
        combinedResponse.logs.push(new LogEntry(`ERROR`, null, message).getLog());
        span.finish();
        return;
      }
    });
    // log about merged Results
    if (Object.keys(mergedResultsCount).length) {
      // Add 1 for first instance of result (not counted during merging)
      const total =
        Object.values(mergedResultsCount).reduce((sum, count) => sum + count, 0) +
        Object.keys(mergedResultsCount).length;
      const message = `Merging Summary: (${total}) inferred-template results were merged into (${Object.keys(mergedResultsCount).length
        }) final results, reducing result count by (${total - Object.keys(mergedResultsCount).length})`;
      debug(message);
      combinedResponse.logs.push(new LogEntry('INFO', null, message).getLog());
    }
    if (Object.keys(combinedResponse.message.results).length) {
      combinedResponse.logs.push(
        new LogEntry(
          'INFO',
          null,
          [
            `Final result count`,
            Object.keys(combinedResponse.message.results).length > this.CREATIVE_LIMIT ? ' (before truncation):' : ':',
            ` ${Object.keys(combinedResponse.message.results).length}`,
          ].join(''),
        ).getLog(),
      );
    }
    const response = combinedResponse as unknown as TrapiResponse;
    // sort records by score
    response.message.results = Object.values(combinedResponse.message.results).sort((a, b) => {
      return b.analyses[0].score - a.analyses[0].score ? b.analyses[0].score - a.analyses[0].score : 0;
    });
    // trim extra results and prune kg
    response.message.results = response.message.results.slice(0, this.CREATIVE_LIMIT);
    response.description = `Query processed successfully, retrieved ${response.message.results.length} results.`;
    this.pruneKnowledgeGraph(response);
    // get the final summary log
    if (successfulQueries) {
      this.parent
        .getSummaryLog(response, response.logs as StampedLog[], resultQueries)
        .forEach((log) => response.logs.push(log));
    }
    response.logs = (response.logs as StampedLog[]).map((log) => log.toJSON());

    return response;
  }
}
