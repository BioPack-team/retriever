import MetaKG from '@biothings-explorer/smartapi-kg';
import path from 'path';
import QueryGraph from './query_graph';
import KnowledgeGraph from './graph/knowledge_graph';
import TrapiResultsAssembler from './results_assembly/query_results';
import InvalidQueryGraphError from './exceptions/invalid_query_graph_error';
import Debug from 'debug';
const debug = Debug('bte:biothings-explorer-trapi:main');
import Graph from './graph/graph';
import EdgeManager from './edge_manager';
import _ from 'lodash';
import QEdge2APIEdgeHandler from './qedge2apiedge';
import { LogEntry, StampedLog } from '@biothings-explorer/utils';
import { promises as fs } from 'fs';
import { getDescendants } from '@biothings-explorer/node-expansion';
import { resolveSRI, SRINodeNormFailure } from 'biomedical_id_resolver';
import InferredQueryHandler from './inferred_mode/inferred_mode';
import KGNode from './graph/kg_node';
import KGEdge from './graph/kg_edge';
import {
  TrapiAuxGraphCollection,
  TrapiAuxiliaryGraph,
  TrapiQNode,
  TrapiQueryGraph,
  TrapiResponse,
  TrapiResult,
} from './types';
import { QueryHandlerOptions } from '@biothings-explorer/types';
import BTEGraph from './graph/graph';
import QEdge from './query_edge';
import { Telemetry } from '@biothings-explorer/utils';

// Exports for external availability
export * from './types';
export { getTemplates, supportedLookups } from './inferred_mode/template_lookup';
export { default as QEdge } from './query_edge';
export { default as QNode } from './query_node';
export { default as InvalidQueryGraphError } from './exceptions/invalid_query_graph_error';
export * from './qedge2apiedge';

export default class TRAPIQueryHandler {
  logs: StampedLog[];
  options: QueryHandlerOptions;
  includeReasoner: boolean;
  path: string;
  predicatePath: string;
  subclassEdges: { [expandedID: string]: string };
  originalQueryGraph: TrapiQueryGraph;
  bteGraph: BTEGraph;
  knowledgeGraph: KnowledgeGraph;
  trapiResultsAssembler: TrapiResultsAssembler;
  auxGraphs: TrapiAuxGraphCollection;
  finalizedResults: TrapiResult[];
  queryGraph: TrapiQueryGraph;
  constructor(
    options: QueryHandlerOptions = {},
    smartAPIPath: string = undefined,
    predicatesPath: string = undefined,
    includeReasoner = true,
  ) {
    this.logs = [];
    this.options = options;
    this.options.provenanceUsesServiceProvider = this.options.smartAPIID || this.options.teamName ? true : false;
    this.includeReasoner = includeReasoner;
    this.options.resolveOutputIDs =
      typeof this.options.enableIDResolution === 'undefined' ? true : this.options.enableIDResolution;
    this.path = smartAPIPath || path.resolve(__dirname, './smartapi_specs.json');
    this.predicatePath = predicatesPath || path.resolve(__dirname, './predicates.json');
    this.options.apiList && this.findUnregisteredAPIs();
    this.subclassEdges = {};
  }

  async findUnregisteredAPIs() {
    const configListAPIs = this.options.apiList['include'];
    const smartapiRegistry = await fs.readFile(this.path, { encoding: 'utf8' });

    const smartapiIds: string[] = [];
    const inforesIds: string[] = [];
    const unregisteredAPIs: string[] = [];

    // TODO typing for smartapiRegistration
    JSON.parse(smartapiRegistry).hits.forEach((smartapiRegistration) => {
      smartapiIds.push(smartapiRegistration._id);
      inforesIds.push(smartapiRegistration.info?.['x-translator']?.infores);
    });
    configListAPIs.forEach((configListApi) => {
      if (
        smartapiIds.includes(configListApi.id ?? null) === false &&
        inforesIds.includes(configListApi.infores ?? null) === false
      ) {
        unregisteredAPIs.push(configListApi.id ?? configListApi.infores);
        debug(`${configListApi['name']} not found in smartapi registry`);
      }
    });
    return unregisteredAPIs;
  }

  _loadMetaKG(): MetaKG {
    const metaKG = new MetaKG(this.path, this.predicatePath);
    debug(
      `Query options are: ${JSON.stringify({
        ...this.options,
        schema: this.options.schema ? this.options.schema.info.version : 'not included',
      })}`,
    );
    debug(`SmartAPI Specs read from path: ${this.path}`);
    metaKG.constructMetaKGSync(this.includeReasoner, this.options);
    return metaKG;
  }

  createSubclassSupportGraphs(): void {
    const ontologyKnowledgeSourceMapping = {
      GO: 'infores:go',
      DOID: 'infores:disease-ontology',
      MONDO: 'infores:mondo',
      CHEBI: 'infores:chebi',
      HP: 'infores:hpo',
      UMLS: 'infores:umls',
    };

    const qNodesbyOriginalID: { [originalID: string]: Set<string> } = {};
    const originalIDsByPrimaryID: { [primaryID: string]: Set<string> } = {};
    const primaryIDsByOriginalID: { [originalID: string]: string } = {};
    const expandedIDsbyPrimaryID: { [primaryID: string]: Set<string> } = {};
    Object.entries(this.originalQueryGraph.nodes).forEach(([qNodeID, node]) => {
      node.ids?.forEach((id) => {
        if (!Object.keys(qNodesbyOriginalID).includes(id)) {
          qNodesbyOriginalID[id] = new Set();
        }
        qNodesbyOriginalID[id].add(qNodeID);
      });
    });
    Object.values(this.bteGraph.nodes).forEach((node) => {
      Object.keys(qNodesbyOriginalID).forEach((originalID) => {
        if (node.curies.includes(originalID)) {
          if (!originalIDsByPrimaryID[node.primaryCurie]) {
            originalIDsByPrimaryID[node.primaryCurie] = new Set();
          }
          originalIDsByPrimaryID[node.primaryCurie].add(originalID);
          primaryIDsByOriginalID[originalID] = node.primaryCurie;
        }
      });
      Object.keys(this.subclassEdges).forEach((expandedID) => {
        if (node.curies.includes(expandedID)) {
          if (!expandedIDsbyPrimaryID[node.primaryCurie]) {
            expandedIDsbyPrimaryID[node.primaryCurie] = new Set();
          }
          expandedIDsbyPrimaryID[node.primaryCurie].add(expandedID);
        }
      });
    });

    // Create subclass edges for nodes that were expanded
    const nodesToRebind: { [nodeID: string]: { newNode: string; subclassEdgeID: string } } = {};
    Object.keys(this.bteGraph.nodes).forEach((nodeID) => {
      const subclassCuries = [...(expandedIDsbyPrimaryID[nodeID] ?? [])]?.map((expandedID) => [
        this.subclassEdges[expandedID],
        expandedID,
      ]);
      if (!subclassCuries.length) return; // Nothing to rebind
      subclassCuries.forEach(([original, expanded]) => {
        const subject = nodeID;
        const object = primaryIDsByOriginalID[original];
        // Don't keep self-subclass
        if (subject === object) return;
        const subclassEdgeID = `expanded-${subject}-subclass_of-${object}`;
        if (subclassEdgeID in this.bteGraph.edges) return;
        const subclassEdge = new KGEdge(subclassEdgeID, {
          predicate: 'biolink:subclass_of',
          subject,
          object,
        });
        const source = Object.entries(ontologyKnowledgeSourceMapping).find(([prefix]) => {
          return expanded.includes(prefix);
        })[1];
        subclassEdge.addSource([
          { resource_id: source, resource_role: 'primary_knowledge_source' },
          {
            resource_id: this.options.provenanceUsesServiceProvider
              ? 'infores:service-provider-trapi'
              : 'infores:biothings-explorer',
            resource_role: 'aggregator_knowledge_source',
          },
        ]);
        this.bteGraph.edges[subclassEdgeID] = subclassEdge;
        nodesToRebind[subject] = { newNode: object, subclassEdgeID };
      });
    });

    // Create new constructed edges and aux graphs for edges that used subclass edges
    let auxGraphs: { [supportGraphID: string]: TrapiAuxiliaryGraph } = {};
    const edgesToRebind = {};
    const edgesIDsByAuxGraphID = {};
    Object.entries(this.bteGraph.edges).forEach(([edgeID, bteEdge]) => {
      if (edgeID.includes('expanded')) return;
      const supportGraph = [edgeID];
      const [subject, object] = [bteEdge.subject, bteEdge.object].map((edgeNodeID) => {
        if (!nodesToRebind[edgeNodeID]) {
          return edgeNodeID; // nothing to rebind
        }
        supportGraph.push(nodesToRebind[edgeNodeID].subclassEdgeID);
        return nodesToRebind[edgeNodeID].newNode;
      });

      if (supportGraph.length === 1) return; // no subclasses
      const boundEdgeID = `${subject}-${bteEdge.predicate.replace('biolink:', '')}-${object}-via_subclass`;
      let suffix = 0;
      while (Object.keys(auxGraphs).includes(`support${suffix}-${boundEdgeID}`)) {
        suffix += 1;
      }
      const supportGraphID = `support${suffix}-${boundEdgeID}`;
      auxGraphs[supportGraphID] = { edges: supportGraph };
      if (!edgesIDsByAuxGraphID[supportGraphID]) {
        edgesIDsByAuxGraphID[supportGraphID] = new Set();
      }
      edgesIDsByAuxGraphID[supportGraphID].add(boundEdgeID);
      if (!this.bteGraph.edges[boundEdgeID]) {
        const boundEdge = new KGEdge(boundEdgeID, {
          predicate: bteEdge.predicate,
          subject: subject,
          object: object,
        });
        boundEdge.addAdditionalAttributes('biolink:support_graphs', [supportGraphID]);
        boundEdge.addSource([
          {
            resource_id: this.options.provenanceUsesServiceProvider
              ? 'infores:service-provider-trapi'
              : 'infores:biothings-explorer',
            resource_role: 'primary_knowledge_source',
          },
        ]);
        this.bteGraph.edges[boundEdgeID] = boundEdge;
      } else {
        (this.bteGraph.edges[boundEdgeID].attributes['biolink:support_graphs'] as Set<string>).add(supportGraphID);
      }
      edgesToRebind[edgeID] = boundEdgeID;
    });

    const resultBoundEdgesWithAuxGraphs = new Set();
    const fixedResults = this.trapiResultsAssembler.getResults().map((result) => {
      result.node_bindings = Object.fromEntries(
        Object.entries(result.node_bindings).map(([qNodeID, bindings]) => {
          return [
            qNodeID,
            bindings.reduce(
              ({ boundIDs, newBindings }, binding) => {
                if (!nodesToRebind[binding.id]) {
                  if (!boundIDs.has(binding.id)) {
                    newBindings.push(binding);
                    boundIDs.add(binding.id);
                  }
                } else if (!boundIDs.has(nodesToRebind[binding.id].newNode)) {
                  newBindings.push({ id: nodesToRebind[binding.id].newNode });
                  boundIDs.add(nodesToRebind[binding.id].newNode);
                }
                return { boundIDs, newBindings };
              },
              { boundIDs: new Set(), newBindings: [] },
            ).newBindings,
          ];
        }),
      );
      result.analyses[0].edge_bindings = Object.fromEntries(
        Object.entries(result.analyses[0].edge_bindings).map(([qEdgeID, bindings]) => {
          return [
            qEdgeID,
            bindings.reduce(
              ({ boundIDs, newBindings }, binding) => {
                if (!edgesToRebind[binding.id]) {
                  if (!boundIDs.has(binding.id)) {
                    newBindings.push(binding);
                    boundIDs.add(binding.id);
                  }
                } else if (!boundIDs.has(edgesToRebind[binding.id])) {
                  newBindings.push({ id: edgesToRebind[binding.id] });
                  boundIDs.add(edgesToRebind[binding.id]);
                  resultBoundEdgesWithAuxGraphs.add(edgesToRebind[binding.id]);
                }
                return { boundIDs, newBindings };
              },
              { boundIDs: new Set(), newBindings: [] },
            ).newBindings,
          ];
        }),
      );
      return result;
    });

    // Prune unused auxGraphs
    auxGraphs = Object.fromEntries(
      Object.entries(auxGraphs).filter(([auxGraphID]) => {
        return [...edgesIDsByAuxGraphID[auxGraphID]].some((edgeID) => resultBoundEdgesWithAuxGraphs.has(edgeID));
      }),
    );

    this.auxGraphs = auxGraphs;
    this.finalizedResults = fixedResults;
  }

  async addQueryNodes(): Promise<void> {
    const qNodeIDsByOriginalID: Map<string, TrapiQNode> = new Map();
    const curiesToResolve = [
      ...Object.values(this.queryGraph.nodes).reduce((set: Set<string>, qNode) => {
        qNode.ids?.forEach((id) => {
          set.add(id);
          qNodeIDsByOriginalID.set(id, qNode);
        });
        return set;
      }, new Set()),
    ] as string[];
    const resolvedCuries = await resolveSRI({ unknown: curiesToResolve });
    Object.entries(resolvedCuries).forEach(([originalCurie, resolvedEntity]) => {
      if (!this.bteGraph.nodes[resolvedEntity.primaryID]) {
        const category = resolvedEntity.primaryTypes?.[0]
          ? `biolink:${resolvedEntity.primaryTypes[0]}`
          : qNodeIDsByOriginalID.get(originalCurie).categories?.[0];

        this.bteGraph.nodes[resolvedEntity.primaryID] = new KGNode(resolvedEntity.primaryID, {
          primaryCurie: resolvedEntity.primaryID,
          qNodeID: qNodeIDsByOriginalID[originalCurie],
          curies: resolvedEntity.equivalentIDs,
          names: resolvedEntity.labelAliases,
          semanticType: category ? [category] : ['biolink:NamedThing'],
          label: resolvedEntity.label,
        });
      }
    });
  }

  getResponse(): TrapiResponse {
    const results = this.finalizedResults ?? [];
    return {
      description: `Query processed successfully, retrieved ${results.length} results.`,
      schema_version: global.SCHEMA_VERSION,
      biolink_version: global.BIOLINK_VERSION,
      workflow: [{ id: 'lookup' }],
      message: {
        query_graph: this.originalQueryGraph,
        knowledge_graph: this.knowledgeGraph.kg,
        auxiliary_graphs: this.auxGraphs,
        results: results,
      },
      logs: this.logs.map((log) => log.toJSON()),
    };
  }

  /**
   * Set TRAPI Query Graph
   * @param { object } queryGraph - TRAPI Query Graph Object
   */
  setQueryGraph(queryGraph: TrapiQueryGraph): void {
    this.originalQueryGraph = _.cloneDeep(queryGraph);
    this.queryGraph = queryGraph;
    for (const nodeId in queryGraph.nodes) {
      // perform node expansion
      if (queryGraph.nodes[nodeId].ids && !this._queryUsesInferredMode()) {
        const descendantsByCurie: { [curie: string]: string[] } = getDescendants(queryGraph.nodes[nodeId].ids);
        let expanded = Object.values(descendantsByCurie).flat();

        expanded = _.uniq([...queryGraph.nodes[nodeId].ids, ...expanded]);

        let log_msg = `Expanded ids for node ${nodeId}: (${queryGraph.nodes[nodeId].ids.length} ids -> ${expanded.length} ids)`;
        debug(log_msg);
        this.logs.push(new LogEntry('INFO', null, log_msg).getLog());

        const foundExpandedIds = expanded.length > queryGraph.nodes[nodeId].ids.length;

        if (foundExpandedIds) {
          Object.entries(descendantsByCurie).forEach(([curie, descendants]) => {
            descendants.forEach((descendant) => {
              if (queryGraph.nodes[nodeId].ids.includes(descendant)) return;
              this.subclassEdges[descendant] = curie;
            });
          });
        }

        queryGraph.nodes[nodeId].ids = expanded;

        const nodeMissingIsSet = !queryGraph.nodes[nodeId].is_set;

        // make sure is_set is true
        if (foundExpandedIds && nodeMissingIsSet) {
          queryGraph.nodes[nodeId].is_set = true;
          log_msg = `Added is_set:true to node ${nodeId}`;
          debug(log_msg);
          this.logs.push(new LogEntry('INFO', null, log_msg).getLog());
        }
      }
    }
  }

  _initializeResponse(): void {
    this.knowledgeGraph = new KnowledgeGraph(this.options?.apiList?.include);
    this.trapiResultsAssembler = new TrapiResultsAssembler(this.options);
    this.bteGraph = new Graph();
    this.bteGraph.subscribe(this.knowledgeGraph);
  }

  async _processQueryGraph(queryGraph: TrapiQueryGraph): Promise<QEdge[]> {
    try {
      const queryGraphHandler = new QueryGraph(queryGraph, this.options.schema);
      const queryEdges = await queryGraphHandler.calculateEdges();
      this.logs = [...this.logs, ...queryGraphHandler.logs];
      return queryEdges;
    } catch (err) {
      if (err instanceof InvalidQueryGraphError || err instanceof SRINodeNormFailure) {
        throw err;
      } else {
        console.log(err.stack);
        throw new InvalidQueryGraphError();
      }
    }
  }

  async _edgesSupported(qEdges: QEdge[], metaKG: MetaKG): Promise<boolean> {
    if (this.options.dryrun) {
      const log_msg =
        'Running dryrun of query, no API calls will be performed. Actual query execution order may vary based on API responses received.';
      this.logs.push(new LogEntry('INFO', null, log_msg).getLog());
    }

    // _.cloneDeep() is resource-intensive but only runs once per query
    qEdges = _.cloneDeep(qEdges);
    const manager = new EdgeManager(qEdges, metaKG, this.options);
    const qEdgesMissingOps: { [qEdgeID: string]: boolean } = {};
    while (manager.getEdgesNotExecuted()) {
      const currentQEdge = manager.getNext();
      const edgeConverter = new QEdge2APIEdgeHandler([currentQEdge], metaKG);
      const metaXEdges = await edgeConverter.getMetaXEdges(currentQEdge);

      if (this.options.dryrun) {
        const apiNames = [...new Set(metaXEdges.map((metaXEdge) => metaXEdge.association.api_name))];

        let log_msg: string;
        if (currentQEdge.reverse) {
          log_msg = `qEdge ${currentQEdge.id} (reversed): ${currentQEdge.object.categories} > ${currentQEdge.predicate ? `${currentQEdge.predicate} > ` : ''
            }${currentQEdge.subject.categories}`;
        } else {
          log_msg = `qEdge ${currentQEdge.id}: ${currentQEdge.subject.categories} > ${currentQEdge.predicate ? `${currentQEdge.predicate} > ` : ''
            }${currentQEdge.object.categories}`;
        }
        this.logs.push(new LogEntry('INFO', null, log_msg).getLog());

        if (metaXEdges.length) {
          const log_msg_2 = `${metaXEdges.length} total planned queries to following APIs: ${apiNames.join(',')}`;
          this.logs.push(new LogEntry('INFO', null, log_msg_2).getLog());
        }

        metaXEdges.forEach((metaXEdge) => {
          log_msg = `${metaXEdge.association.api_name}: ${metaXEdge.association.input_type} > ${metaXEdge.association.predicate} > ${metaXEdge.association.output_type}`;
          this.logs.push(new LogEntry('DEBUG', null, log_msg).getLog());
        });
      }

      if (!metaXEdges.length) {
        qEdgesMissingOps[currentQEdge.id] = currentQEdge.reverse;
      }
      // assume results so next edge may be reversed or not
      currentQEdge.executed = true;

      //use # of APIs as estimate of # of records
      if (metaXEdges.length) {
        if (currentQEdge.reverse) {
          currentQEdge.subject.entity_count = currentQEdge.object.entity_count * metaXEdges.length;
        } else {
          currentQEdge.object.entity_count = currentQEdge.subject.entity_count * metaXEdges.length;
        }
      } else {
        currentQEdge.object.entity_count = 1;
        currentQEdge.subject.entity_count = 1;
      }
    }

    const len = Object.keys(qEdgesMissingOps).length;
    // this.logs = [...this.logs, ...manager.logs];
    const qEdgesToLog = Object.entries(qEdgesMissingOps).map(([qEdge, reversed]) => {
      return reversed ? `(reversed ${qEdge})` : `(${qEdge})`;
    });
    const qEdgesLogStr = qEdgesToLog.length > 1 ? `[${qEdgesToLog.join(', ')}]` : `${qEdgesToLog.join(', ')}`;
    if (len > 0) {
      const terminateLog = `Query Edge${len !== 1 ? 's' : ''} ${qEdgesLogStr} ${len !== 1 ? 'have' : 'has'
        } no MetaKG edges. Your query terminates.`;
      debug(terminateLog);
      this.logs.push(new LogEntry('WARNING', null, terminateLog).getLog());
      return false;
    } else {
      if (this.options.dryrun) {
        return false;
      }
      return true;
    }
  }

  _queryUsesInferredMode(): boolean {
    const inferredEdge = Object.values(this.queryGraph.edges).some((edge) => edge.knowledge_type === 'inferred');
    return inferredEdge;
  }

  _queryIsOneHop(): boolean {
    const oneHop = Object.keys(this.queryGraph.edges).length === 1;
    return oneHop;
  }

  async _handleInferredEdges(): Promise<void> {
    if (!this._queryIsOneHop()) {
      const message = 'Inferred Mode edges are only supported in single-edge queries. Your query terminates.';
      debug(message);
      this.logs.push(new LogEntry('WARNING', null, message).getLog());
      return;
    }
    const inferredQueryHandler = new InferredQueryHandler(
      this,
      this.queryGraph,
      this.logs,
      this.options,
      this.path,
      this.predicatePath,
      this.includeReasoner,
    );
    const inferredQueryResponse = await inferredQueryHandler.query();
    if (inferredQueryResponse) {
      this.getResponse = () => inferredQueryResponse;
    }
  }

  async _checkContraints(): Promise<boolean> {
    const constraints: Set<string> = new Set();
    Object.values(this.queryGraph).forEach((item) => {
      Object.values(item).forEach((element: any) => {
        element.constraints?.forEach((constraint: { name: string }) => constraints.add(constraint.name));
        element.attribute_constraints?.forEach((constraint: { name: string }) => constraints.add(constraint.name));
        // element.qualifier_constraints?.forEach((constraint) => constraints.add(constraint.name));
      });
    });
    if (constraints.size) {
      this.logs.push(
        new LogEntry(
          'ERROR',
          'UnsupportedAttributeConstraint',
          `Unsupported Attribute Constraints: [${[...constraints].join(', ')}]`,
        ).getLog(),
      );
      this.logs.push(
        new LogEntry(
          'ERROR',
          null,
          `BTE does not currently support any type of constraint. Your query Terminates.`,
        ).getLog(),
      );
      return true;
    }
  }

  getSummaryLog = (
    response: TrapiResponse,
    logs: StampedLog[],
    resultTemplates: number[] = undefined,
  ): StampedLog[] => {
    const KGNodes = Object.keys(response.message.knowledge_graph.nodes).length;
    const kgEdges = Object.keys(response.message.knowledge_graph.edges).length;
    const results = response.message.results.length;
    const resultQueries = logs.filter(({ message, data }) => {
      const correctType = data?.type === 'query' && data?.hits;
      if (resultTemplates) {
        return correctType && resultTemplates.some((queryIndex) => message.includes(`[Template-${queryIndex + 1}]`));
      }
      return correctType;
    }).length;
    const queries = logs.filter(({ data }) => data?.type === 'query').length;
    const query_sources = logs
      .filter(({ message, data }) => {
        const correctType = data?.type === 'query' && data?.hits;
        if (resultTemplates) {
          return correctType && resultTemplates.some((queryIndex) => message.includes(`[Template-${queryIndex + 1}]`));
        }
        return correctType;
      })
      .map(({ data }) => data?.api_name);
    const cache_sources = logs
      .filter(({ message, data }) => {
        const correctType = data?.type === 'cacheHit';
        if (resultTemplates) {
          return correctType && resultTemplates.some((queryIndex) => message.includes(`[Template-${queryIndex + 1}]`));
        }
        return correctType;
      })
      .map(({ data }) => data?.api_names)
      .flat();
    const sources = [...new Set(query_sources.concat(cache_sources))];
    const cached = logs.filter(({ data }) => data?.type === 'cacheHit').length;

    return [
      new LogEntry(
        'INFO',
        null,
        `Execution Summary: (${KGNodes}) nodes / (${kgEdges}) edges / (${results}) results; (${resultQueries}/${queries}) queries${cached ? ` (${cached} cached qEdges)` : ''
        } returned results from(${sources.length}) unique API${sources.length === 1 ? 's' : ''}`,
      ).getLog(),
      new LogEntry('INFO', null, `APIs: ${sources.join(', ')} `).getLog(),
    ];
  };

  async query(): Promise<void> {
    this._initializeResponse();
    await this.addQueryNodes();

    const span1 = Telemetry.startSpan({ description: 'loadMetaKG' });

    debug('Start to load metakg.');
    const metaKG = this._loadMetaKG();
    if (!metaKG.ops.length) {
      let error: string;
      if (this.options.smartAPIID) {
        error = `Specified SmartAPI ID(${this.options.smartAPIID}) is either invalid or missing.`;
      } else if (this.options.teamName) {
        error = `Specified Team(${this.options.teamName}) is either invalid or missing.`;
      } else {
        error = `Something has gone wrong and the MetaKG is empty.Please try again later.If this persists, please contact the server admin.`;
      }
      this.logs.push(new LogEntry('ERROR', null, error).getLog());
      return;
    }
    debug('MetaKG successfully loaded!');

    span1?.finish();

    if (global.missingAPIs) {
      this.logs.push(
        new LogEntry(
          'WARNING',
          null,
          `The following APIs were unavailable at the time of execution: ${global.missingAPIs
            .map((spec) => spec.info.title)
            .join(', ')}`,
        ).getLog(),
      );
    }

    const queryEdges = await this._processQueryGraph(this.queryGraph);
    // TODO remove this when constraints implemented
    if (await this._checkContraints()) {
      return;
    }
    if ((this.options.smartAPIID || this.options.teamName) && Object.values(this.queryGraph.edges).length > 1) {
      const message = 'smartAPI/team-specific endpoints only support single-edge queries. Your query terminates.';
      this.logs.push(new LogEntry('WARNING', null, message).getLog());
      debug(message);
      return;
    }
    debug(`(3) All edges created ${JSON.stringify(queryEdges)} `);

    if (this._queryUsesInferredMode()) {
      const span2 = Telemetry.startSpan({ description: 'creativeExecution' });
      await this._handleInferredEdges();
      span2?.finish();
      return;
    }

    if (!(await this._edgesSupported(queryEdges, metaKG))) {
      return;
    }
    const manager = new EdgeManager(queryEdges, metaKG, this.options);

    const executionSuccess = await manager.executeEdges();
    this.logs = [...this.logs, ...manager.logs];
    if (!executionSuccess) {
      return;
    }

    const span3 = Telemetry.startSpan({ description: 'resultsAssembly' });

    // update query graph
    this.bteGraph.update(manager.getRecords());
    //update query results
    await this.trapiResultsAssembler.update(
      manager.getOrganizedRecords(),
      !(this.options.smartAPIID || this.options.teamName),
    );
    this.logs = [...this.logs, ...this.trapiResultsAssembler.logs];
    // fix subclassing
    this.createSubclassSupportGraphs();
    // prune bteGraph
    this.bteGraph.prune(this.finalizedResults, this.auxGraphs);
    this.bteGraph.notify();

    span3?.finish();

    // check primary knowledge sources
    this.logs = [...this.logs, ...this.bteGraph.checkPrimaryKnowledgeSources(this.knowledgeGraph)];
    // finishing logs
    this.getSummaryLog(this.getResponse(), this.logs).forEach((log) => this.logs.push(log));
    debug(`(14) TRAPI query finished.`);
  }
}
