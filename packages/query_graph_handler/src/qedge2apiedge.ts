import _ from 'lodash';
import { LogEntry, StampedLog } from '@retriever/utils';
import * as config from './config';
const CURIE_WITH_PREFIXES = ['MONDO', 'DOID', 'UBERON', 'EFO', 'HP', 'CHEBI', 'CL', 'MGI', 'NCIT'];
import Debug from 'debug';
import { APIEdge, BatchAPIEdge, MetaXEdge, NonBatchAPIEdge, QEdge, TemplateBatchAPIEdge, TemplateNonBatchAPIEdge } from '@retriever/graph';
import MetaKG from '@retriever/smartapi-kg';
import { SmartAPIKGOperationObject } from '@retriever/smartapi-kg';
import { SRIBioEntity } from 'biomedical_id_resolver';
const debug = Debug('retriever:qedge2btedge');


export default class QEdge2APIEdgeHandler {
  qEdges: QEdge[];
  metaKG: MetaKG;
  logs: StampedLog[];
  constructor(qEdges: QEdge[], metaKG: MetaKG) {
    this.qEdges = qEdges;
    this.metaKG = metaKG;
    this.logs = [];
  }

  setQEdges(qEdges: QEdge[]): void {
    this.qEdges = qEdges;
  }

  _findAPIsFromMetaEdges(metaEdges: SmartAPIKGOperationObject[]): string[] {
    return metaEdges.map((edge) => edge.association.api_name);
  }

  /**
   * Get SmartAPI Edges based on TRAPI Query Edge.
   */
  async getMetaXEdges(qEdge: QEdge, metaKG: MetaKG = this.metaKG): Promise<MetaXEdge[]> {
    debug(`Input node is ${qEdge.getInputNode().id}`);
    debug(`Output node is ${qEdge.getOutputNode().id}`);
    this.logs.push(
      new LogEntry(
        'DEBUG',
        null,
        `BTE is trying to find metaKG edges (smartAPI registry, x-bte annotation) connecting from ${qEdge
          .getInputNode()
          .getCategories()} to ${qEdge.getOutputNode().getCategories()} with predicate ${qEdge.getPredicate()}`,
      ).getLog(),
    );
    const filterCriteria = {
      input_type: qEdge.getInputNode().getCategories(),
      output_type: qEdge.getOutputNode().getCategories(),
      predicate: qEdge.getPredicate(),
      qualifiers: qEdge.getSimpleExpandedQualifierConstraints(),
    };
    debug(`KG Filters: ${JSON.stringify(filterCriteria, null, 2)}`);
    const metaXEdges = metaKG.filter(filterCriteria).map((metaEdge) => {
      (metaEdge as MetaXEdge).reasoner_edge = qEdge;
      return metaEdge as MetaXEdge;
    });
    if (metaXEdges.length === 0) {
      debug(`No smartapi edge found for ${qEdge.getID()}`);
      this.logs.push(
        new LogEntry('WARNING', null, `BTE didn't find any metaKG edges corresponding to ${qEdge.getID()}`).getLog(),
      );
    } else {
      this.logs.push(
        new LogEntry(
          'DEBUG',
          null,
          [
            `BTE found ${metaXEdges.length} metaKG edges corresponding to`,
            `${qEdge.getID()}. These metaKG edges comes from`,
            `${new Set(this._findAPIsFromMetaEdges(metaXEdges)).size} unique APIs.`,
            `They are ${Array.from(new Set(this._findAPIsFromMetaEdges(metaXEdges))).join(',')}`,
          ].join(' '),
        ).getLog(),
      );
    }
    return metaXEdges;
  }

  async _createNonBatchSupportAPIEdges(metaXEdge: MetaXEdge): Promise<NonBatchAPIEdge[]> {
    const APIEdges: NonBatchAPIEdge[] = [];
    const inputPrefix = metaXEdge.association.input_id;
    const inputType = metaXEdge.association.input_type;
    const resolvedCuries = metaXEdge.reasoner_edge.getInputNode().getEquivalentIDs();
    Object.entries(resolvedCuries).forEach(([curie, entity]) => {
      if (entity.primaryTypes.includes(inputType.replace('biolink:', ''))) {
        entity.equivalentIDs.forEach((equivalentCurie) => {
          // inputPrefix is only string[] when MetaXEdge is TRAPI
          if (equivalentCurie.includes(inputPrefix as string)) {
            const id = CURIE_WITH_PREFIXES.includes(equivalentCurie.split(':')[0])
              ? equivalentCurie
              : equivalentCurie.split(':').slice(1).join(':');
            const APIEdge: NonBatchAPIEdge = {
              ...metaXEdge,
              input: id,
              input_resolved_identifiers: {
                [curie]: entity,
              },
              original_input: {
                [equivalentCurie]: curie,
              },
            };
            const edgeToBePushed = APIEdge;
            edgeToBePushed.reasoner_edge = metaXEdge.reasoner_edge;
            APIEdges.push(edgeToBePushed);
          }
        });
      }
    });
    return APIEdges;
  }

  async _createBatchSupportAPIEdges(metaXEdge: MetaXEdge): Promise<BatchAPIEdge[]> {
    const id_mapping: {
      [equivalentCurie: string]: string;
    } = {};
    const inputs: string[] = [];
    const APIEdges: BatchAPIEdge[] = [];
    const input_resolved_identifiers: {
      [curie: string]: SRIBioEntity;
    } = {};
    const inputPrefix = metaXEdge.association.input_id;
    const inputType = metaXEdge.association.input_type;
    const resolvedCuries = metaXEdge.reasoner_edge.getInputNode().getEquivalentIDs();
    // debug(`Resolved ids: ${JSON.stringify(resolvedIDs)}`);
    debug(`Input prefix: ${inputPrefix}`);
    Object.entries(resolvedCuries).forEach(([curie, entity]) => {
      if (metaXEdge.tags.includes('bte-trapi')) {
        if (entity.primaryTypes.includes(inputType.replace('biolink:', ''))) {
          input_resolved_identifiers[curie] = entity;
          inputs.push(entity.primaryID);
          id_mapping[entity.primaryID] = curie;
        }
      } else if (entity.primaryTypes.includes(inputType.replace('biolink:', ''))) {
        entity.equivalentIDs.forEach((equivalentCurie) => {
          // inputPrefix is only string[] when MetaXEdge is TRAPI
          if (equivalentCurie.includes(inputPrefix as string)) {
            const id = CURIE_WITH_PREFIXES.includes(equivalentCurie.split(':')[0])
              ? equivalentCurie
              : equivalentCurie.split(':').slice(1).join(':');
            id_mapping[equivalentCurie] = curie;
            input_resolved_identifiers[curie] = entity;
            inputs.push(id);
          }
        });
      }
    });

    let batchSize = Infinity;
    if (metaXEdge.tags.includes('biothings')) {
      batchSize = 1000;
    }
    let configuredLimit = metaXEdge.query_operation.batchSize;
    if (metaXEdge.association['x-trapi']?.batch_size_limit) {
      configuredLimit =
        metaXEdge.association['x-trapi'].batch_size_limit < configuredLimit || !configuredLimit
          ? metaXEdge.association['x-trapi'].batch_size_limit
          : configuredLimit;
    }
    const hardLimit = config.API_BATCH_SIZE.find((api) => {
      return api.id === metaXEdge.association.smartapi.id || api.name === metaXEdge.association.api_name;
    });
    // BTE internal configured limit takes precedence over annotated limit
    batchSize = hardLimit ? hardLimit.max : configuredLimit ? configuredLimit : batchSize;
    if (Object.keys(id_mapping).length > 0) {
      _.chunk(inputs, batchSize).forEach((chunk) => {
        const APIEdge: BatchAPIEdge = {
          ...metaXEdge,
          input: chunk,
          input_resolved_identifiers: input_resolved_identifiers,
          original_input: id_mapping,
        };
        const edgeToBePushed = APIEdge;
        edgeToBePushed.reasoner_edge = metaXEdge.reasoner_edge;
        APIEdges.push(edgeToBePushed);
      });
    }
    return APIEdges;
  }

  /**
   * @private
   * @param {object} resolvedIDs
   * @param {object} metaXEdge
   */
  async _createTemplatedNonBatchSupportAPIEdges(metaXEdge: MetaXEdge): Promise<TemplateNonBatchAPIEdge[]> {
    const APIEdges: TemplateNonBatchAPIEdge[] = [];
    const inputPrefix = metaXEdge.association.input_id;
    const inputType = metaXEdge.association.input_type;
    const resolvedCuries = metaXEdge.reasoner_edge.getInputNode().getEquivalentIDs();
    Object.entries(resolvedCuries).forEach(([curie, entity]) => {
      if (entity.primaryTypes.includes(inputType.replace('biolink:', ''))) {
        entity.equivalentIDs.forEach((equivalentCurie) => {
          // inputPrefix is only string[] for TRAPI; templates are never used for TRAPI
          if (equivalentCurie.includes(inputPrefix as string)) {
            const id = CURIE_WITH_PREFIXES.includes(equivalentCurie.split(':')[0])
              ? equivalentCurie
              : equivalentCurie.split(':').slice(1).join(':');
            const APIEdge: TemplateNonBatchAPIEdge = {
              ...metaXEdge,
              input: { queryInputs: id, ...metaXEdge.query_operation.templateInputs },
              input_resolved_identifiers: {
                [curie]: entity,
              },
              original_input: {
                [equivalentCurie]: curie,
              },
            };
            const edgeToBePushed = APIEdge;
            edgeToBePushed.reasoner_edge = metaXEdge.reasoner_edge;
            APIEdges.push(edgeToBePushed);
          }
        });
      }
    });
    return APIEdges;
  }

  /**
   * @private
   * @param {object} resolvedIDs
   * @param {object} metaXEdge
   */
  async _createTemplatedBatchSupportAPIEdges(metaXEdge: MetaXEdge): Promise<TemplateBatchAPIEdge[]> {
    const id_mapping: { [equivalentCurie: string]: string } = {};
    const inputs: string[] = [];
    const APIEdges: TemplateBatchAPIEdge[] = [];
    const input_resolved_identifiers: { [curie: string]: SRIBioEntity } = {};
    const inputPrefix = metaXEdge.association.input_id;
    const inputType = metaXEdge.association.input_type;
    const resolvedCuries = metaXEdge.reasoner_edge.getInputNode().getEquivalentIDs();
    // debug(`Resolved ids: ${JSON.stringify(resolvedIDs)}`);
    debug(`Input prefix: ${inputPrefix}`);
    Object.entries(resolvedCuries).forEach(([curie, entity]) => {
      if (metaXEdge.tags.includes('bte-trapi')) {
        if (entity.primaryTypes.includes(inputType.replace('biolink:', ''))) {
          input_resolved_identifiers[curie] = entity;
          inputs.push(entity.primaryID);
          id_mapping[entity.primaryID] = curie;
        }
      } else if (entity.primaryTypes.includes(inputType.replace('biolink:', ''))) {
        entity.equivalentIDs.forEach((equivalentCurie) => {
          // inputPrefix is only string[] for TRAPI; templates are never used for TRAPI
          if (equivalentCurie.includes(inputPrefix as string)) {
            const id = CURIE_WITH_PREFIXES.includes(equivalentCurie.split(':')[0])
              ? equivalentCurie
              : equivalentCurie.split(':').slice(1).join(':');
            id_mapping[equivalentCurie] = curie;
            input_resolved_identifiers[curie] = entity;
            inputs.push(id);
          }
        });
      }
    });
    let batchSize = Infinity;
    if (metaXEdge.tags.includes('biothings')) {
      batchSize = 1000;
    }
    const configuredLimit = metaXEdge.query_operation.batchSize;
    const hardLimit = config.API_BATCH_SIZE.find((api) => {
      return api.id === metaXEdge.association.smartapi.id || api.name === metaXEdge.association.api_name;
    });
    // BTE internal configured limit takes precedence over annotated limit
    batchSize = hardLimit ? hardLimit.max : configuredLimit ? configuredLimit : batchSize;
    if (Object.keys(id_mapping).length > 0) {
      _.chunk(inputs, batchSize).forEach((chunk) => {
        const APIEdge = {
          ...metaXEdge,
          input: { queryInputs: chunk, ...metaXEdge.query_operation.templateInputs },
          input_resolved_identifiers: input_resolved_identifiers,
          original_input: id_mapping,
        };
        const edgeToBePushed = APIEdge;
        edgeToBePushed.reasoner_edge = metaXEdge.reasoner_edge;
        APIEdges.push(edgeToBePushed);
      });
    }
    return APIEdges;
  }

  /**
   * Add inputs to smartapi edges
   */
  async _createAPIEdges(metaXEdge: MetaXEdge): Promise<APIEdge[]> {
    const supportBatch = metaXEdge.query_operation.supportBatch;
    const useTemplating = metaXEdge.query_operation.useTemplating;
    let APIEdges: APIEdge[];
    if (supportBatch === false) {
      APIEdges = useTemplating
        ? await this._createTemplatedNonBatchSupportAPIEdges(metaXEdge)
        : await this._createNonBatchSupportAPIEdges(metaXEdge);
    } else {
      APIEdges = useTemplating
        ? await this._createTemplatedBatchSupportAPIEdges(metaXEdge)
        : await this._createBatchSupportAPIEdges(metaXEdge);
    }
    return APIEdges;
  }

  async convert(qEdges: QEdge[]): Promise<APIEdge[]> {
    let APIEdges = [];
    await Promise.all(
      qEdges.map(async (qEdge) => {
        const metaXedges = await this.getMetaXEdges(qEdge);
        const apis = _.uniq(metaXedges.map((api) => api.association.api_name));
        debug(`${apis.length} APIs being used:`, JSON.stringify(apis));
        debug(`${metaXedges.length} SmartAPI edges are retrieved....`);
        await Promise.all(
          metaXedges.map(async (metaXEdge) => {
            const newEdges = await this._createAPIEdges(metaXEdge);
            debug(`${newEdges.length} metaKG are created....`);
            // Not sure what this was meant to do?
            // newEdges = newEdges.map((e) => {
            //   e.filter = qEdge.filter;
            //   return e;
            // });
            APIEdges = [...APIEdges, ...newEdges];
          }),
        );
      }),
    );
    if (APIEdges.length === 0) {
      debug(`No metaKG found for this query batch.`);
      this.logs.push(
        new LogEntry('WARNING', null, `BTE didn't find any metaKG for this batch. Your query terminates.`).getLog(),
      );
    } else {
      debug(`BTE found ${APIEdges.length} metaKG for this batch.`);
      this.logs.push(new LogEntry('DEBUG', null, `BTE found ${APIEdges.length} metaKG for this batch.`).getLog());
    }
    return APIEdges;
  }
}
