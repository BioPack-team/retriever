import call_api from '@biothings-explorer/call-apis';
import { redisClient } from '@biothings-explorer/utils';
import QEdge2APIEdgeHandler, { APIEdge } from './qedge2apiedge';
import NodesUpdateHandler from './update_nodes';
import Debug from 'debug';
const debug = Debug('bte:biothings-explorer-trapi:batch_edge_query');
import CacheHandler from './cache_handler';
import { threadId } from 'worker_threads';
import MetaKG from '@biothings-explorer/smartapi-kg';
import { StampedLog } from '@biothings-explorer/utils';
import { QueryHandlerOptions } from '@biothings-explorer/types';
import QEdge from './query_edge';
import { UnavailableAPITracker } from './types';
import { Record } from '@biothings-explorer/api-response-transform';

export interface BatchEdgeQueryOptions extends QueryHandlerOptions {
  recordHashEdgeAttributes: string[];
  caching: boolean;
}

export default class BatchEdgeQueryHandler {
  metaKG: MetaKG;
  logs: StampedLog[];
  caching: boolean;
  options: QueryHandlerOptions;
  resolveOutputIDs: boolean;
  qEdges: QEdge | QEdge[];
  constructor(metaKG: MetaKG, resolveOutputIDs = true, options?: BatchEdgeQueryOptions) {
    this.metaKG = metaKG;
    this.logs = [];
    this.caching = options && options.caching;
    this.options = options;
    if (options && options.recordHashEdgeAttributes) {
      this.options.EDGE_ATTRIBUTES_USED_IN_RECORD_HASH = options.recordHashEdgeAttributes;
    }
    this.resolveOutputIDs = resolveOutputIDs;
  }

  /**
   * @param {Array} qEdges - an array of TRAPI Query Edges;
   */
  setEdges(qEdges: QEdge | QEdge[]): void {
    this.qEdges = qEdges;
  }

  /**
   *
   */
  getEdges(): QEdge | QEdge[] {
    return this.qEdges;
  }

  /**
   * @private
   */
  _expandAPIEdges(APIEdges: APIEdge[]): APIEdge[] {
    // debug(`BTE EDGE ${JSON.stringify(this.qEdges)}`);
    return APIEdges;
  }

  /**
   * @private
   */
  async _queryAPIEdges(APIEdges: APIEdge[], unavailableAPIs: UnavailableAPITracker = {}): Promise<Record[]> {
    const executor = new call_api(APIEdges, this.options, redisClient);
    const records: Record[] = await executor.query(this.resolveOutputIDs, unavailableAPIs);
    this.logs = [...this.logs, ...executor.logs];
    return records;
  }

  /**
   * @private
   */
  async _postQueryFilter(records: Record[]): Promise<Record[]> {
    debug(`Filtering out "undefined" items (${records.length}) records`);
    records = records.filter((record) => record !== undefined);
    return records;
  }

  /**
   * Remove curies which resolve to the same thing, keeping the first.
   * @private
   */
  async _rmEquivalentDuplicates(qEdges: QEdge[]): Promise<void> {
    Object.values(qEdges).forEach((qEdge) => {
      const nodes = {
        subject: qEdge.subject,
        object: qEdge.object,
      };
      const strippedCuries: string[] = [];
      Object.entries(nodes).forEach(([, node]) => {
        const reducedCuries: string[] = [];
        const nodeStrippedCuries: string[] = [];
        if (!node.curie) {
          return;
        }
        node.curie.forEach((curie) => {
          // if the curie is already present, or an equivalent is, remove it
          if (!reducedCuries.includes(curie)) {
            const equivalentAlreadyIncluded = qEdge
              .getInputNode()
              .getEquivalentIDs()
            [curie].equivalentIDs.some((equivalentCurie) => reducedCuries.includes(equivalentCurie));
            if (!equivalentAlreadyIncluded) {
              reducedCuries.push(curie);
            } else {
              nodeStrippedCuries.push(curie);
            }
          }
        });
        node.curie = reducedCuries;
        strippedCuries.push(...nodeStrippedCuries);
        if (nodeStrippedCuries.length > 0) {
          debug(
            `stripped (${nodeStrippedCuries.length}) duplicate equivalent curies from ${node.id
            }: ${nodeStrippedCuries.join(',')}`,
          );
        }
      });
      strippedCuries.forEach((curie) => {
        qEdge.getInputNode().removeEquivalentID(curie);
      });
    });
  }

  async query(qEdges: QEdge | QEdge[], unavailableAPIs: UnavailableAPITracker = {}): Promise<Record[]> {
    debug('Node Update Start');
    // it's now a single edge but convert to arr to simplify refactoring
    qEdges = Array.isArray(qEdges) ? qEdges : [qEdges];
    const nodeUpdate = new NodesUpdateHandler(qEdges);
    // difference is there is no previous edge info anymore
    await nodeUpdate.setEquivalentIDs(qEdges);
    await this._rmEquivalentDuplicates(qEdges);
    debug('Node Update Success');

    const cacheHandler = new CacheHandler(this.caching, this.metaKG, this.options);
    const { cachedRecords, nonCachedQEdges } = await cacheHandler.categorizeEdges(qEdges);
    this.logs = [...this.logs, ...cacheHandler.logs];
    let queryRecords: Record[];

    if (nonCachedQEdges.length === 0) {
      queryRecords = [];
      if (global.parentPort) {
        global.parentPort.postMessage({ threadId, cacheDone: true });
      }
    } else {
      debug('Start to convert qEdges into APIEdges....');
      const edgeConverter = new QEdge2APIEdgeHandler(nonCachedQEdges, this.metaKG);
      const APIEdges = await edgeConverter.convert(nonCachedQEdges);
      debug(`qEdges are successfully converted into ${APIEdges.length} APIEdges....`);
      this.logs = [...this.logs, ...edgeConverter.logs];
      if (APIEdges.length === 0 && cachedRecords.length === 0) {
        return [];
      }
      const expanded_APIEdges = this._expandAPIEdges(APIEdges);
      debug('Start to query APIEdges....');
      queryRecords = await this._queryAPIEdges(expanded_APIEdges, unavailableAPIs);
      if (queryRecords === undefined) return;
      debug('APIEdges are successfully queried....');
      queryRecords = await this._postQueryFilter(queryRecords);
      debug(`Total number of records is (${queryRecords.length})`);
      const cacheTask = cacheHandler.cacheEdges(queryRecords);
      if (!(process.env.USE_THREADING === 'false')) {
        global.cachingTasks?.push(cacheTask);
      } else {
        await cacheTask;
      }
    }
    queryRecords = [...queryRecords, ...cachedRecords];
    debug('Start to update nodes...');
    nodeUpdate.update(queryRecords);
    debug('Update nodes completed!');
    return queryRecords;
  }
}
