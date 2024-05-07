import { constructQueries } from '@retriever/call-apis';
import { LogEntry, SerializableLog } from '@retriever/utils';
import QEdge2APIEdgeHandler from './qedge2apiedge';
import { APIEdge, RecordPackage } from '@retriever/graph';
import NodesUpdateHandler from './update_nodes';
import Debug from 'debug';
const debug = Debug('retriever:batch_edge_query');
import CacheHandler from './cache_handler';
import { threadId } from 'worker_threads';
import MetaKG from '@retriever/smartapi-kg';
import { StampedLog } from '@retriever/utils';
import { QueryHandlerOptions, ThreadMessage } from '@retriever/types';
import { Record, QEdge } from '@retriever/graph';
import { UnavailableAPITracker } from './types';

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
  constructor(metaKG: MetaKG, options?: BatchEdgeQueryOptions) {
    this.metaKG = metaKG;
    this.logs = [];
    this.caching = options && options.caching;
    this.options = options;
    if (options && options.recordHashEdgeAttributes) {
      this.options.EDGE_ATTRIBUTES_USED_IN_RECORD_HASH = options.recordHashEdgeAttributes;
    }
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
  _queryAPIEdges(APIEdges: APIEdge[], unavailableAPIs: UnavailableAPITracker = {}): Promise<Record[]> {
    // Skip queueing queries to unavailable APIs
    const queries = constructQueries(APIEdges, this.options).filter((query) => {
      if (unavailableAPIs[query.APIEdge.query_operation.server]?.skip === true) {
        unavailableAPIs[query.APIEdge.query_operation.server].skippedQueries += 1;
        return false;
      }
      return true;
    });

    const queriesByHash = Object.fromEntries(queries.map((query) => [query.hash, query]));

    const qEdge = APIEdges[0].reasoner_edge;
    const message = `${queries.length} planned queries for edge ${qEdge.id}`;
    debug(message);
    this.logs.push(new LogEntry('INFO', null, message).getLog());
    let finishedCount = 0;
    const completedLogs = this.logs;
    const completedRecords: Record[] = [];
    return new Promise<Record[]>((resolve) => {
      function listener(msg: ThreadMessage) {
        if (msg.type !== 'subQueryResult') return;
        const { hash, records, logs, apiUnavailable } = msg.value as {
          hash: string;
          records: RecordPackage;
          logs: SerializableLog[];
          apiUnavailable: boolean;
        };
        completedLogs.push(...LogEntry.deserialize(logs));
        completedRecords.push(...Record.unpackRecords(records, qEdge));

        // Update any APIs that were unavailable for this segment
        const server = queriesByHash[hash].APIEdge.query_operation.server;
        if (apiUnavailable) {
          if (!unavailableAPIs[server]) {
            unavailableAPIs[server] = { skip: true, skippedQueries: 0 };
          }
          unavailableAPIs[server].skippedQueries += 1;
        }

        finishedCount += 1;
        if (finishedCount >= queries.length) {
          debug(`Total number of records returned for qEdge ${qEdge.id} is ${completedRecords.length}`);
          resolve(completedRecords);
          global.workerSide.off('message', listener); // Clean up
        }
      }
      global.workerSide.on('message', listener);
      global.workerSide.postMessage({
        type: 'subqueryRequest',
        value: {
          queries: queries.map((query) => query.freeze()),
          options: this.options,
        },
      });
    });
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
            `stripped (${nodeStrippedCuries.length}) duplicate equivalent curies from ${
              node.id
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

    // const cacheHandler = new CacheHandler(this.caching, this.metaKG, this.options);
    // const { cachedRecords, nonCachedQEdges } = await cacheHandler.categorizeEdges(qEdges);
    // this.logs = [...this.logs, ...cacheHandler.logs];
    let queryRecords: Record[];

    debug('Start to convert qEdges into APIEdges....');
    const edgeConverter = new QEdge2APIEdgeHandler(qEdges, this.metaKG);
    const APIEdges = await edgeConverter.convert(qEdges);
    debug(`qEdges are successfully converted into ${APIEdges.length} APIEdges....`);
    this.logs = [...this.logs, ...edgeConverter.logs];
    if (APIEdges.length === 0) {
      return [];
    }
    const expanded_APIEdges = this._expandAPIEdges(APIEdges);
    debug('Start to query APIEdges....');
    queryRecords = await this._queryAPIEdges(expanded_APIEdges, unavailableAPIs);
    if (queryRecords === undefined) return;
    debug('APIEdges are successfully queried....');
    queryRecords = await this._postQueryFilter(queryRecords);
    debug(`Total number of records is (${queryRecords.length})`);
    debug('Start to update nodes...');
    nodeUpdate.update(queryRecords);
    debug('Update nodes completed!');
    return queryRecords;
  }
}
