import { redisClient } from '@biothings-explorer/utils';
import Debug from 'debug';
const debug = Debug('bte:biothings-explorer-trapi:cache_handler');
import { LogEntry, StampedLog } from '@biothings-explorer/utils';
import async from 'async';
import helper from './helper';
import lz4 from 'lz4';
import chunker from 'stream-chunker';
import { Readable, Transform } from 'stream';
import { Record, RecordPackage } from '@biothings-explorer/api-response-transform';
import { threadId } from 'worker_threads';
import MetaKG from '../../smartapi-kg/built';
import QEdge from './query_edge';
import { QueryHandlerOptions } from '@biothings-explorer/types';

export interface RecordPacksByQedgeMetaKGHash {
  [QEdgeHash: string]: RecordPackage;
}

class DelimitedChunksDecoder extends Transform {
  private _buffer: string;
  constructor() {
    super({
      readableObjectMode: true,
      readableHighWaterMark: 32, // limited output reduces RAM usage slightly
      writableHighWaterMark: 100000,
    });
    this._buffer = '';
  }

  _transform(chunk: string, encoding: string, callback: () => void): void {
    this._buffer += chunk;
    if (this._buffer.includes(',')) {
      const parts = this._buffer.split(',');
      this._buffer = parts.pop();
      parts.forEach((part) => {
        const parsedPart = JSON.parse(lz4.decode(Buffer.from(part, 'base64url')).toString());
        if (Array.isArray(parsedPart)) {
          parsedPart.forEach((obj) => this.push(obj));
        } else {
          // backwards compatibility with previous implementation
          this.push(parsedPart);
        }
      });
    }
    callback(); // callback *no matter what*
  }

  _flush(callback: (error?: Error | null | undefined, data?: unknown) => void): void {
    try {
      if (this._buffer.length) {
        const final = JSON.parse(lz4.decode(Buffer.from(this._buffer, 'base64url')).toString());
        callback(null, final);
      }
      callback();
    } catch (error) {
      callback(error);
    }
  }
}

class DelimitedChunksEncoder extends Transform {
  private _buffer: unknown[];
  constructor() {
    super({
      writableObjectMode: true,
      writableHighWaterMark: 128,
    });
    this._buffer = [];
  }

  _transform(obj: unknown, encoding: unknown, callback: () => void) {
    this._buffer.push(obj); // stringify/compress 64 objects at a time limits compress calls
    if (this._buffer.length === 64) {
      const compressedPart = lz4.encode(JSON.stringify(this._buffer)).toString('base64url') + ',';
      this.push(compressedPart);
      this._buffer = [];
    }
    callback();
  }

  _flush(callback: (error?: Error | null | undefined, data?: unknown) => void) {
    try {
      if (this._buffer.length) {
        callback(null, lz4.encode(JSON.stringify(this._buffer)).toString('base64url') + ',');
        return;
      }
      callback();
    } catch (error) {
      callback(error);
    }
  }
}

export default class CacheHandler {
  metaKG: MetaKG;
  logs: StampedLog[];
  cacheEnabled: boolean;
  recordConfig: QueryHandlerOptions;
  constructor(caching: boolean, metaKG = undefined, recordConfig = {}, logs = []) {
    this.metaKG = metaKG;
    this.logs = logs;
    this.cacheEnabled =
      caching === false
        ? false
        : process.env.RESULT_CACHING !== 'false'
          ? !(process.env.REDIS_HOST === undefined) && !(process.env.REDIS_PORT === undefined)
          : false;
    this.recordConfig = recordConfig;
    this.logs.push(
      new LogEntry('DEBUG', null, `REDIS cache is ${this.cacheEnabled === true ? '' : 'not'} enabled.`).getLog(),
    );
  }

  async categorizeEdges(qEdges: QEdge[]): Promise<{ cachedRecords: Record[]; nonCachedQEdges: QEdge[] }> {
    if (this.cacheEnabled === false || process.env.INTERNAL_DISABLE_REDIS === "true") {
      return {
        cachedRecords: [],
        nonCachedQEdges: qEdges,
      };
    }
    const nonCachedQEdges: QEdge[] = [];
    let cachedRecords: Record[] = [];
    debug('Begin edge cache lookup...');
    await async.eachSeries(qEdges, async (qEdge) => {
      const qEdgeMetaKGHash = this._hashEdgeByMetaKG(qEdge.getHashedEdgeRepresentation());
      const unpackedRecords: Record[] = await new Promise((resolve) => {
        const redisID = 'bte:edgeCache:' + qEdgeMetaKGHash;
        redisClient.client.usingLock([`redisLock:${redisID}`], 600000, async () => {
          try {
            const compressedRecordPack = await redisClient.client.hgetallTimeout(redisID);

            if (compressedRecordPack && Object.keys(compressedRecordPack).length) {
              const recordPack = [];

              const sortedPackParts = Object.entries(compressedRecordPack)
                .sort(([key1], [key2]) => parseInt(key1) - parseInt(key2))
                .map(([, val]) => {
                  return val;
                });

              const recordStream = Readable.from(sortedPackParts);
              recordStream
                .pipe(this.createDecodeStream())
                .on('data', (obj) => recordPack.push(obj))
                .on('end', () => resolve(Record.unpackRecords(recordPack as RecordPackage, qEdge, this.recordConfig)));
            } else {
              resolve(null);
            }
          } catch (error) {
            resolve(null);
            debug(`Cache lookup/retrieval failed due to ${error}. Proceeding without cache.`);
          }
        });
      });

      if (unpackedRecords) {
        this.logs.push(
          new LogEntry('DEBUG', null, `BTE finds cached records for ${qEdge.getID()}`, {
            type: 'cacheHit',
            qEdgeID: qEdge.getID(),
            api_names: unpackedRecords.map((record) => record.association?.api_name),
          }).getLog(),
        );
        cachedRecords = [...cachedRecords, ...unpackedRecords];
      } else {
        nonCachedQEdges.push(qEdge);
      }
      debug(`Found (${cachedRecords.length}) cached records.`);
    });

    return { cachedRecords, nonCachedQEdges };
  }

  _hashEdgeByMetaKG(qEdgeHash: string): string {
    if (!this.metaKG) {
      return qEdgeHash;
    }
    const len = String(this.metaKG.ops.length);
    const allIDs = Array.from(new Set(this.metaKG.ops.map((op) => op.association.smartapi.id))).join('');
    return helper._generateHash(qEdgeHash + len + allIDs);
  }

  _groupQueryRecordsByQEdgeHash(queryRecords: Record[]): RecordPacksByQedgeMetaKGHash {
    const groupedRecords: { [qEdgeMetaKGHash: string]: Record[] } = {};
    queryRecords.map((record) => {
      try {
        const qEdgeMetaKGHash = this._hashEdgeByMetaKG(record.qEdge.getHashedEdgeRepresentation());
        if (!(qEdgeMetaKGHash in groupedRecords)) {
          groupedRecords[qEdgeMetaKGHash] = [];
        }
        groupedRecords[qEdgeMetaKGHash].push(record);
      } catch (e) {
        debug('skipping malformed record');
      }
    });
    return Object.fromEntries(
      Object.entries(groupedRecords).map(([qEdgeMetaKGHash, records]) => {
        return [qEdgeMetaKGHash, Record.packRecords(records)];
      }),
    );
  }

  createEncodeStream(): DelimitedChunksEncoder {
    return new DelimitedChunksEncoder();
  }

  createDecodeStream(): DelimitedChunksDecoder {
    return new DelimitedChunksDecoder();
  }

  async cacheEdges(queryRecords: Record[]): Promise<void> {
    if (this.cacheEnabled === false || process.env.INTERNAL_DISABLE_REDIS === "true") {
      if (global.parentPort) {
        global.parentPort.postMessage({ threadId, type: "cacheDone", value: true });
      }
      return;
    }
    if (global.parentPort) {
      global.parentPort.postMessage({ threadId, type: "cacheInProgress", value: 1 });
    }
    debug('Start to cache query records.');
    try {
      const groupedRecords = this._groupQueryRecordsByQEdgeHash(queryRecords);
      const qEdgeHashes = Array.from(Object.keys(groupedRecords));
      debug(`Number of hashed edges: ${qEdgeHashes.length}`);
      const failedHashes = [];
      await async.eachSeries(qEdgeHashes, async (hash) => {
        // lock to prevent caching to/reading from actively caching edge
        const redisID = 'bte:edgeCache:' + hash;
        if (global.parentPort) {
          global.parentPort.postMessage({ threadId, type: "addCacheKey", value: redisID });
        }
        await redisClient.client.usingLock([`redisLock:${redisID}`, 'redisLock:EdgeCaching'], 600000, async () => {
          try {
            await redisClient.client.delTimeout(redisID); // prevents weird overwrite edge cases
            await new Promise<void>((resolve, reject) => {
              let i = 0;
              Readable.from(groupedRecords[hash])
                .pipe(this.createEncodeStream())
                .pipe(chunker(100000, { flush: true }))
                .on('data', async (chunk: string) => {
                  try {
                    await redisClient.client.hsetTimeout(redisID, String(i++), chunk);
                  } catch (error) {
                    reject(error);
                    try {
                      await redisClient.client.delTimeout(redisID);
                    } catch (e) {
                      debug(
                        `Unable to remove partial cache ${redisID} from redis during cache failure due to error ${error}. This may result in failed or improper cache retrieval of this qEdge.`,
                      );
                    }
                  }
                })
                .on('end', () => {
                  resolve();
                });
            });
            if (process.env.QEDGE_CACHE_TIME_S !== "0") {
              await redisClient.client.expireTimeout(redisID, process.env.QEDGE_CACHE_TIME_S || 1800);
            }
          } catch (error) {
            failedHashes.push(hash);
            debug(
              `Failed to cache qEdge ${hash} records due to error ${error}. This does not stop other edges from caching nor terminate the query.`,
            );
          } finally {
            if (global.parentPort) {
              global.parentPort.postMessage({ threadId, type: "completeCacheKey", value: redisID });
            }
          }
        });
      });
      const successCount = Object.entries(groupedRecords).reduce((acc, [hash, records]) => {
        return failedHashes.includes(hash) ? acc : acc + records.length;
      }, 0);
      if (successCount) {
        debug(`Successfully cached (${successCount}) query records.`);
      } else {
        debug(`qEdge caching failed.`);
      }
    } catch (error) {
      debug(`Caching failed due to ${error}. This does not terminate the query.`);
    } finally {
      if (global.parentPort) {
        global.parentPort.postMessage({ threadId, type: "cacheDone", value: 1 });
      }
    }
  }
}
