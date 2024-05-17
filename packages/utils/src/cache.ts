import { Readable, Transform } from "stream";
import lz4 from "lz4";
import { redisClient } from "./redis-client";
import Debug from "debug";
const debug = Debug("retriever:caching");

export async function cacheContent(
  hash: string,
  content: unknown[],
): Promise<void> {
  if (!redisClient.clientEnabled) {
    debug("redis client unavailable, skipping cache step");
    return;
  }

  debug(`Caching ${content.length} items for article ${hash}`);
  const cacheID = `retriever:cacheContent:${hash}`;
  let success = false;

  await redisClient.client.usingLock(
    [`retriever:cachingLock:${hash}`],
    30000,
    async () => {
      try {
        await redisClient.client.delTimeout(cacheID); // delete previous cache to avoid edge cases
        await new Promise<void>((resolve, reject) =>
          streamToCache(hash, content, resolve, reject),
        );
        success = true;
      } catch (error) {
        debug(`Failed to cache article ${hash} due to error ${error}.`);
      }
    },
  );

  if (!success) return;

  if (process.env.QEDGE_CACHE_TIME_S !== "0") {
    try {
      await redisClient.client.expireTimeout(
        cacheID,
        process.env.QEDGE_CACHE_TIME_S || 1800,
      );
    } catch (error) {
      debug("Failed to set cache timeout for article ${hash} due to ${error}");
    }
  }

  debug(`Successfully cached ${content.length} items for article ${hash}`);
}

function streamToCache(
  hash: string,
  content: unknown[],
  resolve: () => void,
  reject: (reason: any) => void,
) {
  let index = 0;
  Readable.from(content)
    .pipe(new DelimitedChunksEncoder())
    .on(
      "data",
      async (chunk: string) => await cacheChunk(hash, index++, chunk, reject),
    )
    .on("end", () => resolve());
}
async function cacheChunk(
  hash: string,
  index: number,
  chunk: string,
  reject: { (reason: any): void; (arg0: any): void },
) {
  const id = `retriever:cacheContent:${hash}`;
  try {
    await redisClient.client.hsetTimeout(id, String(index++), chunk);
  } catch (error) {
    reject(error);
    // Attempt cleanup after reporting failure
    try {
      await redisClient.client.delTimeout(id);
    } catch (e) {
      debug(
        `Unable to remove partial cache article ${id} from redis during cache failure due to error ${error}. This may result in failed or improper cache retrieval of this article.`,
      );
    }
  }
}

export async function cacheLookup(hash: string): Promise<unknown[] | null> {
  if (!redisClient.clientEnabled) {
    debug("redis client unavailable, skipping cache step");
    return null;
  }
  debug(`Beginning cache lookup for article ${hash}`);
  const id = `retriever:cacheContent:${hash}`;

  const content = await new Promise<any>(resolve => {
    redisClient.client.usingLock(
      [`retriever: cachingLock:${hash} `],
      30000,
      async () => readFromCache(hash, resolve),
    );
  });

  if (!content) {
    debug(`No cached content found for article ${hash}`);
    return content;
  }

  const message = `Found ${content.length} cached content for article ${hash}.`;
  debug(message);
  try {
    await redisClient.client.expireTimeout(
      id,
      process.env.QEDGE_CACHE_TIME_S || 1800,
    );
  } catch (error) {
    debug(
      "Failed to re-boost cache timeout for subquery ${hash} due to ${error}",
    );
  }

  return content;
}

async function readFromCache(
  hash: string,
  resolve: (value: any | null) => void,
): Promise<void> {
  const id = `retriever:cacheContent:${hash}`;
  try {
    const compressedContent = await redisClient.client.hgetallTimeout(id);
    if (!(compressedContent && Object.keys(compressedContent).length)) {
      resolve(null);
    }

    const content = [];

    const sortedPackParts = Object.entries(compressedContent)
      .sort(([key1], [key2]) => parseInt(key1) - parseInt(key2))
      .map(([, val]) => val);

    const recordStream = Readable.from(sortedPackParts);
    recordStream
      .pipe(new DelimitedChunksDecoder())
      .on("data", obj => content.push(obj))
      .on("end", () => resolve(content));
  } catch (error) {
    debug(
      `Cache lookup / retrieval for subquery ${hash} failed due to ${error}.`,
    );
    resolve(null);
  }
}

export class DelimitedChunksEncoder extends Transform {
  private _buffer: unknown[];
  constructor() {
    super({
      writableObjectMode: true,
      writableHighWaterMark: 128,
    });
    this._buffer = [];
  }

  _transform(obj: unknown, _encoding: unknown, callback: () => void) {
    this._buffer.push(obj); // stringify/compress 64 objects at a time limits compress calls
    if (this._buffer.length === 64) {
      const compressedPart =
        lz4
          .encode(Buffer.from(JSON.stringify(this._buffer)))
          .toString("base64url") + ",";
      this.push(compressedPart);
      this._buffer = [];
    }
    callback();
  }

  _flush(callback: (error?: Error | null | undefined, data?: unknown) => void) {
    try {
      if (!this._buffer.length) {
        callback();
        return;
      }
      callback(
        null,
        lz4
          .encode(Buffer.from(JSON.stringify(this._buffer)))
          .toString("base64url") + ",",
      );
    } catch (error) {
      callback(error);
    }
  }
}

export class DelimitedChunksDecoder extends Transform {
  private _buffer: string;
  constructor() {
    super({
      readableObjectMode: true,
      readableHighWaterMark: 32, // limited output reduces RAM usage slightly
      writableHighWaterMark: 100000,
    });
    this._buffer = "";
  }

  _transform(chunk: string, _encoding: string, callback: () => void): void {
    this._buffer += chunk;
    if (this._buffer.includes(",")) {
      const parts = this._buffer.split(",");
      this._buffer = parts.pop();
      parts.forEach(part => {
        const parsedPart = JSON.parse(
          lz4.decode(Buffer.from(part, "base64url")).toString(),
        );
        if (Array.isArray(parsedPart)) {
          parsedPart.forEach(obj => this.push(obj));
        } else {
          // backwards compatibility with previous implementation
          this.push(parsedPart);
        }
      });
    }
    callback(); // callback *no matter what*
  }

  _flush(
    callback: (error?: Error | null | undefined, data?: unknown) => void,
  ): void {
    try {
      if (!this._buffer.length) {
        callback();
        return;
      }
      const final = JSON.parse(
        lz4.decode(Buffer.from(this._buffer, "base64url")).toString(),
      );
      callback(null, final);
    } catch (error) {
      callback(error);
    }
  }
}
