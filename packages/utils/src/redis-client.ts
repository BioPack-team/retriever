import Redis, { Cluster, RedisKey } from "ioredis";
import Debug from "debug";
const debug = Debug("bte:biothings-explorer-trapi:redis-client");
import Redlock, { RedlockAbortSignal } from "redlock";

const prefix = `{BTEHashSlotPrefix}:`;

type AsyncFunction = (...args: unknown[]) => Promise<unknown>;
type DropFirst<T extends unknown[]> = T extends [unknown, ...infer U]
  ? U
  : never;
type Awaited<T> = T extends PromiseLike<infer U> ? U : T;

function timeoutFunc<F extends AsyncFunction>(func: F, timeoutms = 0) {
  return (...args: Parameters<F>): ReturnType<F> => {
    return new Promise((resolve, reject) => {
      const timeout = timeoutms
        ? timeoutms
        : parseInt(process.env.REDIS_TIMEOUT || "60000");
      let done = false;
      setTimeout(() => {
        if (!done) {
          reject(
            new Error(
              `redis call timed out, args: ${JSON.stringify([...args])}`,
            ),
          );
        }
      }, timeout);
      func(...args).then((returnValue: ReturnType<F>) => {
        done = true;
        resolve(returnValue);
      });
    }) as ReturnType<F>;
  };
}

/**
 * Decorate a function such that the first argument is given the module-defined prefix
 */
function addPrefix<F extends AsyncFunction>(func: F) {
  return (
    arg0: Parameters<F>[0],
    ...args: DropFirst<Parameters<F>>
  ): ReturnType<F> => {
    if (arg0 && (arg0 as string).length > 0) {
      arg0 = `${prefix}${arg0 as string}`;
    }
    return func(arg0, ...args) as ReturnType<F>;
  };
}

/**
 * Decorate a function such that each argument is given the module-defined prefix
 */
function addPrefixToAll<F extends AsyncFunction>(func: F) {
  return (...args: Parameters<F>): ReturnType<F> => {
    return func(...args.map(arg => `${prefix}${arg}`)) as ReturnType<F>;
  };
}

/**
 * Decorate a Redlock function such that the locks are given the module-defined prefix
 */
function lockPrefix<F extends AsyncFunction>(func: F) {
  return async (
    locks: Parameters<F>[0],
    ...args: DropFirst<Parameters<F>>
  ): Promise<Awaited<ReturnType<F>>> => {
    return (await func(
      (locks as string[]).map((lockName: string) => `${prefix}${lockName}`),
      ...args,
    )) as Awaited<ReturnType<F>>;
  };
}

interface RedisClientInterface {
  clearEdgeCache: () => void;
  getTimeout: (key: RedisKey) => Promise<string>;
  setTimeout: (key: RedisKey, value: string | number | Buffer) => Promise<"OK">;
  hsetTimeout: (
    ...args: [key: RedisKey, ...fieldValues: (string | Buffer | number)[]]
  ) => Promise<number>;
  hgetallTimeout: (key: RedisKey) => Promise<Record<string, string>>;
  expireTimeout: (key: RedisKey, seconds: string | number) => Promise<number>;
  delTimeout: (key: RedisKey | RedisKey[]) => Promise<number>;
  usingLock: (
    resources: string[],
    duration: number,
    routine?: (signal: RedlockAbortSignal) => Promise<unknown>,
  ) => Promise<unknown>;
  incrTimeout: (key: string) => Promise<number>;
  decrTimeout: (key: string) => Promise<number>;
  existsTimeout: (...args: RedisKey[]) => Promise<number>;
  pingTimeout: () => Promise<"PONG">;
}

function addClientFuncs(
  client: Redis | Cluster,
  redlock: Redlock,
): RedisClientInterface {
  function decorate<F extends AsyncFunction>(
    func: F,
    timeoutms?: number,
  ): (...args: Parameters<F>) => ReturnType<F> {
    let wrapped = timeoutFunc(func, timeoutms);
    if (client instanceof Cluster) {
      // Dirty way to cast the function so that typescript doesn't complain
      // But given the extremely limited use-case of this function, it's fine for now
      wrapped = addPrefix(wrapped) as unknown as (
        ...args: Parameters<F>
      ) => ReturnType<F>;
    }

    return wrapped;
  }
  return {
    clearEdgeCache: () => null,
    getTimeout: decorate((key: RedisKey) => client.get(key)),
    setTimeout: decorate((key: RedisKey, value: string | number | Buffer) =>
      client.set(key, value),
    ),
    hsetTimeout: decorate(
      (
        ...args: [key: RedisKey, ...fieldValues: (string | Buffer | number)[]]
      ) => client.hset(...args),
    ),
    hgetallTimeout: decorate((key: RedisKey) => client.hgetall(key)),

    expireTimeout: decorate((key: RedisKey, seconds: string | number) =>
      client.expire(key, seconds),
    ),

    delTimeout:
      client instanceof Cluster
        ? addPrefixToAll(
            timeoutFunc((...args: RedisKey[]) => client.del(...args)),
          )
        : timeoutFunc((...args: RedisKey[]) => client.del(...args)),
    usingLock: lockPrefix(
      (
        resources: string[],
        duration: number,
        routine?: (signal: RedlockAbortSignal) => Promise<unknown>,
      ) => redlock.using(resources, duration, routine),
    ),
    incrTimeout: decorate((key: string) => client.incr(key)),
    decrTimeout: decorate((key: string) => client.decr(key)),
    existsTimeout: decorate((...args: RedisKey[]) => client.exists(...args)),
    pingTimeout: decorate(() => client.ping(), 10000), // for testing
    // hmsetTimeout: decorate((...args) => client.hmset(...args)),
    // keysTimeout: decorate((...args) => client.keys(...args)),
  };
}

class RedisClient {
  client: Record<string, never> | ReturnType<typeof addClientFuncs>;
  enableRedis: boolean;
  clientEnabled: boolean;
  internalClient: Redis | Cluster;
  constructor() {
    this.client;
    this.enableRedis =
      !(process.env.REDIS_HOST === undefined) &&
      !(process.env.REDIS_PORT === undefined);

    if (!this.enableRedis) {
      this.client = {};
      this.clientEnabled = false;
      return;
    }

    interface RedisClusterDetails {
      redisOptions: {
        connectTimeout: number;
        password?: string;
        tls?: {
          checkServerIdentity: () => undefined | Error;
        };
      };
      // How long to wait given how many failed tries
      clusterRetryStrategy?: (times: number) => number;
    }

    if (process.env.REDIS_CLUSTER === "true") {
      const details = {
        redisOptions: {
          connectTimeout: 20000,
        },
        clusterRetryStrategy(times: number) {
          return Math.min(times * 100, 5000);
        },
      } as RedisClusterDetails;

      if (process.env.REDIS_PASSWORD) {
        details.redisOptions.password = process.env.REDIS_PASSWORD;
      }
      if (process.env.REDIS_TLS_ENABLED) {
        details.redisOptions.tls = { checkServerIdentity: () => undefined };
      }

      const cluster = new Redis.Cluster(
        [
          {
            host: process.env.REDIS_HOST,
            port: parseInt(process.env.REDIS_PORT),
          },
        ],
        details,
      );

      // allow up to 10 minutes to acquire lock (in case of large items being saved/retrieved)
      const redlock = new Redlock([cluster], {
        retryDelay: 500,
        retryCount: 1200,
      });

      this.internalClient = cluster;

      this.client = addClientFuncs(cluster, redlock);

      this.client.clearEdgeCache = () => {
        let count = 0;
        const nodes = (this.internalClient as Cluster).nodes("master");
        let completedNodes = 0;
        nodes.forEach((node, i) => {
          const stream = node.scanStream({
            match: "*bte:edgeCache:*",
            count: 50,
          });

          stream
            .on("data", (foundKeys: string[]) => {
              if (!foundKeys.length) return;
              count += foundKeys.length;
              try {
                node
                  .del(...foundKeys.map(key => key.replace(`${prefix} `, "")))
                  .then(
                    () => null,
                    error => {
                      debug(
                        `Cache clear: error deleting ${foundKeys.length} keys`,
                      );
                      debug(error);
                    },
                  );
              } catch (error) {
                debug("Cache clearing failure:");
                debug(error);
              }
            })
            .on("error", error => {
              debug(`Cache clearing failure on node ${i}:`);
              debug(error);
              completedNodes += 1;
              if (completedNodes >= nodes.length) {
                debug(`Cache clearing completes, cleared ${count} keys.`);
              }
            })
            .on("end", () => {
              debug(`Cache clearing completes on cluster node ${i}`);
              completedNodes += 1;
              if (completedNodes >= nodes.length) {
                debug(`Cache clearing completes, cleared ${count} keys.`);
              }
            });
        });
      };

      debug("Initialized redis client (cluster-mode)");
    } else {
      interface RedisDetails {
        host: string;
        port: number;
        connectTimeout: number;
        retryStrategy: (times: number) => number;
        password?: string;
        tls?: {
          checkServerIdentity: () => undefined | Error;
        };
      }

      const details = {
        host: process.env.REDIS_HOST,
        port: parseInt(process.env.REDIS_PORT),
        connectTimeout: 20000,
        retryStrategy(times) {
          return Math.min(times * 100, 5000);
        },
      } as RedisDetails;
      if (process.env.REDIS_PASSWORD) {
        details.password = process.env.REDIS_PASSWORD;
      }
      if (process.env.REDIS_TLS_ENABLED) {
        details.tls = { checkServerIdentity: () => undefined };
      }
      const client = new Redis(details);

      // allow up to 10 minutes to acquire lock (in case of large items being saved/retrieved)
      const redlock = new Redlock([client], {
        retryDelay: 500,
        retryCount: 1200,
      });

      this.internalClient = client;

      this.client = addClientFuncs(client, redlock);

      this.client.clearEdgeCache = () => {
        const stream = (redisClient.internalClient as Redis).scanStream({
          match: "*bte:edgeCache:*",
          count: 50,
        });

        let count = 0;

        stream
          .on("data", (foundKeys: string[]) => {
            if (!foundKeys.length) return;
            count += foundKeys.length;
            try {
              redisClient.internalClient
                .del(...foundKeys.map(key => key.replace(`${prefix} `, "")))
                .then(
                  () => null,
                  error => {
                    debug(
                      `Cache clear: error deleting ${foundKeys.length} keys`,
                    );
                    debug(error);
                  },
                );
            } catch (error) {
              debug("Cache clearing failure:");
              debug(error);
            }
          })
          .on("error", error => {
            debug("Cache clearing failure:");
            debug(error);
          })
          .on("end", () => {
            debug(`Cache clearing completes, cleared ${count} keys.`);
          });
      };

      debug("Initialized redis client (non-cluster-mode)");
    }
    this.clientEnabled = true;
  }
}

const redisClient = new RedisClient();

function getNewRedisClient(): RedisClient {
  return new RedisClient();
}

export { redisClient, getNewRedisClient, RedisClient };
