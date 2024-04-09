import "@biothings-explorer/types";
import Queue, { Queue as BullQueue, FailedEventCallback } from "bull";
import axios from "axios";
import { redisClient, getNewRedisClient } from "@biothings-explorer/utils";
import Debug from "debug";
import { TrapiLog, TrapiQueryGraph } from "@biothings-explorer/types";
import { BullJob } from "../../types";
const debug = Debug("bte:biothings-explorer-trapi:asyncquery_queue");
import { Redis } from "ioredis";
import { QueryQueue } from "packages/types/src/global";

global.queryQueue = {} as QueryQueue;

export function getQueryQueue(name: string): BullQueue {
  const queryQueue: BullQueue = null;
  if (!redisClient.clientEnabled || process.env.INTERNAL_DISABLE_REDIS === "true") {
    return queryQueue;
  }
  debug(
    `Getting queue ${name} using redis in ${process.env.REDIS_CLUSTER === "true" ? "cluster" : "non-cluster"} mode`,
  );
  if (global.queryQueue[name]) {
    return global.queryQueue[name];
  }
  debug(`Initializing queue ${name} for first time...`);
  const details = {
    createClient: undefined,
    prefix: `{BTE:bull}`,
    defaultJobOptions: {
      removeOnFail: {
        age: 24 * 60 * 60, // keep failed jobs for a day (in case user needs to review fail reason)
        count: 2000,
      },
      removeOnComplete: {
        age: 90 * 24 * 60 * 60, // keep completed jobs for 90 days
        count: 2000,
      },
    },
    settings: {
      maxStalledCount: 1,
      //lockDuration: 300000
      lockDuration: 3600000, // 60min
    },
  };

  if (process.env.REDIS_HOST) {
    details.createClient = () => {
      const client = getNewRedisClient();
      client.internalClient.options.enableReadyCheck = false;
      if (!client.internalClient.isCluster) {
        (client.internalClient as Redis).options.maxRetriesPerRequest = null;
      }
      return client.internalClient;
    };
  }

  global.queryQueue[name] = new Queue(name, process.env.REDIS_HOST ?? "redis://127.0.0.1:6379", details)
    .on("error", function(error) {
      console.log("err", error);
    })
    .on("failed", async function(job: BullJob, error) {
      debug(`Async job ${job.id} failed with error ${error.message}`);
      try {
        job.data.abortController.abort();
      } catch (error) {
        debug(error);
      }
      if (job.data.callback_url) {
        const logs: TrapiLog[] = await global.queryQueue[name]?.getJobLogs(job.id)?.logs?.map((log: string) => JSON.parse(log));
        try {
          await axios({
            method: "post",
            url: job.data.callback_url,
            data: {
              schema_version: global.SCHEMA_VERSION,
              biolink_version: global.BIOLINK_VERSION,
              workflow: [{ id: "lookup" }],
              logs: logs,
              message: {
                query_graph: job.data.queryGraph,
                knowledge_graph: { nodes: {}, edges: {} },
                results: [],
              },
              status: "Failed",
              description: error.toString(),
              trace: process.env.NODE_ENV === "production" ? undefined : error.stack,
            },
          });
        } catch (error) {
          debug(`Callback failed with error ${(error as Error).message}`);
        }
      }
    } as FailedEventCallback);

  return global.queryQueue[name];
}
