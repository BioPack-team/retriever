import { MessageChannel, threadId } from "worker_threads";
import Debug from "debug";
import { context, propagation } from "@opentelemetry/api";
const debug = Debug("bte:biothings-explorer-trapi:threading");
import path from "path";
import { redisClient } from "@biothings-explorer/utils";
import Piscina from "piscina";
import os from "os";
import ServerOverloadedError from "../../utils/errors/server_overloaded_error";
import { customAlphabet } from "nanoid";
import { getQueryQueue } from "../async/asyncquery_queue";
import { tasks } from "../../routes/index";

import { Telemetry } from "@biothings-explorer/utils";
import ErrorHandler from "../../middlewares/error";
import { Request, Response } from "express";
import { BullJob, PiscinaWaitTime, ThreadPool } from "../../types";
import { TaskInfo, InnerTaskData } from "@biothings-explorer/types";
import { DialHome, TrapiQuery, TrapiResponse } from "@biothings-explorer/types";
import { Queue } from "bull";

const SYNC_MIN_CONCURRENCY = 2;
const ASYNC_MIN_CONCURRENCY = 3;

// On most instances, there are two nodes, one for Service Provider endpoints and one for everything else
// On Dev and local instances, this isn't the case, so a lower concurrency is needed
const CORE_CONCURRENCY_RATIO = parseInt(process.env.CORE_CONCURRENCY_RATIO) || 0.25;
const MEM_CONCURRENCY_RATIO = parseFloat(process.env.MEM_CONCURRENCY_RATIO) || 0.6;

const CORE_LIMIT = Math.ceil(os.cpus().length * CORE_CONCURRENCY_RATIO);

const MEM_LIMIT = Math.ceil((os.totalmem() / 1e9) * MEM_CONCURRENCY_RATIO);

// Ex. Prod: 16 cores / 64GB mem = min(16 * 2, 32) = 32 allowed concurrently
// Divided by 4 because each instance uses 4 sub-instances for reliability
let SYNC_CONCURRENCY = Math.ceil(Math.min(CORE_LIMIT, MEM_LIMIT) / 4);
if (SYNC_CONCURRENCY < SYNC_MIN_CONCURRENCY) SYNC_CONCURRENCY = SYNC_MIN_CONCURRENCY;

let ASYNC_CONCURRENCY = SYNC_CONCURRENCY;
if (ASYNC_CONCURRENCY < ASYNC_MIN_CONCURRENCY) ASYNC_CONCURRENCY = ASYNC_MIN_CONCURRENCY;

// Async has 3 separate queues, concurrency is distributed between them as such:
const ASYNC_MAIN_CONCURRENCY = ASYNC_CONCURRENCY;
const ASYNC_BY_API_CONCURRENCY = Math.ceil(ASYNC_CONCURRENCY / 2);
const ASYNC_BY_TEAM_CONCURRENCY = Math.ceil(ASYNC_CONCURRENCY / 2);

if (!global.threadpool && !Piscina.isWorkerThread && !(process.env.USE_THREADING === "false")) {
  // Give user a little report of resource availability
  debug(`Computed core limit: ${CORE_LIMIT}`);
  debug(`Computed mem limit: ${MEM_LIMIT}`);
  debug(`Sync concurrency limit: ${SYNC_CONCURRENCY}`);
  debug(`Async concurrency limit: ${ASYNC_CONCURRENCY}`);
  const env = {
    ...process.env,
    DEBUG_COLORS: "true",
  };
  global.threadpool = {
    /**Medium-volume, medium-intensity requests
     * Maximum threads equal to 1/4 CPUs as BTE usually runs 4 intances
     * Maximum queue set, so if the queue if full, requests will be dropped
     * Usually, queuing will be handled by Bull, but if Bull is unavailable
     * (e.g. local debugging w/o a Redis instance), queuing falls back to Piscina
     */
    sync: new Piscina({
      filename: path.resolve(__dirname, "./taskHandler.js"),
      minThreads: 2,
      maxThreads: SYNC_CONCURRENCY,
      maxQueue: 1200,
      idleTimeout: 10 * 60 * 1000, // 10 minutes
      workerData: { queue: "sync" },
      env,
    }),
    /**Low-volume, high-intensity requests
     * No other settings since this queue is handled externally with Bull
     * High timeout because we expect near-constant use, so rather not kill the thread
     */
    async: new Piscina({
      filename: path.resolve(__dirname, "./taskHandler.js"),
      // e.g. 8 cores *
      maxThreads: ASYNC_CONCURRENCY,
      minThreads: 1,
      idleTimeout: 60 * 60 * 1000, // 1 hour
      maxQueue: 1800,
      workerData: { queue: "async" },
      env,
    }),
    /**High-volume, low-intensity requests
     * Expecting frequent use, so timeout is high
     */
    misc: new Piscina({
      filename: path.resolve(__dirname, "./taskHandler.js"),
      idleTimeout: 60 * 60 * 1000, // 1 hour
      minThreads: 2,
      maxQueue: 1800,
      workerData: { queue: "misc" },
      env,
    }),
  } as ThreadPool;
}

async function queueTaskToWorkers(pool: Piscina, taskInfo: TaskInfo, route: string, job?: BullJob): Promise<DialHome> {
  return new Promise((resolve, reject) => {
    let workerThreadID: string;
    const abortController = new AbortController();
    const { port1: toWorker, port2: fromWorker } = new MessageChannel();

    // get otel context

    const otelData: Partial<{ traceparent: string; tracestate: string }> = {};
    propagation.inject(context.active(), otelData);
    const { traceparent, tracestate } = otelData;

    const taskData: InnerTaskData = { req: taskInfo, route, traceparent, tracestate, port: toWorker };

    // Propagate data between task runner and bull job
    if (job) taskData.job = { jobId: job.id, queueName: job.queue.name };
    const task = pool.run(taskData, { signal: abortController.signal, transferList: [toWorker] });
    if (job) {
      void job.update({ ...job.data, abortController });
    }

    // catch failures that cause the worker to outright fail
    task.catch((error: Error) => {
      if (error.name === "AbortError") {
        // Task was aborted externally (probably because it timed out)
        debug(`Worker thread ${workerThreadID} terminated successfully.`);
      } else if (error.message === "Task queue is at limit") {
        // Task has to be dropped due to traffic
        debug(
          [
            pool === (global.threadpool as ThreadPool).sync ? "Synchronous" : "Misc",
            "server queue is at limit, job rejected.",
          ].join(" "),
        );
        const expectedWaitTime = Math.ceil((pool.waitTime as PiscinaWaitTime).p99 / 1000) + 1;
        const message = [
          "The server is currently under heavy load.",
          ` Please try again after about ${expectedWaitTime}s`,
          pool === (global.threadpool as ThreadPool).sync ? ", or try using the asynchronous endpoints." : ".",
        ].join("");
        error = new ServerOverloadedError(message, expectedWaitTime);
      } else {
        debug(`Caught error in worker thread ${workerThreadID}, error below:`);
        debug(error);
      }
      reject(error);
    });

    let reqDone = false;
    let cacheInProgress = 0;
    const cacheKeys: {
      [cacheKey: string]: boolean;
    } = {};
    const timeout = parseInt(process.env.REQUEST_TIMEOUT ?? (60 * 5).toString()) * 1000;

    fromWorker.on("message", (msg: DialHome) => {
      if (msg.cacheInProgress) {
        // Cache handler has started caching
        cacheInProgress += 1;
      } else if (msg.addCacheKey) {
        // Hashed edge id cache in progress
        cacheKeys[msg.addCacheKey] = false;
      } else if (msg.completeCacheKey) {
        // Hashed edge id cache complete
        cacheKeys[msg.completeCacheKey] = true;
      } else if (msg.registerId) {
        // Worker registers itself for better tracking
        workerThreadID = String(msg.threadId);
        if (job) {
          void job.update({ ...job.data, threadId });
        }
      } else if (typeof msg.cacheDone !== "undefined") {
        cacheInProgress = msg.cacheDone
          ? cacheInProgress - 1 // A caching handler has finished caching
          : 0; // Caching has been entirely cancelled
      } else if (typeof msg.result !== "undefined") {
        // Request has finished with a message
        reqDone = true;
        resolve(msg);
      } else if (msg.err) {
        // Request has resulted in a catchable error
        reqDone = true;
        reject(msg.err);
      }
      if (reqDone && cacheInProgress <= 0 && job) {
        void job.progress(100);
      }
    });

    // Handling for timeouts -- we can kill a thread in progress to free resources
    // TODO better timeout handling for async?
    if (timeout && pool !== global.threadpool.async) {
      setTimeout(() => {
        // Clean up any incompletely cached hashes to avoid issues pulling from cache
        const activeKeys = Object.entries(cacheKeys)
          .filter(([, complete]) => !complete)
          .map(([key]) => key);
        if (activeKeys.length) {
          try {
            void redisClient.client.delTimeout(activeKeys);
          } catch (error) {
            null;
          }
        }
        abortController.abort();
        reject(
          new Error(
            [
              `Request timed out (exceeded time limit of ${timeout / 1000}s).`,
              "Please use the asynchronous endpoint (/v1/asyncquery) for long-running queries.",
            ].join(" "),
          ),
        );
      }, timeout);
    }
  });
}

export async function runTask(req: Request, res: Response, route: string, useBullSync = true): Promise<TrapiResponse> {
  const queryQueue: Queue = global.queryQueue.bte_sync_query_queue;
  const taskInfo: TaskInfo = {
    data: {
      route,
      queryGraph: (req.body as TrapiQuery)?.message?.query_graph,
      workflow: (req.body as TrapiQuery)?.workflow,
      options: {
        logLevel: (req.body as TrapiQuery).log_level || (req.query.log_level as string),
        submitter: (req.body as TrapiQuery).submitter,
        smartAPIID: req.params.smartapi_id,
        teamName: req.params.team_name,
        ...req.query,
      },
      params: req.params,
      endpoint: req.originalUrl,
    },
  };

  if (process.env.USE_THREADING === "false") {
    // Threading disabled, just use the provided function in main event loop
    const response = await tasks[route](taskInfo) as TrapiResponse;
    return response;
  } else if (!(queryQueue && useBullSync)) {
    // Redis unavailable or query not to sync queue such as asyncquery_status
    const response = await queueTaskToWorkers(
      useBullSync ? global.threadpool.sync : global.threadpool.misc,
      taskInfo,
      route,
    );

    if (typeof response.result !== "undefined") {
      if (response.status) {
        res?.status(response.status as number);
      }
      return response.result ? response.result : undefined; // null msg means keep response body empty
    } else if (response.err) {
      throw response.err;
    } else {
      throw new Error("Threading Error: Task resolved without message");
    }
  }

  // Otherwise, proceed as normal (Run in a Bull queue)
  const nanoid = customAlphabet("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", 10);
  const jobId = nanoid();

  const jobOpts = {
    jobId,
    attempts: 1,
    timeout: undefined,
    removeOnFail: {
      age: 24 * 60 * 60, // keep failed jobs for a day (in case user needs to review fail reason)
      count: 2000,
    },
    removeOnComplete: {
      age: 90 * 24 * 60 * 60, // keep completed jobs for 90 days
      count: 2000,
    },
  };

  // Check if queue is already at max and new req should be dropped
  const pool = global.threadpool.sync;
  if ((await queryQueue.count()) >= pool.options.maxQueue) {
    const expectedWaitTime = Math.ceil((pool.waitTime as PiscinaWaitTime).p99 / 1000) + 1;
    const message = [
      "The server is currently under heavy load.",
      ` Please try again after about ${expectedWaitTime}s`,
      ", or try using the asynchronous endpoints.",
    ].join("");
    throw new ServerOverloadedError(message, expectedWaitTime);
  }

  const job = await queryQueue.add(taskInfo.data, jobOpts);
  try {
    const response: TrapiResponse = await (job.finished() as Promise<TrapiResponse>);
    return response;
  } catch (error) {
    // Have to reconstruct the error because Bull does some weirdness
    const jobLatest = await queryQueue.getJob(jobOpts.jobId);
    const reconstructedError = new Error();
    try {
      reconstructedError.name = jobLatest.stacktrace[0].split(":")[0];
      reconstructedError.message = jobLatest.stacktrace[0].split("\n")[0];
      reconstructedError.stack = jobLatest.stacktrace[0];
    } catch (constructionError) {
      reconstructedError.name = "ThreadingError";
      reconstructedError.message = JSON.stringify(jobLatest.stacktrace);
    }
    throw reconstructedError;
  }
}

export async function runBullJob(job: BullJob, route: string, useAsync = true) {
  const taskInfo: TaskInfo = {
    id: String(job.id),
    data: { ...job.data },
  };
  const response = await queueTaskToWorkers(
    useAsync ? global.threadpool.async : global.threadpool.sync,
    taskInfo,
    route,
    job,
  );
  if (typeof response.result !== "undefined") {
    return response.result ? response.result : undefined; // null result means keep response body empty
  } else if (response.err) {
    throw response.err;
  } else {
    throw new Error("Threading Error: Task resolved without message");
  }
}

export function taskResponse<T>(response: T, status: string | number = undefined): T {
  if (global.parentPort) {
    global.parentPort.postMessage({ threadId, result: response, status: status });
    return undefined;
  } else {
    return response;
  }
}

export function taskError(error: Error): void {
  if (global.parentPort) {
    if (ErrorHandler.shouldHandleError(error)) {
      Telemetry.captureException(error);
    }
    global.parentPort.postMessage({ threadId, err: error });
    return undefined;
  } else {
    throw error;
  }
}

// TODO: use this usage of runBullTask to figure out runBullTask/queueTaskToWorkers' optional argument
if (!global.queryQueue.bte_sync_query_queue && !Piscina.isWorkerThread) {
  getQueryQueue("bte_sync_query_queue");
  if (global.queryQueue.bte_sync_query_queue) {
    global.queryQueue.bte_sync_query_queue.process(SYNC_CONCURRENCY, async (job: BullJob) => {
      return await runBullJob(job, job.data.route, false);
    });
  }
}

// TODO merge async into one queue
if (!global.queryQueue.bte_query_queue && !Piscina.isWorkerThread) {
  getQueryQueue("bte_query_queue");
  if (global.queryQueue.bte_query_queue) {
    global.queryQueue.bte_query_queue.process(ASYNC_MAIN_CONCURRENCY, async (job: BullJob) => {
      return await runBullJob(job, "asyncquery_v1");
    });
  }
}

if (!global.queryQueue.bte_query_queue_by_api && !Piscina.isWorkerThread) {
  getQueryQueue("bte_query_queue_by_api");
  if (global.queryQueue.bte_query_queue_by_api) {
    global.queryQueue.bte_query_queue_by_api.process(ASYNC_BY_API_CONCURRENCY, async (job: BullJob) => {
      return await runBullJob(job, "asyncquery_v1_by_api");
    });
  }
}

if (!global.queryQueue.bte_query_queue_by_team && !Piscina.isWorkerThread) {
  getQueryQueue("bte_query_queue_by_team");
  if (global.queryQueue.bte_query_queue_by_team) {
    global.queryQueue.bte_query_queue_by_team.process(ASYNC_BY_TEAM_CONCURRENCY, async (job: BullJob) => {
      return await runBullJob(job, "asyncquery_v1_by_team");
    });
  }
}
