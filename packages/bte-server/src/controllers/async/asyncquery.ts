import axios, { AxiosError, AxiosResponse } from "axios";
import { customAlphabet } from "nanoid";
import * as utils from "../../utils/common";
import { redisClient } from "@biothings-explorer/utils";
import { LogEntry } from "@biothings-explorer/utils";
import lz4 from "lz4";
import { Readable } from "stream";
import chunker from "stream-chunker";
import { parser } from "stream-json";
import Assembler from "stream-json/Assembler";
import { Telemetry } from "@biothings-explorer/utils";
import ErrorHandler from "../../middlewares/error";
import { Request, Response, NextFunction } from "express";
import { Queue } from "bull";
import { QueueData, TrapiQueryGraph, TrapiResponse } from "@biothings-explorer/types";
import TRAPIQueryHandler from "@biothings-explorer/query_graph_handler";
import StatusError from "../../utils/errors/status_error";

export async function asyncquery(
  req: Request,
  res: Response,
  next: NextFunction,
  queueData: QueueData,
  queryQueue: Queue,
): Promise<void> {
  try {
    // Behavior for no redis
    if (!queryQueue) {
      res.setHeader("Content-Type", "application/json");
      res.status(503).end(JSON.stringify({ error: "Redis service is unavailable" }));
      return;
    }
    // Otherwise, add job to the queue
    const nanoid = customAlphabet("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", 10);
    let jobId = nanoid();

    if (queryQueue.name === "bte_query_queue_by_api") {
      jobId = `BA_${jobId}`;
    }
    if (queryQueue.name === "bte_query_queue_by_team") {
      jobId = `BT_${jobId}`;
    }
    const url = `${req.protocol}://${req.header("host")}/v1/asyncquery_status/${jobId}`;

    const job = await queryQueue.add(
      { ...queueData, url: url.replace("status", "response") },
      {
        jobId: jobId,
        timeout: parseInt(process.env.JOB_TIMEOUT ?? (1000 * 60 * 5).toString()),
        removeOnFail: {
          age: 24 * 60 * 60, // keep failed jobs for a day (in case user needs to review fail reason)
          count: 2000,
        },
        removeOnComplete: {
          age: 90 * 24 * 60 * 60, // keep completed jobs for 90 days
          count: 2000,
        },
      },
    );
    res.setHeader("Content-Type", "application/json");
    // return the job id so the user can check on it later
    res.end(JSON.stringify({ status: "Accepted", description: "Async query queued", job_id: job.id, job_url: url }));
  } catch (error) {
    next(error);
  }
}

async function storeQueryResponse(jobID: string, response: TrapiResponse | undefined, logLevel = null) {
  return await redisClient.client.usingLock([`asyncQueryResult:lock:${jobID}`], 600000, async () => {
    const defaultExpirySeconds = String(30 * 24 * 60 * 60); // 30 days
    const entries = [];
    if (typeof response === "undefined") {
      return;
    }
    // encode each property separately (accessible separately)
    await Promise.all(
      Object.entries(response).map(async ([key, value]) => {
        if (typeof value === "undefined") {
          return;
        }
        const input = Readable.from(JSON.stringify(value));
        await new Promise<void>(resolve => {
          let i = 0;
          input
            .pipe(chunker(10000000, { flush: true }))
            .on("data", async (chunk: Buffer) => {
              await redisClient.client.hsetTimeout(
                `asyncQueryResult:${jobID}:${key}`,
                String(i++),
                lz4.encode(chunk).toString("base64url"),
              );
            })
            .on("end", () => {
              resolve();
            });
        });
        await redisClient.client.expireTimeout(
          `asyncQueryResult:${jobID}:${key}`,
          process.env.ASYNC_COMPLETED_EXPIRE_TIME || defaultExpirySeconds,
        );
        entries.push(key);
      }),
    );
    // register all keys so they can be properly retrieved
    await redisClient.client.setTimeout(`asyncQueryResult:entries:${jobID}`, JSON.stringify(entries));
    await redisClient.client.expireTimeout(
      `asyncQueryResult:entries:${jobID}`,
      process.env.ASYNC_COMPLETED_EXPIRE_TIME || defaultExpirySeconds,
    );
    // remember log_level setting from original query
    await redisClient.client.setTimeout(`asyncQueryResult:logLevel:${jobID}`, JSON.stringify(logLevel));
    await redisClient.client.expireTimeout(
      `asyncQueryResult:logLevel:${jobID}`,
      process.env.ASYNC_COMPLETED_EXPIRE_TIME || defaultExpirySeconds,
    );
  });
}

export async function getQueryResponse(
  jobID: string,
  logLevel: string | null = null,
): Promise<TrapiResponse | StatusError | undefined> {
  return (await redisClient.client.usingLock(
    [`asyncQueryResult:lock:${jobID}`],
    600000,
    async (): Promise<TrapiResponse> => {
      const entries = await redisClient.client.getTimeout(`asyncQueryResult:entries:${jobID}`);
      if (!entries) {
        return null;
      }

      // JSON.parse because level may be string or null
      const originalLogLevel: string | null = JSON.parse(
        await redisClient.client.getTimeout(`asyncQueryResult:logLevel:${jobID}`),
      ) as string | null;

      const values = await Promise.all(
        (JSON.parse(entries) as string[]).map(async (key: string) => {
          const msgDecoded = Object.entries(await redisClient.client.hgetallTimeout(`asyncQueryResult:${jobID}:${key}`))
            .sort(([key1], [key2]) => parseInt(key1) - parseInt(key2))
            .map(([_key, val]) => lz4.decode(Buffer.from(val, "base64url")).toString(), "");
          const value = await new Promise(resolve => {
            const msgStream = Readable.from(msgDecoded);
            const pipeline = msgStream.pipe(parser());
            const asm = Assembler.connectTo(pipeline);
            asm.on("done", asm => resolve(asm.current));
          });
          return [key, value];
        }),
      );
      const response = Object.fromEntries(values) as TrapiResponse;
      if (response.logs && logLevel) {
        response.logs = utils.filterForLogLevel(response.logs, logLevel);
      } else if (response.logs && originalLogLevel) {
        response.logs = utils.filterForLogLevel(response.logs, originalLogLevel);
      }
      return response ? response : undefined;
    },
  )) as TrapiResponse | StatusError | undefined;
}

export async function asyncqueryResponse(
  handler: TRAPIQueryHandler,
  callback_url: string,
  jobID: string = null,
  jobURL: string = null,
  queryGraph: TrapiQueryGraph = null,
) {
  let response: TrapiResponse;
  let callback_response: AxiosResponse;
  try {
    await handler.query();
    response = handler.getResponse();
    if (jobURL) {
      response.logs.unshift(new LogEntry("INFO", null, `job results available at: ${jobURL}`).getLog());
    }
    if (jobID) {
      await storeQueryResponse(jobID, response, handler.options.logLevel);
    }
  } catch (e) {
    console.error(e);

    if (ErrorHandler.shouldHandleError(e as Error)) {
      Telemetry.captureException(e as Error);
    }

    //shape error > will be handled below
    response = {
      message: {
        query_graph: queryGraph,
        knowledge_graph: { nodes: {}, edges: {} },
        results: [],
      },
      status: "Failed",
      schema_version: global.SCHEMA_VERSION,
      biolink_version: global.BIOLINK_VERSION,
      workflow: [{ id: "lookup" }],
      description: (e as Error).toString(),
      trace: process.env.NODE_ENV === "production" ? undefined : (e as Error).stack,
      logs: handler.logs,
    };

    if (jobID) {
      await storeQueryResponse(jobID, response);
    }

    throw e;
  }

  if (callback_url) {
    if (!utils.stringIsAValidUrl(callback_url)) {
      return {
        response: "TRAPI Execution complete",
        status: 200,
        callback: "The callback url must be a valid url",
      };
    }
    try {
      const userAgent = `BTE/${process.env.NODE_ENV === "production" ? "prod" : "dev"} Node/${process.version} ${
        process.platform
      }`;
      callback_response = await axios.post(callback_url, JSON.stringify(response), {
        headers: {
          "Content-Type": "application/json",
          "User-Agent": userAgent,
        },
        timeout: 300000, // 5min
        maxBodyLength: 2 * 1000 * 1000 * 1000, // 2GB
      });
      //console.log(res)
    } catch (e) {
      return {
        response: "TRAPI Execution complete",
        status: (e as AxiosError).response?.status,
        callback: `Request failed, received code ${(e as AxiosError).response?.status}`,
      };
    }
  } else {
    return {
      response: "TRAPI Execution complete",
      status: 200,
      callback: "Callback url was not provided",
    };
  }
  return {
    response: "TRAPI Execution complete",
    status: callback_response?.status,
    callback: "Data sent to callback_url",
  };
}
