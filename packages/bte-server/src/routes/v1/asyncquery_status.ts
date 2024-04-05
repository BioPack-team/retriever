import path from "path";
import { LogEntry, redisClient } from "@biothings-explorer/utils";
import { getQueryQueue } from "../../controllers/async/asyncquery_queue";
import { getQueryResponse } from "../../controllers/async/asyncquery";
import * as utils from "../../utils/common";

import swaggerValidation from "../../middlewares/validate";
import { runTask, taskResponse, taskError } from "../../controllers/threading/threadHandler";
import Debug from "debug";
const debug = Debug("bte:biothings-explorer-trapi:async");
import { Express, NextFunction, Request, RequestHandler, Response } from "express";
import { TaskInfo, TrapiAsyncStatusResponse, TrapiLog, TrapiResponse } from "@biothings-explorer/types";
import { BteRoute } from "../../types";
import { Queue } from "bull";
import StatusError from "../../utils/errors/status_error";

class V1CheckQueryStatus implements BteRoute {
  setRoutes(app: Express) {
    app
      .route(["/v1/asyncquery_status/:id", "/v1/asyncquery_response/:id"])
      .get(swaggerValidation.validate, (async (req: Request, res: Response, next: NextFunction) => {
        try {
          const response = await runTask(req, res, path.parse(__filename).name, false);
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(response));
        } catch (err) {
          next(err);
        }
      }) as RequestHandler)
      .all(utils.methodNotAllowed);
  }

  async task(taskInfo: TaskInfo) {
    //logger.info("query /query endpoint")
    try {
      debug(`checking query status of job ${taskInfo.data.params.id}`);
      const jobID: string = taskInfo.data.params.id;
      let queryQueue: Queue;
      if (!redisClient.clientEnabled) {
        taskResponse({ error: "Redis service is unavailable" }, 503);
      }

      if (jobID.startsWith("BT_")) {
        queryQueue = getQueryQueue("bte_query_queue_by_team");
      } else if (jobID.startsWith("BA_")) {
        queryQueue = getQueryQueue("bte_query_queue_by_api");
      } else {
        queryQueue = getQueryQueue("bte_query_queue");
      }

      const job = await queryQueue.getJob(jobID);
      if (job === null) {
        return taskResponse(null, 404);
      }

      await queryQueue.isReady();

      const state = await job.getState();
      const progress = job.progress() as number;

      const bullLogs = await queryQueue.getJobLogs(jobID);
      let logs: TrapiLog[] = bullLogs.logs.map(log => JSON.parse(log) as TrapiLog);
      const originalLogLevel = JSON.parse(await redisClient.client.getTimeout(`asyncQueryResult:logLevel:${jobID}`)) as
        | string
        | null;
      logs = utils.filterForLogLevel(logs, taskInfo.data.options.logLevel ?? originalLogLevel);

      // convert to TRAPI states
      const [status, description] = {
        completed: ["Completed", "The query has finished executing."],
        failed: ["Failed", job.failedReason],
        delayed: ["Queued", "The query is queued, but has been delayed."],
        active: ["Running", "The query is currently being processed."],
        waiting: ["Queued", "The query is waiting in the queue."],
        paused: ["Queued", "The query is queued, but the queue is temporarily paused."],
        stuck: ["Failed", "The query is stuck (if you see this, raise an issue)."],
        null: ["Failed", "The query status is unknown, presumed failed (if you see this, raise an issue)."],
      }[state];

      if (status === "Failed" && !taskInfo.data.endpoint.includes("asyncquery_response")) {
        if (description.includes("Promise timed out")) {
          // something might break when calculating process.env.JOB_TIMEOUT so wrap it in try catch
          try {
            return taskResponse({
              job_id: jobID,
              status,
              description: `Job was stopped after exceeding time limit of ${
                parseInt(process.env.JOB_TIMEOUT ?? (1000 * 60 * 5).toString()) / 1000
              }s`,
              logs,
            });
          } catch (e) {
            return taskResponse({ job_id: jobID, status, description, logs });
          }
        }
        return taskResponse({ job_id: jobID, status, description, logs });
      }

      // If done, just give response if using asyncquery_response
      if ((state === "completed" || state === "failed") && taskInfo.data.endpoint.includes("asyncquery_response")) {
        const storedResponse = await getQueryResponse(jobID, taskInfo.data.options.logLevel);

        if (storedResponse && !("logs" in storedResponse) && logs) {
          (storedResponse as Partial<TrapiAsyncStatusResponse>).logs = logs;
        }

        let returnValue: TrapiResponse | TrapiAsyncStatusResponse;

        if (!storedResponse) {
          returnValue = {
            status: "Error",
            description: "Response expired. Responses are kept 30 days.",
            logs: [],
          };
        } else if ("statusCode" in storedResponse) {
          returnValue = {
            status: `Error ${storedResponse.name}`,
            description: storedResponse.message,
            logs: storedResponse.logs,
          };
        } else {
          returnValue = storedResponse;
        }

        return taskResponse(returnValue, (storedResponse as StatusError)?.statusCode || 200);
      }

      // Otherwise respond for asyncquery_status
      taskResponse({ job_id: jobID, status, progress, description, response_url: job.data.url, logs }, 200);
    } catch (error) {
      taskError(error as Error);
    }
  }
}

export default new V1CheckQueryStatus();
