import { getQueryQueue } from "../controllers/async/asyncquery_queue";
import { createBullBoard } from "@bull-board/api";
import { BullAdapter } from "@bull-board/api/bullAdapter";
import { ExpressAdapter } from "@bull-board/express";
import Debug from "debug";
const debug = Debug("bte:biothings-explorer-trapi:bullboard");
import { redisClient } from "@biothings-explorer/utils";
import { Express, NextFunction, Request, Response } from "express";

class BullBoardPage {
  setRoutes(app: Express) {
    debug("Initializing Bull Dashboard");
    if (!redisClient.clientEnabled || process.env.INTERNAL_DISABLE_REDIS === "true") {
      debug("Redis is not enabled, disabling Bull Dashboard");
      app.use("/queues", async (_req: Request, res: Response, _next: NextFunction) => {
        res
          .status(503)
          .set("Retry-After", "600")
          .set("Content-Type", "application/json")
          .end(JSON.stringify({ error: "Redis service is unavailable, so async job queuing is disabled." }));
      });
      return;
    }
    const queues = {
      "/v1/asynquery": getQueryQueue("bte_query_queue"),
      "/v1/smartapi/{smartapi_id}/asyncquery": getQueryQueue("bte_query_queue_by_api"),
      "/v1/team/{team_name}/asyncquery": getQueryQueue("bte_query_queue_by_team"),
      "/v1/query": getQueryQueue("bte_sync_query_queue"),
    };

    const serverAdapter = new ExpressAdapter();
    serverAdapter.setBasePath("/queues");

    const instance = {
      prod: "Prod",
      test: "Test",
      ci: "Staging",
      dev: "Dev",
    }[process.env.INSTANCE_ENV ?? "dev"];

    createBullBoard({
      queues: Object.entries(queues).map(([name, queue]) => {
        const adapter = new BullAdapter(queue, {
          readOnlyMode: true,
          description: name,
        });
        adapter.setFormatter(
          "name",
          job => `${name === "/v1/query" ? "Synchronous" : "Asynchronous"} Request #${job.id}`,
        );
        adapter.setFormatter("data", ({ abortController, threadId, route, ...rest }) => rest);
        return adapter;
      }),
      serverAdapter,
      options: {
        uiConfig: {
          boardTitle: `BTE ${instance}`,
        },
      },
    });

    app.use("/queues", serverAdapter.getRouter());
  }
}

export default new BullBoardPage();
