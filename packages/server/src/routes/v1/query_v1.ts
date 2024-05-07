import path from "path";
import apiList from "../../config/api_list";
import TRAPIQueryHandler from "@retriever/query_graph_handler";
import swaggerValidation from "../../middlewares/validate";
const smartAPIPath = path.resolve(
  __dirname,
  process.env.STATIC_PATH ? `${process.env.STATIC_PATH}/data/smartapi_specs.json` : "../../../data/smartapi_specs.json",
);
const predicatesPath = path.resolve(
  __dirname,
  process.env.STATIC_PATH ? `${process.env.STATIC_PATH}/data/predicates.json` : "../../../data/predicates.json",
);
import * as utils from "../../utils/common";
import { runTask, taskResponse, taskError } from "../../controllers/threading/threadHandler";
import { Express, NextFunction, Request, RequestHandler, Response } from "express";
import { TrapiResponse } from "@retriever/types";
import { BteRoute } from "../../types";
import { TaskInfo } from "@retriever/types";

class V1Query implements BteRoute {
  setRoutes(app: Express) {
    app
      .route("/v1/query")
      .post(swaggerValidation.validate, (async (req: Request, res: Response, next: NextFunction) => {
        try {
          const response: TrapiResponse = await runTask(req, res, path.parse(__filename).name);
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(response));
        } catch (err) {
          next(err);
        }
      }) as RequestHandler)
      .all(utils.methodNotAllowed);
  }

  async task(taskInfo: TaskInfo) {
    const queryGraph = taskInfo.data.queryGraph,
      workflow = taskInfo.data.workflow,
      options = { ...taskInfo.data.options, schema: await utils.getSchema() };

    try {
      utils.validateWorkflow(workflow);

      const handler = new TRAPIQueryHandler({ apiList, ...options }, smartAPIPath, predicatesPath);
      handler.setQueryGraph(queryGraph);
      await handler.query();

      const response = handler.getResponse();
      response.logs = utils.filterForLogLevel(response.logs, options.logLevel);
      return taskResponse(response);
    } catch (error) {
      taskError(error as Error);
    }
  }
}

export default new V1Query();
