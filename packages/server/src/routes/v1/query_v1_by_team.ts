import path from "path";
import apiList from "../../config/api_list";
import TRAPIQueryHandler from "@biothings-explorer/query_graph_handler";
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
import { TaskInfo } from "@biothings-explorer/types";

class V1QueryByTeam {
  setRoutes(app: Express) {
    app
      .route("/v1/team/:team_name/query")
      .post(swaggerValidation.validate, (async (req: Request, res: Response, next: NextFunction) => {
        try {
          const response = await runTask(req, res, path.parse(__filename).name);
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(response));
        } catch (error) {
          next(error);
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

      const handler = new TRAPIQueryHandler(
        {
          apiList,
          ...options,
        },
        smartAPIPath,
        predicatesPath,
        false,
      );
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

export default new V1QueryByTeam();
