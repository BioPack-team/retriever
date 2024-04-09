import MetaKG from "./metakg";
import FrontPage from "./frontpage";
import Performance from "./performance";
import V1QueryByAPI from "./v1/query_v1_by_api";
import V1QueryByTeam from "./v1/query_v1_by_team";
import V1Query from "./v1/query_v1";
import V1AsyncQuery from "./v1/asyncquery_v1";
import V1AsyncQueryByAPI from "./v1/asyncquery_v1_by_api";
import V1AsyncQueryByTeam from "./v1/asyncquery_v1_by_team";
import V1CheckQueryStatus from "./v1/asyncquery_status";
import V1MetaKG from "./v1/meta_knowledge_graph_v1";
import V1MetaKGByAPI from "./v1/meta_knowledge_graph_v1_by_api";
import V1MetaKGByTeam from "./v1/meta_knowledge_graph_v1_by_team";
import ErrorHandler from "../middlewares/error";
import LoggingHandler from "../middlewares/logging";
import routesBullBoardPage from "./bullboard";
import { Express } from "express";
import { TaskFunction } from "../types";
import { TaskInfo } from "@biothings-explorer/types";

export function setRoutes(app: Express): void {
  MetaKG.setRoutes(app);
  V1MetaKG.setRoutes(app);
  V1AsyncQuery.setRoutes(app);
  V1AsyncQueryByAPI.setRoutes(app);
  V1AsyncQueryByTeam.setRoutes(app);
  V1MetaKGByAPI.setRoutes(app);
  V1MetaKGByTeam.setRoutes(app);
  V1CheckQueryStatus.setRoutes(app);
  routesBullBoardPage.setRoutes(app);
  Performance.setRoutes(app);
  V1QueryByAPI.setRoutes(app);
  V1QueryByTeam.setRoutes(app);
  LoggingHandler.setRoutes(app);
  V1Query.setRoutes(app);
  ErrorHandler.setRoutes(app);
  FrontPage.setRoutes(app);
}

interface TaskByRoute {
  [route: string]: TaskFunction;
}

export const tasks: TaskByRoute = {
  query_v1: (taskInfo: TaskInfo) => V1Query.task(taskInfo),
  query_v1_by_api: (taskInfo: TaskInfo) => V1QueryByAPI.task(taskInfo),
  query_v1_by_team: (taskInfo: TaskInfo) => V1QueryByTeam.task(taskInfo),
  asyncquery_status: (taskInfo: TaskInfo) => V1CheckQueryStatus.task(taskInfo),
  // async processor uses thread
  asyncquery_v1: (taskInfo: TaskInfo) => V1AsyncQuery.task(taskInfo),
  asyncquery_v1_by_api: (taskInfo: TaskInfo) => V1AsyncQueryByAPI.task(taskInfo),
  asyncquery_v1_by_team: (taskInfo: TaskInfo) => V1AsyncQueryByTeam.task(taskInfo),
  // Not threaded due to being lightweight/speed being higher priority
  // performance: (taskInfo: TaskInfo) => Performance.task(taskInfo),
  // metakg: (taskInfo: TaskInfo) => MetaKG.task(taskInfo),
  // meta_knowledge_graph_v1: (taskInfo: TaskInfo) => V1MetaKG.task(taskInfo),
  // meta_knowledge_graph_v1_by_api: (taskInfo: TaskInfo) => V1MetaKGByAPI.task(taskInfo),
  // meta_knowledge_graph_v1_by_team: (taskInfo: TaskInfo) => V1MetaKGByTeam.task(taskInfo),
};
