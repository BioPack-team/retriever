import Piscina from "piscina";
import Queue from "bull";
import { TrapiQuery, TrapiResponse } from "@biothings-explorer/types";
import { Express, Request } from "express";
import { TaskInfo, TaskData } from "@biothings-explorer/types";

export interface ThreadPool {
  sync: Piscina;
  async: Piscina;
  misc: Piscina;
}

export interface AsyncResultSummary {
  response: string;
  status: number;
  callback: string;
}

export interface TaskFunction {
  (job: TaskInfo): Promise<unknown>; // Should be TrapiResponse else void
}

export interface BullJob extends Queue.Job {
  data: TaskData;
}

export interface AsyncQueryData {}

export interface BteRoute {
  setRoutes: {
    (app: Express): void;
  };
  task: TaskFunction;
}

export interface SmartApiOverrideConfig {
  only_overrides: boolean;
}

export interface SmartApiOverrideList {
  [smartapi_id: string]: string;
}

export interface SmartApiOverrides {
  config: SmartApiOverrideConfig;
  apis: SmartApiOverrideList;
}

/* 
Piscina doesn't bother to type various values
so here are the types we use as a stop-gap
*/
export interface PiscinaWaitTime {
  p99: number;
}
